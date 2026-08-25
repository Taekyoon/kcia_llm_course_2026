"""work/notebook/ 의 노트북 사본을 2026년 스택으로 마이그레이션한다.

    uv run python tools/migrate_notebooks.py           # 적용
    uv run python tools/migrate_notebooks.py --dry-run # 미리보기

설계 원칙
---------
1. **원본은 절대 건드리지 않는다.** `notebook/` 은 읽기만 하고, 결과는 `work/notebook/`
   에 쓴다 (CLAUDE.md §1).
2. **매번 원본에서 다시 만든다.** 사본을 제자리에서 고치면 규칙의 `new` 가 `old` 를
   포함할 때 재실행마다 중복 적용된다(실제로 겪었다). 항상 pristine 원본에서
   시작하면 이 문제가 원천적으로 사라지고, 몇 번을 돌려도 결과가 같다.
3. **명시적 문자열 치환만 한다.** 정규식으로 넓게 훑지 않는다. 대상 문자열을 정확히
   적고, 못 찾으면 **오류를 내고 중단**한다. 조용히 넘어가면 마이그레이션 누락을
   눈치채지 못한 채 "완료"로 착각하게 된다.
4. 전역 치환은 **예상 출현 횟수를 명시**하고 실제와 다르면 중단한다.
5. 변경 근거는 각 규칙의 `why` 에 남긴다. 리포트에 그대로 출력된다.

근거: review/02_문제점.md (D축) · verify/결과.md (실측)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebook"          # 원본 — 읽기 전용
WORK = ROOT / "work" / "notebook"  # 산출물 — 매번 새로 씀

# 확정된 모델 구성 (review/03_개선계획.md 필-1)
ENCODER = "jhu-clsp/mmBERT-base"          # 분류·NER — MIT, 인코더
TRAIN_LM = "Qwen/Qwen3-0.6B-Base"         # SFT·DPO·GRPO — Apache-2.0
SERVE_LM = "Qwen/Qwen3-4B-Instruct-2507"  # vLLM 서빙 — Apache-2.0, text-only

OLD_HCX = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"
OLD_EXAONE = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"


@dataclass
class Rule:
    """한 셀 안에서의 정확한 문자열 치환."""

    cell: int          # 1-based 셀 번호
    old: str
    new: str
    why: str
    optional: bool = False   # True 면 못 찾아도 경고만


@dataclass
class GlobalRule:
    """노트북 전체에서 반복되는 문자열 치환.

    모델 ID처럼 수십 곳에 흩어진 것을 셀 단위로 적는 것은 비현실적이다.
    대신 **예상 출현 횟수를 명시**하고, 실제와 다르면 중단한다.
    그래야 "몇 군데는 놓쳤는데 성공한 줄 아는" 상황을 막을 수 있다.
    """

    old: str
    new: str
    why: str
    expect: int        # 예상 출현 횟수. 다르면 중단


@dataclass
class Migration:
    path: str
    rules: list[Rule] = field(default_factory=list)
    globals_: list[GlobalRule] = field(default_factory=list)


# 공통 조각 -----------------------------------------------------------------

PIP_TRAIN = """# 2026 스택. datasets==3.5.1 핀을 제거했다 — 스크립트 데이터셋 지원이
# datasets 4.0 에서 사라져 핀을 유지하면 오히려 최신 데이터셋을 못 읽는다.
%pip install -q -U transformers datasets evaluate accelerate"""

WHY_PIP = "!pip/%pip/pip 혼용 정리 + datasets 3.5.1 핀 제거 (D-8, D-0)"
WHY_PROC = "transformers 5: Trainer(tokenizer=) 제거 → processing_class= (D-4 실측 확정)"
WHY_TOKATTR = "transformers 5: trainer.tokenizer 속성 제거 → processing_class (D-4 실측 확정)"
WHY_PEFT = "TRL 현행 권장: get_peft_model() 선감싸기 → peft_config= 인자 전달 (D-4)"


MIGRATIONS: list[Migration] = [
    # =====================================================================
    Migration("1일차/HPC_Classification실습.ipynb", [
        Rule(1,
             "!pip install datasets==3.5.1\n!pip install evaluate\n!pip install transformers[torch]",
             PIP_TRAIN, WHY_PIP),
        Rule(3,
             f'tokenizer = AutoTokenizer.from_pretrained("{OLD_HCX}", do_lower_case=False)',
             f'# 인코더 모델로 전환했다. 바로 앞 슬라이드(1일차 p54-58)에서 "Encoder는 BERT,\n'
             f'# MLM+NSP" 를 배우는데 실습은 decoder-only 에 분류 헤드를 붙이고 있어\n'
             f'# 배운 것과 실습이 어긋나 있었다.\n'
             f'tokenizer = AutoTokenizer.from_pretrained("{ENCODER}")',
             "decoder-only → 인코더 전환. 커리큘럼 ⑤ 항목과 실습 연결 (필-1c)"),
        Rule(2,
             "from transformers import DataCollatorWithPadding\n\nimport matplotlib.pyplot as plt",
             "from transformers import DataCollatorWithPadding",
             "matplotlib 은 import 만 하고 쓰지 않는다(plt. 사용 없음). "
             "기본 환경에 없어 ModuleNotFoundError 로 죽는다"),
        Rule(8,
             'dataset = load_dataset("nsmc",  trust_remote_code=True)',
             '# datasets 5.x 는 로딩 스크립트를 지원하지 않는다.\n'
             '#   RuntimeError: Dataset scripts are no longer supported, but found nsmc.py\n'
             '# parquet 변환 리비전을 직접 지정한다.\n'
             'dataset = load_dataset("e9t/nsmc", revision="refs/convert/parquet")',
             "nsmc 스크립트형 → parquet 리비전 (D-1 실측 확정)"),
        Rule(17,
             f'model = AutoModelForSequenceClassification.from_pretrained(\n'
             f'    "{OLD_HCX}", num_labels=2, id2label=id2label, label2id=label2id, torch_dtype="auto",\n'
             f'    device_map="auto",\n'
             f')',
             f'# 307M 인코더라 device_map="auto" 가 필요 없다.\n'
             f'# Trainer 와 함께 쓰면 오히려 문제가 될 수 있어 제거했다.\n'
             f'model = AutoModelForSequenceClassification.from_pretrained(\n'
             f'    "{ENCODER}", num_labels=2, id2label=id2label, label2id=label2id,\n'
             f')',
             "인코더 전환 + 불필요한 device_map/torch_dtype 제거"),
        Rule(19, "    tokenizer=tokenizer,", "    processing_class=tokenizer,", WHY_PROC),
        Rule(20,
             'classifier = pipeline("sentiment-analysis", model="./nsmc_classifier/checkpoint-938")',
             '# checkpoint-938 을 하드코딩하면 배치·에폭·데이터 크기가 조금만 달라져도 깨진다.\n'
             '# 방금 학습한 trainer 에서 경로를 받아온다.\n'
             'best = trainer.state.best_model_checkpoint or training_args.output_dir\n'
             'classifier = pipeline("sentiment-analysis", model=best, tokenizer=tokenizer)',
             "하드코딩 체크포인트 경로 제거 — 재현 불가 원인 (D-7)"),
    ]),

    # =====================================================================
    Migration("1일차/HPC_NER실습.ipynb", [
        Rule(1,
             "!pip install datasets==3.5.0\n!pip install evaluate\n!pip install transformers[torch]\n!pip install seqeval",
             PIP_TRAIN + "\n%pip install -q seqeval", WHY_PIP),
        Rule(3,
             f'tokenizer = AutoTokenizer.from_pretrained("{OLD_HCX}", do_lower_case=False)',
             f'# 분류 실습과 동일하게 인코더로 전환했다.\n'
             f'tokenizer = AutoTokenizer.from_pretrained("{ENCODER}")',
             "decoder-only → 인코더 전환 (필-1c)"),
        Rule(2,
             "from transformers import DataCollatorWithPadding\n\nimport matplotlib.pyplot as plt",
             "from transformers import DataCollatorWithPadding",
             "matplotlib 은 import 만 하고 쓰지 않는다. 기본 환경에 없어 죽는다"),
        Rule(8,
             'dataset = load_dataset("kor_ner")',
             '# kor_ner 는 로딩 스크립트 방식이라 datasets 5.x 에서 읽지 못한다.\n'
             '#   RuntimeError: Dataset scripts are no longer supported, but found kor_ner.py\n'
             '# KLUE-NER 로 교체한다. 컬럼명(tokens/ner_tags)이 같아 이후 코드가 그대로 동작한다.\n'
             '# 라벨 13종: B/I-{DT,LC,OG,PS,QT,TI} + O\n'
             'dataset = load_dataset("klue/klue", "ner")',
             "kor_ner 스크립트형 → klue/klue ner (D-1 실측 확정)"),
        Rule(10,
             'dataset["test"][0]',
             '# KLUE 는 train/validation 구성이다 (test 스플릿 없음).\n'
             'dataset["validation"][0]',
             "klue/klue ner 은 test 스플릿이 없다 — 교체에 따른 연쇄 수정"),
        Rule(13,
             '    tokenized_inputs["labels"] = new_labels.replace()',
             '    tokenized_inputs["labels"] = new_labels',
             "원래부터 있던 버그. list 에 .replace() 호출 → AttributeError (D-6). "
             "슬라이드 1일차 p40 에도 같은 버그가 있다"),
        Rule(20,
             f'model = AutoModelForTokenClassification.from_pretrained(\n'
             f'    "{OLD_HCX}", id2label=id2label, label2id=label2id\n'
             f')',
             f'model = AutoModelForTokenClassification.from_pretrained(\n'
             f'    "{ENCODER}", id2label=id2label, label2id=label2id\n'
             f')',
             "인코더 전환 (필-1c)"),
        Rule(21, '    eval_dataset=tokenized_dataset["test"],',
             '    eval_dataset=tokenized_dataset["validation"],',
             "klue/klue ner 은 test 스플릿이 없다"),
        Rule(21, "    tokenizer=tokenizer,", "    processing_class=tokenizer,", WHY_PROC),
        Rule(22,
             '# Replace this with your own checkpoint\nmodel_checkpoint = "./ner_seq_classifier/checkpoint-183"',
             '# checkpoint-183 하드코딩은 재현되지 않는다. 학습 결과에서 받아온다.\n'
             'model_checkpoint = trainer.state.best_model_checkpoint or training_args.output_dir',
             "하드코딩 체크포인트 경로 제거 (D-7)"),
    ]),

    # =====================================================================
    Migration("2일차/HPC_SFT실습.ipynb", [
        Rule(2,
             "%pip install -q transformers[torch] datasets\n%pip install -q trl peft",
             "%pip install -q -U transformers datasets trl peft accelerate", WHY_PIP),
        Rule(3, "pip install -U transformers[torch]",
             "# (셀 2 에서 한 번에 설치하므로 삭제)", WHY_PIP),
        Rule(5,
             'raw_datasets = load_dataset("beomi/KoAlpaca-v1.1a")',
             '# KoAlpaca 는 CC BY-NC 4.0 (GitHub DATA_LICENSE) 이라 상업 강의에 쓸 수 없다.\n'
             '# HF 페이지에는 라이선스 태그가 없어 놓치기 쉽다.\n'
             '# Apache-2.0 인 KULLM 으로 교체한다.\n'
             'raw_datasets = load_dataset("nlpai-lab/kullm-v2")',
             "라이선스 저촉 — CC BY-NC 4.0 → Apache-2.0 (F-1)"),
        Rule(7,
             'def build_messages(example):\n'
             '    messages = [\n'
             '        {"role": "user", "content": example["instruction"]},\n'
             '        {"role": "assistant", "content": example["output"]}\n'
             '    ]',
             'def build_messages(example):\n'
             '    # KULLM 은 Alpaca 계열이라 instruction 외에 input 컬럼이 있다.\n'
             '    # input 이 있는 행을 합쳐주지 않으면 "다음 문장을 요약하세요" 처럼\n'
             '    # 지시만 남고 정작 대상이 빠진 학습 예시가 된다.\n'
             '    user = example["instruction"]\n'
             '    if example.get("input"):\n'
             '        user = f\'{user}\\n\\n{example["input"]}\'\n\n'
             '    messages = [\n'
             '        {"role": "user", "content": user},\n'
             '        {"role": "assistant", "content": example["output"]}\n'
             '    ]',
             "kullm-v2 는 input 컬럼이 있다(실측). 병합하지 않으면 학습 예시가 깨진다"),
        Rule(20, "    max_seq_length=tokenizer.model_max_length,",
             "    max_length=tokenizer.model_max_length,",
             "TRL: SFTConfig.max_seq_length → max_length 개명 (D-4 실측 확정)"),
        Rule(20, "    overwrite_output_dir=True,\n", "",
             "transformers 5 에서 overwrite_output_dir 제거됨 → TypeError (실행 검증에서 확인)"),
        Rule(23, "trainer.tokenizer.save_pretrained(output_dir)",
             "trainer.processing_class.save_pretrained(output_dir)", WHY_TOKATTR),
    ]),

    # =====================================================================
    Migration("2일차/HPC_DPO실습.ipynb", [
        Rule(2, "!pip install -q transformers[torch] datasets==3.5.1",
             "%pip install -q -U transformers datasets accelerate", WHY_PIP),
        Rule(3, "!pip install -q trl peft", "%pip install -q -U trl peft", WHY_PIP),
        Rule(15, "    overwrite_output_dir=True,\n", "",
             "transformers 5 에서 overwrite_output_dir 제거됨 → TypeError (실행 검증에서 확인)"),
        Rule(18, "trainer.tokenizer.save_pretrained(output_dir)",
             "trainer.processing_class.save_pretrained(output_dir)", WHY_TOKATTR),
    ]),

    # =====================================================================
    Migration("2일차/HPC_GRPO실습.ipynb", [
        Rule(2, "!pip install transformers[torch] datasets==3.5.1",
             "%pip install -q -U transformers datasets accelerate", WHY_PIP),
        Rule(3, "!pip install -q trl peft math_verify",
             "%pip install -q -U trl peft math-verify", WHY_PIP),
        Rule(9, 'model_id = "Qwen/Qwen2.5-0.5B-Instruct"',
             f'# GRPO 만 구세대 Qwen2.5 를 쓰고 있어 SFT/DPO 와 계열이 갈렸다.\n'
             f'# Qwen3 로 통일한다.\n'
             f'model_id = "{TRAIN_LM}"',
             "세대 혼재 제거 — Qwen2.5 → Qwen3 계열 통일 (필-1a)"),
        Rule(18, "    max_prompt_length=128,\n", "",
             "TRL 1.x 에서 GRPOConfig.max_prompt_length 제거됨 → TypeError (실행 검증에서 확인)"),
        Rule(21, "trainer.tokenizer.save_pretrained(output_dir)",
             "trainer.processing_class.save_pretrained(output_dir)", WHY_TOKATTR),
    ]),

    # =====================================================================
    Migration("2일차/HPC_퓨샷실습.ipynb", [
        Rule(4, "pip install openai vllm datasets==3.5.1",
             "# vLLM 은 별도 venv 에서 서버로 띄운다 (verify/02_vllm_venv.sh 참조).\n"
             "# 이 노트북은 HTTP 로 붙기만 하므로 openai 클라이언트만 있으면 된다.\n"
             "%pip install -q -U openai datasets", WHY_PIP),
        # 주의: 전역 규칙이 먼저 돌아 모델 ID가 이미 교체된 상태다. old 는 교체 후 문자열을 쓴다.
        Rule(6,
             f"! nohup python -m vllm.entrypoints.openai.api_server --model {SERVE_LM} &",
             f'# 구 실행 방식(python -m vllm.entrypoints.openai.api_server)은 deprecated 다.\n'
             f'# 그리고 vLLM 은 torch 를 하드핀하므로 학습 환경과 같은 venv 에 깔면 안 된다.\n'
             f'# 아래는 별도 터미널에서 실행한다:\n'
             f'#   source /opt/vllm-env/bin/activate\n'
             f'#   nohup vllm serve {SERVE_LM} --port 8000 > vllm.log 2>&1 &\n'
             f'# 기동 확인:\n'
             f'!curl -s http://localhost:8000/v1/models || echo "서버가 아직 준비되지 않았습니다"',
             "api_server deprecated → vllm serve, 별도 venv 분리 (D-2, D-10)"),
        Rule(11, 'dataset = load_dataset("e9t/nsmc")',
             '# e9t/nsmc 는 로딩 스크립트 방식이라 datasets 5.x 에서 읽지 못한다.\n'
             'dataset = load_dataset("e9t/nsmc", revision="refs/convert/parquet")',
             "nsmc 스크립트형 → parquet 리비전 (D-1 실측 확정)"),
    ], [
        GlobalRule(OLD_EXAONE, SERVE_LM,
                   "EXAONE 은 NC 라이선스라 상업 강의 사용 불가 (F-1). Qwen3 로 교체", 18),
        GlobalRule('extra_body={"guided_choice": ["긍정", "부정"]},',
                   'extra_body={"structured_outputs": {"choice": ["긍정", "부정"]}},',
                   "vLLM v0.12 에서 guided_choice 제거 → structured_outputs (D-2)", 3),
        GlobalRule('extra_body={"guided_json": json_schema},',
                   'response_format={"type": "json_schema",\n'
                   '                     "json_schema": {"name": "answer", "schema": json_schema}},',
                   "vLLM v0.12 에서 guided_json 제거 → response_format(OpenAI 표준) (D-2)", 2),
    ]),

    # =====================================================================
    Migration("3일차/HPC_Amazon요약실습.ipynb", [
        Rule(2, "pip install openai vllm datasets==3.5.1",
             "# vLLM 은 별도 venv 에서 서버로 띄운다 (verify/02_vllm_venv.sh 참조).\n"
             "%pip install -q -U openai datasets pydantic", WHY_PIP),
        # 셀 5 (Taekyoon/test_amazon) 는 손대지 않는다.
        # HF 에 라이선스 표기가 없어 F-1 에서 교체 대상으로 올렸으나,
        # 강사 본인 계정의 데이터셋이라 이용 권한 판단은 소유자 몫이다.
        # 소유자가 리스크를 인지하고 현행 유지를 결정했다 (2026-08-25).
        # 주의: 전역 규칙이 먼저 돌아 모델 ID가 이미 교체된 상태다.
        Rule(9,
             f"! nohup python -m vllm.entrypoints.openai.api_server --model {SERVE_LM} &",
             f'# 별도 터미널에서 실행한다:\n'
             f'#   source /opt/vllm-env/bin/activate\n'
             f'#   nohup vllm serve {SERVE_LM} --port 8000 > vllm.log 2>&1 &\n'
             f'!curl -s http://localhost:8000/v1/models || echo "서버가 아직 준비되지 않았습니다"',
             "api_server deprecated → vllm serve, 별도 venv 분리 (D-2, D-10)"),
    ], [
        GlobalRule(OLD_EXAONE, SERVE_LM,
                   "EXAONE NC 라이선스 → Qwen3 교체 (F-1)", 2),
        GlobalRule("completion = client.beta.chat.completions.parse(",
                   "completion = client.chat.completions.create(",
                   "openai SDK 2.x 에서 .beta 네임스페이스 이동. 표준 create + response_format 으로 통일", 6),
        GlobalRule('extra_body={"guided_json": feature_type_schema},',
                   'response_format={"type": "json_schema", "json_schema":\n'
                   '                  {"name": "feature_type_list", "schema": feature_type_schema}},',
                   "guided_json 제거 → response_format (Step 1)", 1),
        GlobalRule('extra_body={"guided_json": subsectoin_schema},',
                   'response_format={"type": "json_schema", "json_schema":\n'
                   '                  {"name": "subsection_list", "schema": subsectoin_schema}},',
                   "guided_json 제거 → response_format (Step 2)", 1),
        GlobalRule('extra_body={"guided_json": extracted_feature_schema},',
                   'response_format={"type": "json_schema", "json_schema":\n'
                   '                  {"name": "extracted_features", "schema": extracted_feature_schema}},',
                   "guided_json 제거 → response_format (Step 3)", 1),
        GlobalRule('extra_body={"guided_json": consumer_category_schema},',
                   'response_format={"type": "json_schema", "json_schema":\n'
                   '                  {"name": "consumer_categories", "schema": consumer_category_schema}},',
                   "guided_json 제거 → response_format (Step 4)", 1),
        GlobalRule('extra_body={"guided_json": summary_schema},',
                   'response_format={"type": "json_schema", "json_schema":\n'
                   '                  {"name": "summary", "schema": summary_schema}},',
                   "guided_json 제거 → response_format (Step 6·7)", 2),
    ]),

    # =====================================================================
    Migration("3일차/HPC_BM25_RAG실습.ipynb", [
        Rule(3,
             "%pip install llama-index\n%pip install llama-index-retrievers-bm25\n"
             "%pip install llama-index-llms-vllm\n%pip install datasets\n%pip install vllm",
             "# vLLM 은 별도 venv 에서 서버로 띄우고 여기서는 HTTP 로 붙는다.\n"
             "# llama-index-llms-vllm(in-process) 대신 openai-like 를 쓴다.\n"
             "# 주의: %pip 매직에서 백슬래시 줄바꿈은 불안정하다. 한 줄로 쓴다.\n"
             "%pip install -q -U llama-index llama-index-retrievers-bm25 "
             "llama-index-llms-openai-like datasets PyStemmer",
             "in-process vLLM → HTTP 접속. torch 충돌 회피 (D-10)"),
        # 이 노트북에는 전역 규칙이 없으므로 old 는 원본 문자열 그대로다.
        Rule(4,
             'from llama_index.core import Settings\n'
             'from llama_index.llms.vllm import Vllm\n\n'
             'Settings.llm = Vllm(\n'
             "    dtype='float16',\n"
             f"    model='{OLD_EXAONE}'\n"
             ')',
             'from llama_index.core import Settings\n'
             'from llama_index.llms.openai_like import OpenAILike\n\n'
             '# 별도 venv 에서 띄운 vLLM 서버에 HTTP 로 붙는다.\n'
             '#   source /opt/vllm-env/bin/activate\n'
             f'#   nohup vllm serve {SERVE_LM} --port 8000 > vllm.log 2>&1 &\n'
             'Settings.llm = OpenAILike(\n'
             f'    model="{SERVE_LM}",\n'
             '    api_base="http://localhost:8000/v1",\n'
             '    api_key="EMPTY",\n'
             '    is_chat_model=True,\n'
             ')\n\n'
             '# BM25 는 어휘 기반 검색이라 임베딩 모델이 필요 없다.\n'
             '# 다만 LlamaIndex 기본값이 OpenAI 임베딩이라, 지정하지 않으면\n'
             '# 다른 경로에서 OpenAI API 키를 요구할 수 있다. 명시적으로 꺼둔다.\n'
             'Settings.embed_model = None',
             "EXAONE NC 라이선스 + in-process vLLM 제거 → OpenAILike HTTP 접속 (F-1, D-10)"),
        Rule(7,
             "dataset = load_dataset('heegyu/kowikitext', trust_remote_code=True, split='train[:1000]')",
             "# heegyu/kowikitext 는 로딩 스크립트 방식이라 datasets 5.x 에서 읽지 못한다.\n"
             "#   RuntimeError: Dataset scripts are no longer supported, but found kowikitext.py\n"
             "# data_files= 로 parquet 를 직접 지정해도 스크립트를 먼저 감지해 실패한다(실측 확인).\n"
             "# 위키미디어 공식 배포판으로 교체한다.\n"
             "# 컬럼(title/text)이 같아 이후 셀들이 그대로 동작한다.\n"
             "dataset = load_dataset('wikimedia/wikipedia', '20231101.ko', split='train[:1000]')",
             "kowikitext 스크립트형 → wikimedia/wikipedia (D-1). parquet 직접지정 실패 실측 후 교체"),
        # 셀 전체를 한 번에 교체한다. 두 규칙으로 쪼개면 두 번째 규칙의 new 가 old 를
        # 포함하게 되어 재실행마다 update_prompts() 호출이 중복 삽입된다.
        Rule(21,
             'prompts[\'response_synthesizer:refine_template\'].default_template.template = """원 질의는 다음과 같습니다.: {query_str}\n'
             '우리에게 주어진 응답은 다음과 같습니다.: {existing_answer}\n'
             '여기서 아래 컨텍스트를 참고하여, 응답을 더 낫게 만들 수 있는 상황입니다.\n'
             '------------\n'
             '{context_msg}\n'
             '------------\n'
             '주어진 새로운 컨텍스트에서, 주어진 질의에 더 나은 답을 응답해주세요. 주어진 컨텍스트가 충분히 유용하지 않으면 기존 응답을 내주세요. 모든 응답은 한국어로 해주세요.\n'
             '개선된 답변: """',

             '# ⚠️ 기존 코드는 get_prompts() 가 돌려준 객체의 template 을 직접 고쳤다.\n'
             '#    그런데 get_prompts() 는 deepcopy 를 반환하므로 엔진에 반영되지 않는다.\n'
             '#    에러도 나지 않고 셀 22 에서 바뀐 것처럼 출력돼 오히려 더 위험했다.\n'
             '#    (수강생은 프롬프트가 적용된 줄 알고 넘어가게 된다)\n'
             '#    update_prompts() 로 실제 반영한다.\n'
             'from llama_index.core import PromptTemplate\n\n'
             'new_refine = PromptTemplate("""원 질의는 다음과 같습니다.: {query_str}\n'
             '우리에게 주어진 응답은 다음과 같습니다.: {existing_answer}\n'
             '여기서 아래 컨텍스트를 참고하여, 응답을 더 낫게 만들 수 있는 상황입니다.\n'
             '------------\n'
             '{context_msg}\n'
             '------------\n'
             '주어진 새로운 컨텍스트에서, 주어진 질의에 더 나은 답을 응답해주세요. 주어진 컨텍스트가 충분히 유용하지 않으면 기존 응답을 내주세요. 모든 응답은 한국어로 해주세요.\n'
             '개선된 답변: """)\n\n'
             'query_engine.update_prompts(\n'
             '    {"response_synthesizer:refine_template": new_refine}\n'
             ')',
             "get_prompts() 직접 대입은 조용히 무시된다 → update_prompts() (D-3)"),
        Rule(22,
             "print(prompts['response_synthesizer:refine_template'].default_template.template)",
             "# 반영됐는지 엔진에서 다시 읽어 확인한다.\n"
             "# (기존처럼 사본을 출력하면 실제 반영 여부를 알 수 없다)\n"
             "print(query_engine.get_prompts()['response_synthesizer:refine_template'].get_template())",
             "사본이 아니라 엔진에서 다시 읽어 실제 반영을 확인 (D-3)"),
    ]),

    # =====================================================================
    # MiniGPT 는 여기서 다루지 않는다.
    #
    # 원본은 9종 중 혼자만 Keras/TensorFlow 스택이었다. keras-hub API 자체는
    # 유효했지만(D-5) tensorflow 2.20 과 nvidia-*-cu12 를 끌어와 기본 커널의
    # torch cu130 과 충돌해 **별도 venv 가 필요**했고, 수강생은 나머지 8종에서
    # 배운 도구를 여기서만 다시 배워야 했다.
    #
    # 그래서 문자열 치환이 아니라 **PyTorch/HF 스택으로 새로 작성**했다.
    #   → tools/build_minigpt_notebook.py 가 유일한 생성원이다.
    #
    # 여기에 통과 복사 규칙을 두면 새로 만든 노트북을 원본으로 덮어쓴다.
]


# ---------------------------------------------------------------------------


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_source(cell: dict, text: str) -> None:
    lines = text.splitlines(keepends=True)
    cell["source"] = lines


def find_source(work_rel: str) -> Path:
    """work 상대경로에 대응하는 원본을 찾는다.

    원본은 확장자가 없고 이름에 공백이 있다 ("HPC_DPO 실습").
    work 쪽은 공백을 없애고 .ipynb 를 붙였다 ("HPC_DPO실습.ipynb").
    공백을 제거한 이름으로 대조한다.
    """
    day, name = work_rel.split("/")
    want = name.removesuffix(".ipynb")
    for p in (SRC / day).iterdir():
        if p.is_file() and p.name.replace(" ", "") == want:
            return p
    raise SystemExit(f"[중단] 원본을 찾지 못했습니다: {SRC/day} 에서 {want!r}")


def apply(mig: Migration, dry: bool) -> tuple[int, int, list[str]]:
    src_path = find_source(mig.path)
    out_path = WORK / mig.path

    # 항상 원본에서 읽는다. 이전 실행 결과에 의존하지 않으므로 몇 번을 돌려도 같다.
    nb = json.loads(src_path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    applied = skipped = 0
    log: list[str] = []

    # --- 전역 규칙 먼저 ---
    for g in mig.globals_:
        hits = sum(cell_source(c).count(g.old) for c in cells)
        if hits == 0:
            raise SystemExit(
                f"\n[중단] {mig.path} 전역 치환 대상을 찾지 못했습니다.\n"
                f"  사유: {g.why}\n  찾던 문자열:\n    {g.old[:200]!r}\n"
                f"  → 원본이 바뀌었는지 확인하세요."
            )
        if hits != g.expect:
            raise SystemExit(
                f"\n[중단] {mig.path} 전역 치환 횟수 불일치.\n"
                f"  사유: {g.why}\n  예상 {g.expect}회, 실제 {hits}회\n"
                f"  찾던 문자열:\n    {g.old[:200]!r}\n"
                f"  → 자료가 바뀌었는지 확인하고 expect 값을 고치세요."
            )
        for c in cells:
            s = cell_source(c)
            if g.old in s:
                set_source(c, s.replace(g.old, g.new))
        applied += 1
        log.append(f"  [적용] 전역 {hits:>2}곳 · {g.why}")

    for r in mig.rules:
        if not (1 <= r.cell <= len(cells)):
            raise SystemExit(f"[중단] {mig.path} 셀 {r.cell} 범위 초과 (총 {len(cells)})")
        cell = cells[r.cell - 1]
        src = cell_source(cell)

        if r.old in src:
            set_source(cell, src.replace(r.old, r.new, 1))
            applied += 1
            log.append(f"  [적용] 셀 {r.cell:>2} · {r.why}")
        elif r.optional:
            log.append(f"  [경고] 셀 {r.cell:>2} · 대상 못 찾음(optional): {r.why}")
        else:
            raise SystemExit(
                f"\n[중단] {mig.path} 셀 {r.cell} 에서 대상 문자열을 찾지 못했습니다.\n"
                f"  사유: {r.why}\n"
                f"  찾던 문자열:\n    {r.old[:200]!r}\n"
                f"  실제 셀 내용:\n    {src[:400]!r}\n"
                f"  → 규칙표를 실제 내용에 맞게 고치세요. 조용히 넘어가지 않습니다."
            )

    if not dry:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    return applied, skipped, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="쓰지 않고 미리보기")
    args = ap.parse_args()

    print("=" * 78)
    print("노트북 마이그레이션" + ("  [DRY RUN — 파일을 쓰지 않습니다]" if args.dry_run else ""))
    print("=" * 78)
    print(f"원본(읽기전용): {SRC}")
    print(f"산출물        : {WORK}")
    print(f"\n인코더 : {ENCODER}\n학습   : {TRAIN_LM}\n서빙   : {SERVE_LM}")

    total = 0
    for mig in MIGRATIONS:
        a, _s, log = apply(mig, args.dry_run)
        total += a
        note = "" if log else "  (변경 없음 — 통과 복사)"
        print(f"\n── {mig.path}  (적용 {a}){note}")
        for line in log:
            print(line)

    print()
    print("=" * 78)
    print(f"완료: 총 {total}건 적용 · 노트북 {len(MIGRATIONS)}종")
    print("매번 원본에서 다시 생성하므로 몇 번을 실행해도 결과가 같습니다.")
    if args.dry_run:
        print("DRY RUN 이었습니다. 실제 적용하려면 --dry-run 없이 다시 실행하세요.")
    print("=" * 78)


if __name__ == "__main__":
    main()
