"""슬라이드 재조립기 — 원본 3덱에서 새 3덱을 **매번 다시** 만든다.

    uv run --project tools python tools/migrate_slides.py

`migrate_notebooks.py` 와 같은 규율:
- 원본(`ppt/`)은 읽기만 한다. 산출물은 `work/ppt/` 에 통째로 다시 쓴다.
- 치환 규칙에는 근거(why)를 붙인다. 앵커를 못 찾으면 조용히 넘어가지 않고 멈춘다.
- 배치는 `slide_layout.PLACEMENT` 가 단독으로 정한다.

신규 슬라이드는 **도너 복제** 방식이다: 그림 없는 텍스트 전용 장(1일차 p19)을
복제해 헤더·본문 텍스트만 갈아 끼운다. placeholder 방식은 이 덱들과 맞지 않는다 —
전 장이 Blank 레이아웃 + 슬라이드 내 장식 도형 구조라, 도너 쪽이 장식을 보존한다.
"""

from __future__ import annotations

import copy
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.slide import Slide
from pptx.util import Pt

import pptxcommon as pc
from slide_layout import DECKS, DECK_PAGES, PLACEMENT, WORK_PPT, NewSlide, SlideRef, out_path

# 신규 슬라이드의 도너 — 그림 없는 텍스트 전용 장 (2026-08-27 도형 실사로 선정)
DONOR = ("1일차", 19)
BOILER = "쉬운 HPC 활용교육"


# ---------------------------------------------------------------------------
# 치환 규칙
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlideRule:
    """원본 (덱, 페이지) 한 장 안의 문자열 치환. 임포트 시점에 적용되므로
    재배치로 번호가 밀릴 일이 없다. 그 장에서 정확히 count 회 치환돼야 한다."""
    deck: str
    page: int
    old: str
    new: str
    why: str
    count: int = 1


@dataclass(frozen=True)
class GlobalSlideRule:
    """재사용되는 원본 전 장 대상. 총 치환 횟수가 expect 와 다르면 중단."""
    old: str
    new: str
    why: str
    expect: int


# 코드 슬라이드의 현행 노트북 동기화 (E-2, 2026-08-27 XML 수확 기준).
# 문단(=줄) 단위로만 치환된다 — 여러 줄 교체는 줄별 규칙으로 나눈다.
RULES: list[SlideRule] = [
    SlideRule('2일차', 8, '!pip install openai vllm datasets',
              "%pip install -q -U 'openai<3' datasets",
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('2일차', 8, '!nohup python -m vllm.entrypoints.openai.api_server --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct &',
              '# 서버는 별도 venv 터미널에서: nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 --gpu-memory-utilization 0.80 --max-model-len 16384 &',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다 — api_server deprecated + torch 하드핀 venv 분리'),
    SlideRule('2일차', 9, '실행된 모델은 Colab 실습 환경 안에 모델 서비스로 실행중에 있습니다.',
              '실행된 모델은 실습 환경(VESSL) 안에 모델 서비스로 실행 중입니다.',
              '실습 환경이 Colab → VESSL 로 바뀌었다 (E-2)'),
    SlideRule('2일차', 11, 'dataset = load_dataset("e9t/nsmc", trust_remote_code=True)',
              'dataset = load_dataset("e9t/nsmc", revision="refs/convert/parquet")',
              'datasets 5.x 스크립트 로딩 제거 — parquet 리비전 (CLAUDE.md §9 실측)'),
    SlideRule('2일차', 31, 'raw_datasets = load_dataset("beomi/KoAlpaca-v1.1a")',
              'raw_datasets = load_dataset("json", data_files=["assets/amazon_ko_sft.jsonl.gz", "data/amazon_ko_sft.mine.jsonl"])',
              'KoAlpaca 는 CC BY-NC. 현행 SFT 는 2일차에 직접 만든 데이터로 학습한다'),
    SlideRule('2일차', 31, '학습 데이터는 KoAlpaca 데이터셋을 활용 합니다.',
              '학습 데이터는 2일차에 직접 만든 상품 요약 데이터셋입니다.',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('2일차', 75, '"Qwen/Qwen2.5-0.5B-Instruct"',
              '"Qwen/Qwen3-0.6B-Base"',
              'GRPO 만 구세대 모델이었다 — Qwen3 계열 통일 (CLAUDE.md §10)'),
    SlideRule('3일차', 23, '! nohup python -m vllm.entrypoints.openai.api_server --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct &',
              '# 서버는 별도 venv 터미널에서: nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 &',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('3일차', 57, '여기서는 LG 엑사원 모델을 활용합니다.',
              '여기서는 vLLM 서버의 Qwen3-4B 모델을 활용합니다.',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('3일차', 57, 'from llama_index.llms.vllm import Vllm',
              'from llama_index.llms.openai_like import OpenAILike',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다 — 현행 RAG 는 vLLM 서버에 HTTP 로 붙는다'),
    SlideRule('3일차', 57, 'Settings.llm = Vllm(',
              'Settings.llm = OpenAILike(',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('3일차', 57, "dtype='float16',",
              "model='Qwen/Qwen3-4B-Instruct-2507', api_base='http://localhost:8000/v1',",
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('3일차', 57, "model='LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct'",
              "api_key='EMPTY', is_chat_model=True,",
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('1일차', 38, 'dataset = load_dataset("kor_ner")',
              'dataset = load_dataset("klue/klue", "ner")',
              'kor_ner 스크립트형 제거 — klue/klue ner (컬럼명 동일, CLAUDE.md §9 실측)'),
    SlideRule('2일차', 15, '[10:30]',
              '[100:105]',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 예시 20개가 아니라 5개', 2),
    SlideRule('2일차', 16, 'for i in range(100):',
              'for i in range(10):',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 노트북은 10건 (통계적 무의미도 명시)', 1),
    SlideRule('2일차', 32, 'indices = range(0,1000)',
              "n_train = min(1000, int(len(raw_datasets['train']) * 0.9))",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 고정 인덱스는 평가셋 0건 사고 전력, 비율 분할로', 1),
    SlideRule('2일차', 32, 'test_indices = range(1000, 1050)',
              'indices, test_indices = range(0, n_train), range(n_train, n_train + 50)',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 46, "output_dir = 'data/test_model'",
              "output_dir = 'data/sft_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 경로 분리 (DPO 이어받기의 전제)', 1),
    SlideRule('2일차', 46, 'overwrite_output_dir=True,',
              '',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — transformers 5 에서 제거된 인자', 1),
    SlideRule('2일차', 49, "'data/test_model'",
              "'data/sft_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 50, 'input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")',
              'inputs = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — transformers 5: return_dict + **inputs', 1),
    SlideRule('2일차', 50, 'input_ids=input_ids,',
              '**inputs,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 55, 'indices = range(0,500)',
              "n_train = min(500, int(len(raw_datasets['train']) * 0.9))",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 55, 'test_indices = range(500,550)',
              'indices, test_indices = range(0, n_train), range(n_train, n_train + 50)',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 63, 'from peft import LoraConfig, get_peft_model',
              'from peft import LoraConfig, PeftModel',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 63, 'model = get_peft_model(model, lora_config)',
              "model = PeftModel.from_pretrained(model, 'data/sft_model', is_trainable=True)  # SFT 이어받기",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — DPO 는 SFT 를 전제한다 (베이스에서 새 어댑터가 아니라)', 1),
    SlideRule('2일차', 64, "'data/test_model'",
              "'data/dpo_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 64, 'num_train_epochs=5,',
              'num_train_epochs=2,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 64, 'overwrite_output_dir=True,',
              '',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 67, "'data/test_model'",
              "'data/dpo_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 68, 'input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")',
              'inputs = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 68, 'input_ids=input_ids,',
              '**inputs,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 83, 'r=64,',
              'r=8,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 노트북: 생성 부담이 커서 가볍게', 1),
    SlideRule('2일차', 83, 'lora_alpha=16,',
              'lora_alpha=32,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 83, 'target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],',
              'target_modules=["q_proj", "v_proj"],',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 84, "'data/test_model'",
              "'data/grpo_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 84, 'max_prompt_length=128, ',
              '',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — GRPOConfig 에서 제거된 인자 (실측)', 1),
    SlideRule('2일차', 87, "'data/test_model'",
              "'data/grpo_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 88, 'input_ids = tokenizer.apply_chat_template(messages, truncation=True,',
              'inputs = tokenizer.apply_chat_template(messages, truncation=True,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 88, 'add_generation_prompt=True, return_tensors="pt").to("cuda")',
              'add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 88, 'input_ids=input_ids,',
              '**inputs,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('3일차', 56, '%pip install llama-index-llms-vllm',
              '%pip install llama-index-llms-openai-like',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 현행 RAG 는 vLLM 서버에 HTTP 로 붙는다', 1),
    SlideRule('3일차', 56, '%pip install vllm',
              '# vLLM 서버는 별도 venv 에서 — 노트북 환경에는 설치하지 않습니다',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — torch 하드핀 충돌 (venv 분리)', 1),
    SlideRule('3일차', 58, "dataset = load_dataset('heegyu/kowikitext', trust_remote_code=True, split='train[:1000]’)",
              "dataset = load_dataset('wikimedia/wikipedia', '20231101.ko', split='train[:1000]')",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — kowikitext 는 datasets 5.x 에서 로딩 불가 (실측)', 1),
    SlideRule('2일차', 65, 'eval_dataset=eval_dataset,',
              'eval_dataset=eval_dataset, processing_class=tokenizer,',
              '슬라이드↔현행 노트북 대조(E-3) — DPOTrainer 는 processing_class 필요', 1),
    SlideRule('1일차', 28, '먼저 개체명 추출 모델에 필요한 파이썬 패키지들을 선언합니다.',
              '먼저 감정 분류 모델에 필요한 파이썬 패키지들을 선언합니다.',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 28, 'Transformers: 개체명 추출 모델을 학습하기 위한 도구',
              'Transformers: 분류 모델을 학습하기 위한 도구',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 28, 'Datasets: 개체명 추출 모델 학습을 위한 오픈소스 데이터 전처리 도구',
              'Datasets: 학습 데이터를 불러오고 전처리하는 도구',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 28, 'Evaluate: 개체명 추출 평가 점수를 산출 하기 위한 도구',
              'Evaluate: 평가 점수를 산출하기 위한 도구',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 28, 'from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer',
              'from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer',
              '분류 실습인데 TokenClassification 을 import (C-4)', 1),
    SlideRule('1일차', 31, '학습 데이터는 “nsmc” 이라고 하는 허깅페이스 허브 내 데이터를 활용합니다.',
              '학습 데이터는 e9t/nsmc 라는 허깅페이스 허브 내 데이터를 활용합니다.',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 31, 'dataset = load_dataset("nsmc")',
              'dataset = load_dataset("e9t/nsmc", revision="refs/convert/parquet")',
              'datasets 5.x 스크립트 로딩 제거 (CLAUDE.md §9)', 1),
    SlideRule('1일차', 39, 'OG: 그룹 또는 집단, TI: 시간, LC: 위치, DT: 날짜, PS: 사람',
              'PS 인물 · LC 장소 · OG 기관 · DT 날짜 · TI 시간 · QT 수량  (B-/I- 접두 + O = 13종)',
              'klue/klue ner 의 라벨 체계로 (kor_ner 7종 → 13종)', 1),
    SlideRule('1일차', 39, "Sequence(feature=ClassLabel(names=['I', 'O', 'B_OG', 'B_TI', 'B_LC', 'B_DT', 'B_PS'], id=None), length=-1, id=None)",
              '# 13개: O, B-DT, I-DT, B-LC, I-LC, B-OG, I-OG, B-PS, I-PS, B-QT, I-QT, B-TI, I-TI',
              '옛 kor_ner 출력 → klue 라벨 (실측 CLAUDE.md §9)', 1),
    SlideRule('1일차', 40, 'tokenized_inputs["labels"] = new_labels.replace()',
              'tokenized_inputs["labels"] = new_labels',
              '원본 슬라이드 오타 — list 에 replace 없음 (복사 시 AttributeError)', 1),
    SlideRule('1일차', 43, "['O', 'O', 'O', 'O', 'B-PS', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']",
              "# (예: ['O', 'O', 'B-PS', 'I-PS', 'O', ...] — 실행해 실제 나열을 확인하세요)",
              '옛 kor_ner 데이터의 출력 예시 — klue 와 다르다', 1),
    SlideRule('1일차', 47, "[{'entity_group': 'B_LC', 'score': 0.8979496, 'word': '대한민국 서울', 'start': 6, 'end': 13}, {'entity_group': 'B_PS', 'score': 0.8987875, 'word': '홍길동', 'start': 18, 'end': 21}]",
              '# 출력: 병합된 개체와 라벨(LC, PS 등)을 확인하세요 — 쪼갠 토큰이 다시 하나로 합쳐집니다',
              '옛 kor_ner 라벨 체계(B_LC)의 출력 예시', 1),
    SlideRule('2일차', 17, '여기서 정확한 예측값을 “긍정“ 또는 “부정＂으로 출력값을 받기 위해 Structured Guide 생성을 적용해봅니다.',
              "예측값을 '긍정'/'부정' 둘 중 하나로만 받도록 제약 디코딩(structured outputs)을 적용합니다 — 부탁이 아니라 문법으로 막는 것입니다.",
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인 — 다음 신규 장(강제력 3단계)과 용어 일치', 1),
    SlideRule('2일차', 50, '저장된 학습 모델을 활용하여 실제 학습이 잘 되었는지 확인해 봅니다.',
              '학습 데이터가 상품 요약이었으므로, 일부러 다른 주제(정보 이론)를 물어봅니다 — 파인튜닝의 경계가 보입니다.',
              '생성 예제가 학습 도메인과 무관하다는 오해 방지 (대조 지적)', 1),
    SlideRule('2일차', 55, '# remove this when done debugging',
              '',
              '원본 디버그 잔재 (노트북에서는 제거됨)', 1),
    SlideRule('2일차', 63, '효율적인 학습을 위해 LoRA 파인튜닝으로 진행합니다.',
              'SFT 에서 만든 LoRA 어댑터를 이어받아 학습을 계속합니다. (아래 LoraConfig 는 SFT 가 쓴 값의 참고용입니다)',
              '이어받기로 바뀐 코드와 설명 일치 (자기검토)', 1),
    SlideRule('2일차', 81, 'rewards_list = [1.0 if match else 0.0 for match in matches]',
              '',
              '미사용 죽은 줄 (바로 아래 같은 식을 return)', 1),
    SlideRule('3일차', 19, 'pip install openai vllm datasets',
              "%pip install -q -U 'openai<3' datasets    (vLLM 서버는 별도 venv)",
              'Amazon 환경 세팅 장의 옛 설치 명령 (E-2 누락분)', 1),
    SlideRule('3일차', 65, '만약 프롬프트를 수정하고자 한다면 템플릿 값에 새로운 프롬프트를 할당해주면 됩니다.',
              '프롬프트 교체는 update_prompts() 로 합니다 — 대입은 에러 없이 무시됩니다 (다음 장의 함정).',
              '바로 뒤 신규 장과 자기모순이던 옛 대입 방식 (자기검토)', 1),
    SlideRule('3일차', 65, 'prompts[\'response_synthesizer:refine_template\'].default_template.template = """원 질의는 다음과 같습니다.: {query_str}',
              'REFINE_KO = """원 질의는 다음과 같습니다.: {query_str}',
              '위와 동일', 1),
    SlideRule('3일차', 65, "print(prompts['response_synthesizer:refine_template'].default_template.template)",
              "query_engine.update_prompts({'response_synthesizer:refine_template': PromptTemplate(REFINE_KO)})   # from llama_index.core import PromptTemplate",
              '위와 동일 — 확인 print 대신 실제 등록', 1),
    SlideRule('3일차', 68, '왜 꼬리질문이 필요할까요? \uf0e8 Q&A를 활용한 챗봇을 만든다면, 사용자가 직접 질문을 구체적으로 하기가 쉽지 않습니다.',
              '한 번의 검색으로 안 되는 질문이 있습니다 — "백남준과 맥스웰은 각각 어떤 분야의 인물인가요?" 를 통째로 검색하면 두 인물이 같이 나오는 문서를 찾게 됩니다.',
              '노트북의 프레이밍(복합 질문 분해)과 일치시킴', 1),
    SlideRule('3일차', 68, '꼬리질문 생성을 통해 사용자에게 제안을 할 수 있는 기능을 추가하여 Q&A 챗봇의 활성도를 높힐 기회를 얻을 수 있습니다.',
              '질문을 하위 질문으로 쪼개 각각 검색하고 합치면 됩니다 — 그 분해를 LLM 이 합니다.',
              '위와 동일 + 높힐 표기', 1),
]

# 전역 규칙. expect 는 재사용 257장의 XML 결합 텍스트 실측값 (2026-08-27).
# 섹션 규칙은 이름으로 구분되어 서로의 출력을 다시 잡지 않는다 (충돌 분석 완료).
GLOBAL_RULES: list[GlobalSlideRule] = [
    GlobalSlideRule('3.\t트렌스포머와 GPT', '2.\t트랜스포머와 ChatGPT',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 14),
    GlobalSlideRule('2.\t자연어처리 머신러닝 소개', '3.\t자연어처리 머신러닝 소개',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('2. 자연어처리 머신러닝 소개', '3. 자연어처리 머신러닝 소개',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여. 자기검토에서 3장 드롭 → 23', 23),
    GlobalSlideRule('1. 도메인 최적화 프리트레인에 대해서', '4. 도메인 최적화 프리트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측) + 표기 정정', 1),
    GlobalSlideRule('1. 도메인 최적화 프리트레이닝에 대해서', '4. 도메인 최적화 프리트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 13),
    GlobalSlideRule('2. VLLM을 활용한 데이터처리 실습', '5. vLLM을 활용한 데이터 생성 실습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측). Amazon 은 정제가 아니라 생성 축이라 명칭도 맞춘다', 35),
    GlobalSlideRule('3. Llama Index를 활용한 RAG 실습', '1. Llama Index를 활용한 RAG 실습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여. 자기검토에서 출력 3장 드롭 → 15', 15),
    GlobalSlideRule('4. 지속적인 LLM 챗봇 개발을 위해서', '2. 태스크 정의와 평가',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측). 커리큘럼 세부항목명과 일치. 자기검토에서 p80 까지 드롭 → 간지만 1', 1),
    GlobalSlideRule('1. 퓨샷 러닝에 대해서', '3. 퓨샷 러닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('1. 퓨샷러닝에 대해서', '3. 퓨샷러닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 14),
    GlobalSlideRule('2. 포스트 트레이닝에 대해서', '4. 포스트 트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 11),
    GlobalSlideRule('3.\t 인스트럭션 모델 학습', '5.\t 인스트럭션 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('3. 인스트럭션 모델학습', '5. 인스트럭션 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여. E-3 에서 KoAlpaca 4장 드롭 → 18', 18),
    GlobalSlideRule('4.\t 선호기반 모델 학습', '6.\t 선호기반 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('4. 선호기반 모델학습', '6. 선호기반 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 17),
    GlobalSlideRule('5.\t 리즈닝 모델 학습', '7.\t 리즈닝 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('5. 리즈닝 모델학습', '7. 리즈닝 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 18),
    GlobalSlideRule('Vllm', 'vLLM',
                    '용어 표준 (CLAUDE.md §7). p57 줄 교체가 먼저 2곳을 없애 13', 13),
    GlobalSlideRule('evaluation_strategy', 'eval_strategy',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측) — transformers 5 에서 제거, 슬라이드 코드가 깨진다 (C-3)', 2),
    GlobalSlideRule('trainer.tokenizer', 'trainer.processing_class',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측)', 3),
    GlobalSlideRule('max_seq_length', 'max_length',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측) — SFTConfig 인자명 변경', 1),
    GlobalSlideRule('RLPH', 'RLHF',
                    '용어 표준 (CLAUDE.md §7)', 1),
    GlobalSlideRule('높히', '높이',
                    '용어 표준 (CLAUDE.md §7)', 1),
    GlobalSlideRule('실습에서는 ChatML 템플릿을 적용합니다.',
                    '실습에서는 커스텀 챗 템플릿을 정의해 적용합니다.',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인 — ChatML 이 아니라 커스텀 템플릿 (명칭 오류)', 4),
    GlobalSlideRule('파라메터',
                    '파라미터',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인 — 표기 통일 (노트북과 동일)', 8),
    GlobalSlideRule('Bert 모델',
                    'BERT 모델',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인 — 표기', 4),
    GlobalSlideRule('Bert를',
                    'BERT를',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    GlobalSlideRule('Bert는',
                    'BERT는',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    GlobalSlideRule('Gemma-2-ko GPT 모델을 활용하여 모델 학습을 실습하고자 합니다.',
                    'mmBERT 인코더 모델을 활용하여 사전학습+파인튜닝 방식 그대로 실습합니다.',
                    '분류·NER 본문 서술이 여전히 Gemma 였다 (C-1 잔여)', 2),
    GlobalSlideRule('tokenizer=tokenizer,',
                    'processing_class=tokenizer,',
                    'transformers 5 — Trainer(tokenizer=) 제거 (CLAUDE.md §9)', 2),
    GlobalSlideRule('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct',
                    'Qwen/Qwen3-4B-Instruct-2507',
                    'EXAONE NC 라이선스 → Qwen3 (CLAUDE.md §10). 서빙 줄 교체(RULES)로 3곳이 먼저 사라져 7', 7),
    GlobalSlideRule('"beomi/gemma-ko-2b"',
                    '"jhu-clsp/mmBERT-base"',
                    '분류·NER 모델 교체 — 인코더 전환. 자기검토에서 p29 드롭 → 2', 2),
    GlobalSlideRule('client.beta.chat.completions.parse(',
                    'client.chat.completions.create(',
                    'openai SDK 2.x — .beta 네임스페이스 이동 (노트북과 동일 처리)', 6),
    GlobalSlideRule('extra_body={"guided_json": feature_type_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "feature_type_list", "schema": feature_type_schema}},',
                    'vLLM 0.12 에서 guided_json 정식 제거 — 조용히 무시되는 함정 (CLAUDE.md §9)', 1),
    GlobalSlideRule('extra_body={"guided_json": subsectoin_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "subsection_list", "schema": subsectoin_schema}},',
                    'guided_json 제거 (위와 동일)', 1),
    GlobalSlideRule('extra_body={"guided_json": extracted_feature_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "extracted_feature_list", "schema": extracted_feature_schema}},',
                    'guided_json 제거 (위와 동일)', 1),
    GlobalSlideRule('extra_body={"guided_json": consumer_category_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "consumer_category_list", "schema": consumer_category_schema}},',
                    'guided_json 제거 (위와 동일)', 1),
    GlobalSlideRule('extra_body={"guided_json": summary_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "summary", "schema": summary_schema}},',
                    'guided_json 제거 (위와 동일)', 2),
    GlobalSlideRule('extra_body={"guided_choice": ["긍정", "부정"]},',
                    'extra_body={"structured_outputs": {"choice": ["긍정", "부정"]}},',
                    'guided_choice 제거 — structured_outputs (CLAUDE.md §9 실측)', 2),
    GlobalSlideRule('공유드린 Github 실습자료를 Colab에서 불러옵니다.',
                    '실습 노트북(work/notebook)을 VESSL 워크스페이스에서 엽니다.',
                    '실습 환경 Colab → VESSL (E-2)', 3),
    GlobalSlideRule('%pip install -q bitsandbytes trl peft math_verify',
                    '%pip install -q math_verify   # 나머지 학습 스택은 사전 설치됨',
                    'setup_vessl.sh 가 스택을 설치한다 — 긴 줄 먼저 (접두사 충돌)', 1),
    GlobalSlideRule('%pip install -q bitsandbytes trl peft',
                    '# torch 는 constraints 로 고정되어 재설치되지 않습니다',
                    'setup_vessl.sh 가 스택을 설치한다', 2),
    GlobalSlideRule('%pip install -q transformers[torch] datasets',
                    '# 학습 스택은 setup_vessl.sh 가 설치해 둡니다 (transformers·datasets·trl·peft)',
                    'setup_vessl.sh 가 스택을 설치한다', 3),
]

# ---------------------------------------------------------------------------
# 신규 슬라이드 레지스트리 — key → 장 목록
#   각 장: dict(header=우상단 헤더, title=소제목, bullets=[(level, text), ...])
#   Phase F 에서 원고로 대체한다. 지금은 파이프라인 검증용 스텁.
# ---------------------------------------------------------------------------
NEW_SLIDES: dict[str, list[dict]] = {
    # 목차 3장은 각 덱의 원본 목차(p2)를 도너로 쓴다 — 목차 서식을 그대로 물려받는다
    # (본문형 도너로 찍었더니 정리가 안 돼 보인다는 사용자 지적, 2026-08-27).
    "toc1": [{"donor": ("1일차", 2), "toc": [
        '1.\tAI 소프트웨어 개발 개론',
        '2.\t트랜스포머와 ChatGPT',
        '3.\t자연어처리 머신러닝 소개와 실습',
    ]}],
    "toc2": [{"donor": ("2일차", 2), "toc": [
        '1.\tPre-training 등장 배경과 목적',
        '2.\tPre-training 데이터 처리',
        '3.\t미니 GPT 만들기',
        '4.\t도메인 최적화 프리트레이닝(CPT)',
        '5.\tvLLM을 활용한 데이터 생성 실습',
        '6.\t프롬프트 자동 최적화',
    ]}],
    "toc3": [{"donor": ("3일차", 2), "toc": [
        '1.\tLlama Index를 활용한 RAG 실습',
        '2.\t태스크 정의와 평가',
        '3.\t퓨샷 러닝',
        '4.\t포스트 트레이닝',
        '5.\t인스트럭션 모델 학습',
        '6.\t선호기반 모델 학습',
        '7.\t리즈닝 모델 학습',
        '8.\t마무리 — 방법 선택 가이드와 RAG 개선 확인',
    ]}],
    "rag_bm25": [
        {"header": '1. Llama Index를 활용한 RAG 실습 – 위키피디아 Q&A 실습',
         "title": 'BM25 — 임베딩 없이 찾는다',
         "bullets": [
            (0, 'BM25 는 단어가 겹치는 정도로 문서를 고른다 — 의미를 이해하지 않는다'),
            (1, '강한 곳: 고유명사 · 전문용어 · 숫자   /   약한 곳: 바꿔 말한 질문'),
            (1, '임베딩 검색과 반대 — 실무에서는 둘을 섞는다 (하이브리드)'),
            (0, '그래서 Settings.embed_model = None — 임베딩을 일부러 끈다'),
            (0, '청킹: 너무 크면 검색이 뭉뚝하고, 너무 잘면 문맥이 끊긴다'),
            (1, 'chunk_size=512 는 타협값 — RAG 품질이 안 나오면 가장 먼저 손보는 곳'),
        ]},
    ],
    "rag_prompt": [
        {"header": '1. Llama Index를 활용한 RAG 실습 – 위키피디아 Q&A 실습',
         "title": '프롬프트 교체의 함정 · 실패 진단',
         "bullets": [
            (0, '기본 프롬프트는 영어 — 한국어로 물어도 영어로 답하기 쉽다'),
            (0, '함정: get_prompts() 는 deepcopy 를 돌려준다'),
            (1, '받은 객체의 template 을 고쳐도 에러 없이 무시된다 → update_prompts() 를 쓴다'),
            (1, '서브질문 생성 프롬프트도 영어 — 영어 서브질문은 한국어 위키에서 아무것도 못 찾는다'),
            (0, '답만 보지 말고 source_nodes(근거 문서)를 본다'),
            (1, '근거가 무관 → 검색 실패  /  근거는 맞는데 답이 틀림 → 생성 실패'),
            (0, './bm25_retriever 인덱스를 지우지 말 것 — 하루의 끝에 다시 씁니다'),
        ]},
    ],
    "evalsec_a": [
        {"header": '2. 태스크 정의와 평가',
         "title": '평가는 태스크를 적는 것에서 시작한다',
         "bullets": [
            (0, '지표를 고르기 전에 네 가지를 적는다:'),
            (1, '입력 · 출력 · 성공 기준 · 실패 사례'),
            (1, '예: 한국어 질문 → 한국어 답 / 사실이 맞고 질문에 답할 것 / 영어로 답함·회피'),
            (0, '성공 기준이 여러 개면 지표도 여러 개 — 하나로 뭉뚱그리면 무엇이 나빠졌는지 모른다'),
            (0, '이 확인 수단을 학습(SFT·DPO·GRPO)보다 먼저 갖춘다 — 오늘 이 순서의 이유'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": '자동 지표 — BLEU 와 ROUGE',
         "bullets": [
            (0, '둘 다 모델 응답과 정답이 표면적으로 얼마나 겹치는가를 잰다'),
            (1, 'BLEU: 응답의 n-gram 중 정답에도 있는 비율 (정밀도 중심 · 번역)'),
            (1, 'ROUGE: 정답의 n-gram 중 응답에도 있는 비율 (재현율 중심 · 요약)'),
            (1, 'ROUGE-1 단어 / ROUGE-2 두 단어 연속 / ROUGE-L 최장 공통 부분열'),
            (0, '빠르고 싸고 재현 가능 — 그런데 다음 장의 한계가 있다'),
        ]},
    ],
    "evalsec_b": [
        {"header": '2. 태스크 정의와 평가',
         "title": '자동 지표의 한계 — 실측으로 확인',
         "bullets": [
            (0, '표면 일치만 재기 때문에 두 방향으로 속는다 (실습에서 직접 측정):'),
            (1, '뜻이 같아도 표현이 다르면 점수 하락 — "서울이 대한민국의 수도입니다"'),
            (1, '틀린 답이 높은 점수 — "대한민국의 수도는 부산입니다" (한 단어만 다르니까)'),
            (0, '의미와 사실 여부는 못 잰다 — 열린 생성에서 유일한 기준으로 삼으면 위험'),
            (0, '번역·요약처럼 정답 형태가 정해진 작업에는 여전히 유용하다'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": 'LLM-as-judge — 모델에게 채점을 맡긴다',
         "bullets": [
            (0, '사람 채점(정확·비쌈)과 자동 지표(싸고 얕음)의 중간'),
            (0, '핵심은 채점 기준을 명확히 주는 것 — 축을 나눠 지정한다'),
            (1, 'correctness(사실) · relevance(질문에 답했나) · fluency(자연스러운가)'),
            (0, '출력은 json_schema 로 강제 — 점수를 프로그램이 읽어야 하므로'),
            (1, '문자열 필드에 maxLength 상한 필수 — 없으면 끝없이 생성한다 (90초 실측)'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": 'judge 의 함정',
         "bullets": [
            (0, '재현성: temperature=0 으로 고정해야 같은 답에 같은 점수'),
            (0, '그래도 점수는 흔들린다 — 거르는 용도는 충분, 최적화 목적함수로는 위험'),
            (0, 'self-preference: 자기가 만든 답을 후하게 주는 편향'),
            (1, '평가 대상 모델과 채점 모델을 다르게 하는 편이 안전'),
            (0, '비용: 자동 지표는 즉시, judge 는 호출마다 시간과 돈'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": '무엇을 언제 쓰나',
         "bullets": [
            (0, '번역·요약 (정답 형태 고정)  →  BLEU · ROUGE'),
            (0, '분류·추출 (정답 하나)  →  정확도 · F1'),
            (0, '열린 생성 (챗봇 · QA)  →  LLM-as-judge + 자동 지표 보조'),
            (0, '최종 출시 판단  →  사람 평가 (표본이라도)'),
            (1, '실무는 섞어 쓴다: 개발 중 자동 지표 → 릴리즈 전 judge → 최종 사람'),
            (0, '이제 이 뒤의 SFT·DPO·GRPO 가 정말 나아졌는지 말할 수단이 생겼다'),
        ]},
    ],
    "fewshot_extra": [
        {"header": '3. 퓨샷러닝에 대해서 – vLLM을 활용한 퓨샷러닝 실습해보기',
         "title": '출력이 제멋대로다 — 형식을 잡는 3단계',
         "bullets": [
            (0, "형식을 말해주지 않으면 '긍정' / '이 리뷰는 긍정적입니다' / 이유 설명이 섞여 나온다"),
            (1, '모델이 틀린 게 아니라 우리가 형식을 안 정해준 것'),
            (0, '강제력의 3단계:'),
            (1, '① 프롬프트로 부탁한다 — 대체로 따르지만 보장 없음'),
            (1, '② seed 로 재현성 확보 — 흔들림만 줄인다'),
            (1, '③ 제약 디코딩(structured_outputs) — 다른 답이 나올 수 없다 (문법으로 차단)'),
        ]},
        {"header": '3. 퓨샷러닝에 대해서 – vLLM을 활용한 퓨샷러닝 실습해보기',
         "title": '답이 흔들리는 폭을 잰다',
         "bullets": [
            (0, '같은 질문을 n=10 으로 여러 번 — 보려는 것은 답이 아니라 흔들림의 폭'),
            (1, 'temperature 낮음 = 보수적 · 높음 = 과감  /  top_p 로 후보 범위 제한'),
            (0, '같은 입력에 답이 매번 다르면 그 프롬프트의 결과는 신뢰하기 어렵다'),
            (1, '분류처럼 답이 정해진 일에는 temperature 를 낮게'),
            (0, 'judge 점수에 노이즈가 있다던 프롬프트 최적화의 그 얘기와 같은 뿌리'),
        ]},
        {"header": '3. 퓨샷러닝에 대해서 – vLLM을 활용한 퓨샷러닝 실습해보기',
         "title": '결론은 숫자 둘의 비교다',
         "bullets": [
            (0, '같은 평가 루프를 zero-shot / few-shot 으로 두 번 돌려 정확도를 비교한다'),
            (1, '10건은 통계적으로 무의미 — 실습이라 작게, 실무는 수백 건'),
            (0, 'few-shot 이 높다 → 예시가 도움됐다'),
            (0, '비슷하다 → 이 작업에는 지시만으로 충분'),
            (0, '낮다 → 예시가 편향됐거나 문제를 헷갈리게 했다 (라벨 균형을 확인했는가?)'),
            (1, 'few-shot 이 항상 낫지는 않다 — 재봐야 안다'),
        ]},
        {"header": '3. 퓨샷러닝에 대해서 – vLLM을 활용한 퓨샷러닝 실습해보기',
         "title": '정보 추출 — 스키마로 강제한다 (KorQuAD)',
         "bullets": [
            (0, '분류는 choice 로 막았지만, 추출은 답의 개수·내용을 미리 모른다'),
            (0, '3단계 비교가 요점:'),
            (1, '① 그냥 묻는다 → 자유 문장, 파싱 불가'),
            (1, "② 프롬프트에 JSON 예시 → 대체로 되지만 가끔 깨진다 — 그 '가끔'이 운영 사고"),
            (1, '③ pydantic 스키마 + response_format → 항상 그 구조'),
            (0, '스키마 문자열 필드에 길이 상한(maxLength) — 없으면 잘려서 JSON 이 깨진다'),
        ]},
    ],
    "sft_data": [
        {"header": '5. 인스트럭션 모델학습 – SFT 모델 학습',
         "title": '어제 만든 데이터로 학습한다 — 877건',
         "bullets": [
            (0, '두 파일을 합쳐 읽는다:'),
            (1, 'assets/amazon_ko_sft.jsonl.gz — 강사 사전생성 (상품 100개 · 약 800건)'),
            (1, 'data/amazon_ko_sft.mine.jsonl — 어제 여러분이 직접 만든 것 (상품 10개)'),
            (0, '(instruction, output) 완전 중복만 제거 — 생성이 비결정적이라 거의 안 겹친다'),
            (0, '파일이 없으면 공개 데이터셋(KULLM)으로 폴백하되 크게 알린다'),
            (1, "폴백인 줄 모르고 '내 데이터로 학습했다' 고 오해하면 안 되니까"),
        ]},
    ],
    "sft_tmpl": [
        {"header": '5. 인스트럭션 모델학습 – SFT 모델 학습',
         "title": '-Base 모델에는 대화 형식이 없다',
         "bullets": [
            (0, '사전학습만 한 모델은 <|user|> 같은 대화 표시를 모른다 — 우리가 정해서 넣는다'),
            (1, '역할을 특수 문자열로 감싸는 커스텀 챗 템플릿 (Jinja 한 줄)'),
            (0, '-Instruct 모델은 이미 템플릿이 있다 — 덮어쓰면 오히려 성능이 떨어진다'),
            (0, '원칙: 학습할 때와 쓸 때의 형식이 같아야 한다'),
            (1, '어긋나면 학습은 정상 종료되는데 답이 이상한, 조용한 실패가 된다'),
            (1, '이 원칙이 SFT → DPO → GRPO 를 관통한다'),
        ]},
    ],
    "sft_len": [
        {"header": '5. 인스트럭션 모델학습 – SFT 모델 학습',
         "title": '학습 전 토큰 길이 확인 — 조용한 실패',
         "bullets": [
            (0, 'max_length 를 넘는 예시는 뒤가 잘린다 — 뒤 = 정답(assistant 답변)'),
            (0, '정답이 잘려나가도 에러가 없고 loss 는 오히려 낮게 나온다'),
            (1, '학습은 완주했는데 아무것도 못 배운 상태 — 써보기 전엔 모른다'),
            (0, '그래서 학습 전에 토큰 길이 분포를 반드시 찍어 본다'),
            (1, '상한 초과 비율이 몇 %인지, 잘리면 무엇이 잘리는지'),
        ]},
    ],
    "dpo_resume": [
        {"header": '6. 선호기반 모델학습 – DPO 모델 학습',
         "title": 'SFT 를 이어받는다 — 그리고 참조 모델이 안 보이는 이유',
         "bullets": [
            (0, 'DPO 는 SFT 완료 모델에서 출발한다 — 베이스에 바로 걸면 잘 안 된다'),
            (1, "PeftModel.from_pretrained(model, 'data/sft_model', is_trainable=True)"),
            (1, 'is_trainable=True 를 빠뜨리면 어댑터가 얼어붙어 학습이 안 되는 함정'),
            (0, '참조 모델이 코드에 없는 이유 — LoRA 라서'),
            (1, '원래 가중치는 얼려 둔 채 어댑터만 학습 → 어댑터를 잠깐 끄면 그게 곧 참조 모델'),
            (1, 'TRL 이 내부에서 처리 — 모델 하나 분량의 메모리를 아낀다'),
        ]},
    ],
    "dpo_logs": [
        {"header": '6. 선호기반 모델학습 – DPO 모델 학습',
         "title": '학습 로그 읽는 법 · ORPO 라는 선택지',
         "bullets": [
            (0, 'rewards/chosen 과 rewards/rejected — 둘의 차이가 벌어지는지 본다'),
            (1, 'rewards/accuracies: chosen 을 더 높게 준 비율 — 올라가야 정상'),
            (0, '학습률은 SFT(5e-4)보다 낮은 3e-4 — 이미 배운 것을 잃지 않으려고'),
            (0, 'ORPO: SFT 없이 선호 쌍만으로 한 단계 (loss = SFT 항 + 승산비 항)'),
            (1, '선호 데이터부터 모인 상황이라면 파이프라인이 짧아진다 — 마무리 가이드에서 비교'),
        ]},
    ],
    "grpo_reward": [
        {"header": '7. 리즈닝 모델학습 – GRPO 모델 학습',
         "title": '채점 함수의 품질이 곧 학습의 품질',
         "bullets": [
            (0, 'GRPO 에는 정답 라벨이 없다 — 답을 채점하는 함수가 그 자리를 차지한다'),
            (1, 'format_reward: <think>…</think><answer>…</answer> 형식을 지켰나'),
            (1, 'accuracy_reward: 답이 수학적으로 같은가 — math_verify 가 1/2 == 0.5 를 같게 본다'),
            (0, '함수가 엉성하면 모델이 그 엉성함을 파고든다 (리워드 해킹)'),
            (1, 'format_reward 하나만 쓰면 형식만 그럴듯하고 답은 틀리는 모델이 된다'),
            (0, "PPO 와의 차이: 보상 '모델'(신경망) 대신 보상 '함수' — 모델 하나를 통째로 아낀다"),
        ]},
    ],
    "grpo_group": [
        {"header": '7. 리즈닝 모델학습 – GRPO 모델 학습',
         "title": 'Group Relative — num_generations 가 정의다',
         "bullets": [
            (0, '문제 하나에 답을 4개 만들고(num_generations=4), 그 4개를 서로 비교한다'),
            (1, '그룹 평균보다 잘한 답은 확률↑, 못한 답은 확률↓ — 그룹 안에서 상대 평가'),
            (1, '절대 점수가 아니라서 별도의 기준선 모델이 필요 없다'),
            (0, '생성이 학습 루프 안에 있다 — GRPO 가 가장 비싼 이유'),
            (0, '로그 읽는 법: format 이 먼저 1.0 에 가고 accuracy 가 천천히 따라온다'),
            (1, '형식을 지키는 것이 답을 맞히는 것보다 쉬우니까'),
        ]},
    ],
    "minigpt": [
        {"header": '3. 미니 GPT 만들기 – 실습 준비',
         "title": '밑바닥부터 만든다',
         "bullets": [
            (0, '라이브러리가 주는 완성품 대신 GPT 구조를 직접 쌓고 데이터로 채운다'),
            (0, '약 2백만 파라미터 — 실습 시간 안에 학습이 끝나는 크기'),
            (1, '구조는 대형 모델과 완전히 같다. 크기만 다르다'),
            (0, '전체(사전학습 + CPT 3회)가 약 4분 (L40S 실측 225초)'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 데이터',
         "title": '한국어 동화 (TinyStories)',
         "bullets": [
            (0, '어휘와 문장 구조를 단순하게 설계한 데이터 — 작은 모델도 문법을 익힐 수 있게'),
            (1, '일반 웹 텍스트로 2M 모델을 학습하면 의미 없는 글자만 나온다'),
            (0, '이 선택이 뒤의 이어학습(CPT) 실험의 무대가 된다 — 동화만 아는 모델'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 토크나이저',
         "title": '토크나이저를 직접 학습한다 — ByteLevel BPE',
         "bullets": [
            (0, '모델은 글자를 모른다 — 숫자(토큰 ID)만 다룬다. 그 변환표를 데이터에서 만든다'),
            (0, '자주 붙어 나오는 바이트 쌍을 반복해서 합쳐 가는 방식 (BPE)'),
            (0, '왜 ByteLevel 인가: 한국어 음절이 수천 종이라 사전 5,000 으로는 [UNK] 대량 발생'),
            (1, '모든 텍스트를 바이트로 먼저 쪼개므로 모르는 글자가 원천적으로 없다'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 토크나이저',
         "title": '토큰화와 청킹',
         "bullets": [
            (0, '사전학습은 긴 글을 고정 길이(SEQ_LEN)로 잘라 학습한다'),
            (1, '문서 경계에 맞추지 않고 이어붙인 뒤 자르는 것이 일반적'),
            (1, '패딩 낭비 없이 GPU 를 꽉 채워 쓰기 위해서'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 모델 구현',
         "title": 'GPT 구조 개관',
         "bullets": [
            (0, '입력 토큰 ID → 토큰 임베딩 + 위치 임베딩'),
            (0, '→ [ Decoder Block ] × N   (LayerNorm → Causal Self-Attention → 잔차 / FFN → 잔차)'),
            (0, '→ LayerNorm → Linear(vocab 크기) → 다음 토큰의 확률'),
            (1, '앞서 이론에서 본 그 구조를 코드로 그대로 옮긴다'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 모델 구현',
         "title": 'Causal Self-Attention — 뒤를 가린다',
         "bullets": [
            (0, '학습할 때는 정답 문장을 통째로 넣는다 — 가리지 않으면 뒤를 보고 베낀다'),
            (0, '상삼각 마스크를 -inf 로 — softmax 를 거치면 확률 0'),
            (1, "GPT 가 '다음 토큰 맞히기'가 되는 핵심 장치"),
        ]},
        {"header": '3. 미니 GPT 만들기 – 모델 구현',
         "title": 'FeedForward 와 잔차 연결',
         "bullets": [
            (0, '어텐션 = 토큰끼리 정보를 주고받는 부분'),
            (0, 'FeedForward = 토큰 하나하나를 따로 가공하는 부분'),
            (0, 'Block 은 둘을 잔차(x + f(x))로 묶는다 — 깊게 쌓아도 그래디언트가 흐른다'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 모델 구현',
         "title": 'PreTrainedModel 상속 — 생태계에 얹는다',
         "bullets": [
            (0, '직접 만든 모델을 HuggingFace PreTrainedModel 로 감싼다'),
            (1, 'Trainer 로 학습하고 generate() 로 생성할 수 있게 된다'),
            (0, '파라미터는 임베딩·출력 레이어에 몰려 있다 — 사전 크기를 함부로 못 늘리는 이유'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 사전학습',
         "title": '학습 — 목표는 다음 토큰 맞히기 하나',
         "bullets": [
            (0, '그것만 반복하면 문법과 표현을 스스로 익힌다 — 이것이 사전학습'),
            (0, 'Trainer 에 데이터셋만 넘기면 루프는 표준 — loss 가 내려가는지 본다'),
            (1, '너무 빨리 0 에 가까워지는 것도 좋은 신호가 아니다 (외우는 중일 수 있다)'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 텍스트 생성',
         "title": '디코딩 전략 — 같은 모델, 다른 글',
         "bullets": [
            (0, '모델은 매 순간 다음 토큰의 확률 분포를 내놓는다'),
            (0, '그 분포에서 어떻게 고를 것인가가 디코딩 전략:'),
            (1, 'Greedy / Beam / Random / Top-K / Top-P — 다섯을 같은 프롬프트로 비교한다'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 텍스트 생성',
         "title": 'Greedy 와 Beam — 반복의 함정',
         "bullets": [
            (0, 'Greedy: 항상 최고 확률 토큰 — 결정적이지만 다양성이 없다'),
            (0, 'Beam: 전체 확률이 높은 문장을 찾는다 — 그런데 오히려 더 반복한다'),
            (1, '확률 높은 문장 = 짧고 안전하고 반복적인 문장이기 쉽다'),
            (1, '번역·요약(정답 있는 작업)엔 맞고, 이야기 생성(열린 작업)엔 단점'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 텍스트 생성',
         "title": 'Top-K 와 Top-P — 자르는 기준이 다르다',
         "bullets": [
            (0, 'Top-K: 후보 개수를 고정 — 상황에 따라 적절한 개수가 다른 게 약점'),
            (0, 'Top-P: 확률 합으로 자른다 — 확신하면 후보가 줄고, 애매하면 늘어난다'),
            (1, '실무에서 Top-P 를 더 많이 쓰는 이유'),
            (0, '글자가 깨져 나오면(예: 깜짝�어) 버그가 아니라 ByteLevel 의 구조'),
            (1, '한글은 한 글자가 3바이트 — 낮은 확률 토큰까지 뽑으면 글자 중간 바이트만 나온다'),
        ]},
    ],
    "minigpt_cpt": [
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": '이어서 학습시키기 — Continuous Pre-training',
         "bullets": [
            (0, '이 모델은 동화만 안다 — 새 도메인(위키)을 가르치고 싶다면?'),
            (0, '처음부터 다시는 너무 비싸다 → 이미 학습된 모델에 이어서 학습한다'),
            (1, 'llama-2-ko, EEVE-Korean 이 이 방식 — 한국어 LLM 대부분의 출생 경로'),
            (0, '재료는 오늘 첫 실습에서 정제한 ko_wiki_clean.jsonl (3,524건)'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": '첫 번째 문제 — 토크나이저가 안 맞는다',
         "bullets": [
            (0, '동화로 학습한 사전 5,000 — 위키의 한자·용어·연도가 잘게 쪼개진다'),
            (1, '실측: 동화 0.417 vs 위키 0.849 토큰/글자 — 2배 차이'),
            (0, '실무의 답은 어휘 확장 (llama-2-ko: 32,000→46,336)'),
            (1, '단 확장 직후엔 성능이 떨어졌다가 수십억 토큰을 학습해야 회복 — 실습에선 안 한다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": '두 번째 문제 — 배운 것을 잊는다',
         "bullets": [
            (0, '새 데이터만 계속 넣으면 원래 알던 것을 잊는다 (치명적 망각)'),
            (0, '대응: 원래 데이터를 조금 섞어 같이 학습 — replay'),
            (1, '문헌(Ibrahim 2024): 1%만으로도 유의미, 도메인이 크게 다르면 25%, 50%는 과함'),
            (0, '동화 → 백과사전은 강한 이동 — 0 / 5 / 25% 를 직접 비교한다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": 'learning rate 는 1/10 로',
         "bullets": [
            (0, '본학습 5e-4 → 이어학습 5e-5 — 임의가 아니라 공개 레시피의 관례'),
            (1, 'EEVE-Korean 4e-5 · llama-2-ko 1e-5 (원 LR 의 1/10~1/30)'),
            (0, 'LR 을 낮추면 망각이 줄고, 높이면 새 도메인에 빨리 적응 — 트레이드오프'),
            (1, 'warmup 길이는 망각·적응 어디에도 영향 없음(같은 논문) — 튜닝 대상이 아니다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": '결과 — 숫자를 먼저 본다 (L40S 실측)',
         "bullets": [
            (0, '동화 Perplexity:  학습 전 18.9  →  replay 0%: 37.3  /  5%: 22.5  /  25%: 20.5'),
            (1, '위키 Perplexity 는 반대로 53.5 → 54.8 → 59.7 — 공짜가 아니다'),
            (0, "replay 를 조금만 섞어도 상당히 돌아온다 — 문헌의 '1%만으로도 유의미' 그대로"),
            (0, '생성 결과에도 문체 오염이 보인다 — 동화를 쓰다 백과사전 말투가 튀어나온다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 미니GPT 마무리',
         "title": '여기서 만든 것',
         "bullets": [
            (0, '토크나이저를 직접 학습 → GPT 를 직접 구현 → 다음 토큰 맞히기로 사전학습'),
            (0, '디코딩 전략에 따라 같은 모델이 다른 글을 쓰는 것을 확인'),
            (0, '이어학습으로 새 도메인을 가르쳤고 그 대가(망각)와 막는 법(replay)을 봤다'),
            (0, '실제 LLM 은 같은 구조를 수천 배 키운 것 — 구조 자체는 방금 만든 것과 같다'),
            (1, '이 모델을 쓸 만하게 만드는 것(포스트트레이닝)이 내일의 주제'),
        ]},
    ],
    "tok_demo": [
        {"header": '3. 자연어처리 머신러닝 소개 – 챗봇을 위한 자연어 처리 실습',
         "title": '토크나이저 확인 — 문장이 몇 개로 쪼개지나',
         "bullets": [
            (0, '출력(토큰 ID·조각)은 실행해서 직접 확인하세요 — 모델마다 다릅니다'),
            (0, '볼 것: 한 문장이 몇 개로 쪼개지는가 · [CLS]/[SEP] 가 자동으로 붙는가'),
            (1, '같은 뜻의 한국어와 영어의 토큰 수가 다른 것 — 언어별 효율 차이'),
            (0, '모델과 토크나이저는 항상 짝 — 다른 모델의 토크나이저를 섞으면 안 된다'),
        ]},
    ],
    "ner_data": [
        {"header": '3. 자연어처리 머신러닝 소개 – 챗봇을 위한 자연어 처리 실습',
         "title": 'NER 데이터 — klue/klue (ner)',
         "bullets": [
            (0, 'dataset = load_dataset("klue/klue", "ner")'),
            (0, '학습 21,008 문장 · 컬럼은 tokens 와 ner_tags (어절 단위 라벨)'),
            (1, '예전의 kor_ner 는 datasets 5.x 에서 로딩되지 않습니다 (스크립트형)'),
            (0, 'KLUE 는 test 정답이 비공개 — 평가는 validation 으로 합니다'),
        ]},
    ],
    "rag_run": [
        {"header": '1. Llama Index를 활용한 RAG 실습 – 위키피디아 Q&A 실습',
         "title": 'Q&A 실행 — 결과를 읽는 법',
         "bullets": [
            (0, 'response = query_engine.query("백남준은 누구인가요?")'),
            (0, '답을 읽을 때 함께 볼 것:'),
            (1, '한국어로 답하는가 (프롬프트 교체가 실제로 반영됐는가)'),
            (1, '근거 문서(source_nodes)와 답이 맞는가 — 다음 장에서 근거를 직접 확인'),
            (0, '4B 모델이라 문장이 어색한 부분도 있습니다 — 그것도 관찰 대상입니다'),
        ]},
    ],
    "rag_subq": [
        {"header": '1. Llama Index를 활용한 RAG 실습 – 위키피디아 Q&A 실습',
         "title": '서브질문 실행 — 프롬프트 교체와 한계',
         "bullets": [
            (0, '서브질문 생성 프롬프트도 기본이 영어 — 영어 하위 질문은 한국어 위키에서 0건'),
            (1, '한국어 생성 프롬프트로 update_prompts() 교체 후 실행합니다 (노트북 참조)'),
            (0, 'response = sub_query_engine.query("백남준과 맥스웰은 각각 어떤 분야의 인물인가요?")'),
            (1, '하위 질문 2개가 각각 검색되고 답이 합쳐지는 과정을 로그로 확인'),
            (0, '단일 개체 질문에 쓰면 호출만 늘고 낫지 않습니다 — 항상 좋은 도구는 없다'),
        ]},
    ],
    "ml2dl": [
        {"header": '2.\t트랜스포머와 ChatGPT – 머신러닝에서 딥러닝으로',
         "title": '머신러닝 — 사람이 피처를 설계하던 시대',
         "bullets": [
            (0, '텍스트를 숫자로 바꾸는 일을 사람이 했다 — 단어 빈도(BoW), TF-IDF'),
            (0, '그 위에 통계 모델(로지스틱 회귀 · SVM)을 얹는 구조'),
            (0, '성능은 모델보다 피처 설계 손맛이 좌우했다'),
            (1, '새 도메인마다 피처를 다시 설계 — 옮겨 쓸 수 있는 것이 적었다'),
        ]},
        {"header": '2.\t트랜스포머와 ChatGPT – 머신러닝에서 딥러닝으로',
         "title": '딥러닝 — 표현을 데이터에서 배운다',
         "bullets": [
            (0, '단어를 사람이 설계한 규칙이 아니라 학습된 벡터(임베딩)로 표현'),
            (0, '다층 신경망이 피처를 스스로 만들어 낸다 — 표현 학습'),
            (0, '사람의 일이 피처 설계에서 구조(아키텍처) 설계로 옮겨 갔다'),
            (1, '데이터가 많을수록 좋아지는 성질이 여기서 시작된다'),
        ]},
        {"header": '2.\t트랜스포머와 ChatGPT – 머신러닝에서 딥러닝으로',
         "title": '순차 데이터와 RNN 의 한계',
         "bullets": [
            (0, '문장은 순서가 있는 데이터 — RNN 은 앞에서부터 차례로 읽는다'),
            (0, '한계 ① 멀리 떨어진 단어의 관계가 흐려진다 (장거리 의존)'),
            (0, '한계 ② 순서대로만 계산할 수 있어 병렬화가 안 된다 — 크게 못 키운다'),
            (1, '문장이 길어질수록, 데이터가 커질수록 두 한계가 함께 조여 온다'),
        ]},
        {"header": '2.\t트랜스포머와 ChatGPT – 머신러닝에서 딥러닝으로',
         "title": '어텐션 — 관계를 직접 본다',
         "bullets": [
            (0, '순서대로 전달하지 말고, 필요한 단어를 바로 참조하자는 발상'),
            (0, '모든 단어 쌍의 관계를 점수로 계산 — 멀어도 흐려지지 않는다'),
            (0, '순서 의존이 사라져 병렬 계산이 가능해졌다 — 키울 수 있게 됐다'),
            (1, '이 어텐션만으로 쌓아 올린 구조가 다음 장의 트랜스포머입니다'),
        ]},
    ],
    "pretrain_intro": [
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '과제마다 모델을 새로 만들던 시대',
         "bullets": [
            (0, '분류 모델 하나, 번역 모델 하나 — 과제마다 데이터를 모으고 처음부터 학습'),
            (0, '과제별 라벨 데이터 수천~수만 건이 늘 발목을 잡았다'),
            (1, '어제 분류·NER 실습을 사전학습 없이 했다면 15만 건으로도 부족했을 것'),
        ]},
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '전이학습 — 한 번 배운 언어를 옮겨 쓴다',
         "bullets": [
            (0, '언어 자체를 크게 한 번 배워 두고(사전학습), 과제에는 살짝 맞춘다(파인튜닝)'),
            (0, 'BERT · GPT 가 이 2단계 구조를 표준으로 만들었다'),
            (1, '어제 0.1 에폭 만으로 분류가 됐던 이유 — 모델이 이미 언어를 알고 있었다'),
            (0, '과제별 데이터 요구량이 수만 건에서 수백 건 수준으로 내려왔다'),
        ]},
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '자기지도학습 — 라벨 없이 배운다',
         "bullets": [
            (0, '사전학습의 라벨은 텍스트 자체다 — 다음 토큰 맞히기, 가린 토큰 맞히기'),
            (0, '사람이 라벨을 달지 않으므로 웹 전체가 학습 데이터가 된다'),
            (1, '라벨 병목이 사라지자 남은 병목은 데이터의 양과 질'),
        ]},
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '스케일링 — 크기가 능력을 산다',
         "bullets": [
            (0, '모델 · 데이터 · 연산을 함께 키우면 성능이 예측 가능하게 좋아진다 (스케일링 법칙)'),
            (0, '일정 규모를 넘으면 가르치지 않은 능력이 나타나기 시작했다'),
            (1, '지시 따르기 · 몇 개의 예시로 배우기(few-shot) — 내일 3일차의 주제들'),
            (0, '그래서 경쟁은 "누가 더 좋은 데이터를 더 많이 태우는가" 가 됐다'),
        ]},
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '그래서 오늘 하는 것',
         "bullets": [
            (0, '사전학습의 재료 — 데이터 처리 (바로 다음 절 + 실습)'),
            (0, '사전학습의 원리 — 미니 GPT 를 밑바닥부터 만들어 학습'),
            (0, '사전학습의 연장 — 이어학습(CPT)으로 새 도메인 가르치기'),
            (1, '구조와 코드는 공개돼 있습니다 — 오늘의 결론은 "데이터가 성능을 가른다" 입니다'),
        ]},
    ],
    "datacleaning": [
        {"header": '2.\tPre-training 데이터 처리',
         "title": '왜 데이터인가',
         "bullets": [
            (0, '모델 구조와 학습 코드는 공개돼 있다 — 같은 구조로 학습해도 성능이 갈린다'),
            (0, '가장 큰 이유는 데이터. 실무에서 시간이 가장 많이 드는 곳도 여기다'),
            (1, 'GPU 시간은 비싸다 — 쓰레기를 학습시킬 여유가 없다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '원본을 눈으로 본다',
         "bullets": [
            (0, '실습: 한국어 위키백과 5,000건 — 비교적 깨끗한 코퍼스조차 지저분하다'),
            (0, '중복 문서 · 목록뿐인 문서 · 다른 언어 · 마크업 잔해'),
            (1, '숫자(통계)만 보지 말고 실물을 열어 보는 것 — 이 실습에서 반복되는 원칙'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '정확 중복 제거 — 해시',
         "bullets": [
            (0, '완전히 같은 문서부터 걷어낸다'),
            (0, '문자열끼리 직접 비교하면 문서 수의 제곱 — 해시로 바꾸면 한 번씩만 훑는다'),
            (0, '그 전에 유니코드 정규화(NFC) — 눈에 같아 보여도 표현이 다르면 다른 문자열'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '진짜 문제는 근사 중복',
         "bullets": [
            (0, '글자 하나만 달라도 해시는 완전히 달라진다 — 정확 중복은 사실 별로 없다'),
            (0, '웹 코퍼스의 실제 중복: 같은 기사의 재게시 · 같은 템플릿 · 문단 하나 차이'),
            (1, '이걸 잡아야 모델이 특정 문장을 통째로 외우는 것을 막는다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '자카드 유사도와 shingle',
         "bullets": [
            (0, '문서를 연속된 n글자 조각(shingle)의 집합으로 만든다'),
            (0, '겹침 정도 = 자카드 유사도  J(A,B) = |교집합| / |합집합|'),
            (0, '문제: 문서 2만 개면 비교 쌍이 2억 개 — 전부 비교할 수 없다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": 'MinHash — 시그니처로 압축',
         "bullets": [
            (0, 'shingle 집합을 여러 해시 함수로 돌려 각 함수의 최솟값만 남긴다'),
            (0, '두 시그니처가 일치하는 비율 ≈ 자카드 유사도 (통계적 성질)'),
            (0, '긴 문서도 고정 길이(예: 64개) 숫자로 — 이제 비교가 싸졌다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": 'LSH 밴딩 — 비교 자체를 줄인다',
         "bullets": [
            (0, '시그니처를 여러 밴드로 쪼갠다'),
            (0, '한 밴드라도 완전히 같은 문서끼리만 후보로 본다 — 나머지 쌍은 보지도 않는다'),
            (1, '실습에서는 numpy 벡터화로 수천 문서를 수 초에 처리한다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '품질 필터 — 정답 없는 휴리스틱',
         "bullets": [
            (0, '최소 길이 · 한국어 비율 · 반복도 · 특수문자 비율 · 문장 부호'),
            (0, '정답이 있는 작업이 아니다 — 코퍼스마다, 목적마다 기준이 달라진다'),
            (0, '중요한 것은 걸러진 것을 직접 보는 것'),
            (1, '정상 문서가 잘리면 기준이 너무 센 것 · 쓰레기가 남으면 조여야 하는 것'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '실무 도구 — 원리는 방금 만든 것과 같다',
         "bullets": [
            (0, 'datatrove (HuggingFace) — 파이프라인 방식, 로컬·Slurm·S3 같은 코드'),
            (0, 'dolma (AI2) · text-dedup (커뮤니티)'),
            (0, 'GopherQualityFilter = 우리가 손으로 만든 휴리스틱들의 표준 구현'),
            (1, '도구를 쓰더라도 무엇이 걸러지는지는 직접 확인해야 한다 — 기본값을 믿지 말 것'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '산출물 — 오늘 만든 데이터로 오늘 학습한다',
         "bullets": [
            (0, 'ko_wiki_clean.jsonl — 5,000건 → 정제 후 3,524건 (실습 29초, L40S 실측)'),
            (0, '이 파일을 잠시 뒤 미니 GPT 의 이어학습(CPT)이 그대로 읽는다'),
            (0, '데이터셋 카드를 함께 남긴다 — 몇 달 뒤엔 출처도 기준도 기억나지 않는다'),
        ]},
    ],
    "promptopt": [
        {"header": '6.\t프롬프트 자동 최적화',
         "title": '학습 전에 최선을 다하는 방법',
         "bullets": [
            (0, '파인튜닝은 비쌉니다 — GPU 만이 아니라 데이터 · 평가 체계 · 유지보수'),
            (0, '실무 순서: 프롬프트로 해결되는가? → 예: 끝 / 아니오: 그때 학습'),
            (0, '이 실습은 왼쪽 가지를 끝까지 밀어붙입니다 — 그것도 자동으로'),
            (1, '프롬프트를 파라미터로 보고, 점수를 목적함수로 삼아 탐색합니다'),
        ]},
        {"header": '6.\t프롬프트 자동 최적화',
         "title": '3막 구조',
         "bullets": [
            (0, '1막 — 번역: 영문 상품 설명을 한국어로. 없던 한국어 데이터를 만든다'),
            (1, '고유명사는 원문 유지 · 쪼개진 브랜드명(F, a, t, S, h, a, r, k) 복원'),
            (0, '2막 — LLM-as-judge: 번역 품질을 채점하고 나쁜 것을 거른다'),
            (1, '같은 번역을 세 번 채점해 점수가 흔들리는 것도 직접 확인'),
            (0, '3막 — DSPy GEPA: 프롬프트를 자동으로 고쳐 쓴다'),
            (1, '산출물(amazon_ko.jsonl)은 3일차 SFT 의 학습 데이터가 된다'),
        ]},
        {"header": '6.\t프롬프트 자동 최적화',
         "title": '핵심 규칙 — 목적함수에 노이즈가 없어야 한다',
         "bullets": [
            (0, 'judge 점수를 그대로 목적함수로 쓰면? — 옵티마이저는 노이즈를 맞춘다'),
            (1, '점수는 오르는데 품질은 그대로인 결과. 실무에서 실제로 저지르는 실수'),
            (0, '그래서 최적화 대상을 바꿉니다: 번역 → 구조화 추출'),
            (1, '형식 준수 · 항목 수 · 원문 근거 · 한국어 — 전부 프로그램 판정 (노이즈 0)'),
            (0, '출력 형식은 고정하고, 지시문만 최적화합니다'),
            (1, '요구사항을 안 알려주고 맞히게 하면 개선이 나지 않습니다'),
        ]},
        {"header": '6.\t프롬프트 자동 최적화',
         "title": '결과 — 학습 없이 얻은 개선',
         "bullets": [
            (0, 'GEPA 120회 호출, 약 8분 (L40S 실측) — val 0.80 → 1.00'),
            (0, 'reflection LM 이 feedback 을 읽고 지시문을 스스로 고쳐 썼다'),
            (1, 'category 는 대분류로 · features 는 명사 기반으로 간결하게'),
            (0, '이 점수가 내일 학습과 비교할 기준선이 됩니다'),
            (1, '학습이 사주는 것은 프롬프트 위에 얹히는 만큼 — 그 차이가 판단 기준'),
        ]},
    ],
    "lora": [
        {"header": '5.\t인스트럭션 모델 학습 – LoRA 이론',
         "title": 'LoRA — 전부 학습시키지 않는다',
         "bullets": [
            (0, '원래 가중치 W 는 얼려 두고, 옆에 작은 행렬 두 개(A·B)만 붙여 학습한다'),
            (1, '출력 = W·x + B·A·x   (B·A 가 학습되는 부분 — 저랭크 보정)'),
            (0, '왜 되는가: 파인튜닝이 실제로 바꾸는 양은 저차원으로 근사된다'),
            (0, '저장은 어댑터(A·B)만 — 수백 MB 가 아니라 수십 MB'),
            (1, 'SFT 실습에서 trainable 2.99% 였던 것이 바로 이 구조입니다'),
        ]},
        {"header": '5.\t인스트럭션 모델 학습 – LoRA 이론',
         "title": 'r 과 alpha — 두 개의 손잡이',
         "bullets": [
            (0, 'r: 붙이는 행렬의 크기. 클수록 표현력이 늘고 무거워진다'),
            (0, 'alpha: 그 결과를 얼마나 세게 반영할지 (스케일 = alpha / r)'),
            (0, '이 과정의 실습값:'),
            (1, 'SFT: r=64, alpha=16 — 데이터가 877건이라 표현력을 넉넉히'),
            (1, 'GRPO: r=8, alpha=32 — 생성 부담이 커서 가볍게'),
            (0, '정답이 있는 값이 아닙니다 — 작업과 자원에 따라 고릅니다'),
        ]},
        {"header": '5.\t인스트럭션 모델 학습 – LoRA 이론',
         "title": '어디에 붙이고, 어떻게 배포하나',
         "bullets": [
            (0, 'target_modules: 어디에 붙일지 — 관례는 어텐션 4곳 (q·k·v·o proj)'),
            (1, '피드포워드까지 붙이면 표현력↑ 비용↑ — 실습은 어텐션만'),
            (0, '배포 두 가지 길:'),
            (1, '어댑터만 배포 — 베이스는 공용, 작업마다 어댑터 교체'),
            (1, 'merge_and_unload() — 베이스에 합쳐 단일 모델로 (추론 오버헤드 0)'),
            (0, 'DPO 실습에서 참조 모델이 필요 없던 이유 — 어댑터를 끄면 그게 원래 모델'),
        ]},
        {"header": '5.\t인스트럭션 모델 학습 – LoRA 이론',
         "title": 'QLoRA 와 알아둘 성질',
         "bullets": [
            (0, 'QLoRA: 베이스를 4bit 로 양자화해 올리고 그 위에 LoRA — 메모리 대폭 절감'),
            (1, '0.6B 실습엔 불필요하지만, 7B+ 를 GPU 한 장으로 다룰 때의 표준'),
            (0, '성질: LoRA 는 full FT 보다 도메인 밖 성능을 잘 보존한다'),
            (1, '망각 정도가 rank 로 조절된다 (Biderman 2024)'),
            (1, '단 무조건 덜 잊는다는 과장 — 비교 조건(LR)에 따라 달라진다'),
        ]},
    ],
    "method_guide": [
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '"우리 문제에는 뭘 써야 하죠?"',
         "bullets": [
            (0, '3일 동안 방법을 배웠습니다 — 프롬프트, SFT, DPO, ORPO, GRPO, RAG, CPT'),
            (0, '이 장은 그 사이에서 고르는 법입니다. 핵심 메시지는 셋:'),
            (1, '① 데이터가 방법을 정한다 — 방법을 먼저 고르지 않는다'),
            (1, '② 위에서부터 시도한다 — 프롬프트로 되면 학습하지 않는다'),
            (1, '③ DPO 는 SFT 를 전제한다 — ORPO 는 그 전제를 없앤 것'),
            (0, '"GRPO 가 최신이니 GRPO" 는 실수입니다 — 검증 함수 없이는 시작도 못 합니다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '의사결정 표 — 가진 것이 방법을 정한다',
         "bullets": [
            (0, '아무것도 없음  →  프롬프트 최적화  (학습 안 함 · 2일차)'),
            (0, '입력–정답 쌍  →  SFT  (3일차)'),
            (0, '선호 쌍 + SFT 완료 모델  →  DPO  (2단계 · 3일차)'),
            (0, '선호 쌍만, SFT 아직  →  ORPO  (1단계 · DPO 실습 비교 절)'),
            (0, '정답을 검증하는 함수  →  GRPO  (가장 비쌈 · 3일차)'),
            (0, '최신 사실이 필요  →  RAG  (학습 안 함 · 3일차)'),
            (0, '도메인 언어 자체가 다름  →  CPT  (가장 큼 · 2일차)'),
            (1, '위에서 아래로 갈수록 비싸집니다 — 위에서부터 시도합니다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '비용 감각 — 이 과정에서 직접 돌린 시간 (L40S 실측)',
         "bullets": [
            (0, '프롬프트 최적화 (GEPA 120회)   약 8분  /  실무: 수십 분~수 시간'),
            (0, 'SFT (0.6B · 877건 · LoRA)      약 8분  /  실무: 수 시간~수 일'),
            (0, 'DPO (0.6B · LoRA)              약 5분  /  SFT 를 마친 뒤에 다시 그만큼'),
            (0, 'GRPO (0.6B)                    약 8분  /  가장 비쌈 — 생성이 학습 안에 있다'),
            (0, 'CPT (2M 미니GPT · replay 3회)  약 2분  /  실무: 수 주 (llama-2-ko 는 40B+ 토큰)'),
            (1, 'GRPO 가 비싼 이유: 매 스텝 답을 여러 개 생성하고 각각 채점합니다'),
            (1, '실습 시간은 2026-08 실측, 실무 규모는 참고 수치입니다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '전제와 함정 ① — 프롬프트 최적화 · SFT',
         "bullets": [
            (0, '프롬프트 최적화'),
            (1, '전제: 채점할 수 있어야 한다 — 점수를 못 매기면 최적화도 없다'),
            (1, '함정: 노이즈 큰 점수를 목적함수로 삼으면 노이즈를 최적화한다'),
            (1, '→ 가능하면 프로그램으로 검증되는 metric (JSON 유효성 · 필수 항목 · 형식)'),
            (0, 'SFT'),
            (1, '전제: 입력–정답 쌍. 정답이 하나로 정해지는 작업에 맞는다'),
            (1, '함정: 정답이 여럿인 작업에 SFT 만 쓰면 한 표현만 정답이라고 가르치게 된다'),
            (1, '함정: 긴 입력이 잘려 정답이 날아가면 loss 는 낮은데 아무것도 안 배운다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '전제와 함정 ② — DPO · ORPO · GRPO',
         "bullets": [
            (0, 'DPO — 전제: SFT 완료 모델 + 선호 쌍'),
            (1, '베이스에 바로 걸면 잘 안 된다. 참조 모델로 메모리도 두 배 (LoRA 면 절약)'),
            (0, 'ORPO — 전제: 선호 쌍만'),
            (1, 'SFT 없이 한 단계로. 선호 데이터부터 모인 상황이라면 파이프라인이 짧아진다'),
            (0, 'GRPO — 전제: 정답을 코드로 검증할 수 있어야 한다'),
            (1, '수학(답 검증) · 코드(테스트) · 형식(스키마) — 이게 없으면 못 쓴다'),
            (1, '리워드 모델을 새로 학습시키는 길은 학습을 하나 더 얹는 것 — 비용이 다시 뛴다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '전제와 함정 ③ — 학습의 대안과 극단',
         "bullets": [
            (0, 'RAG — 문제가 "모델이 모른다" 일 때. 사실이 자주 바뀔 때. 학습이 아니다'),
            (0, 'CPT — 도메인 언어 자체가 다를 때 (법률 · 의료 · 특정 언어)'),
            (0, '자주 틀리는 판단: "우리 회사 문서를 학습시키자"'),
            (1, '대개는 RAG 가 맞습니다'),
            (1, '파인튜닝은 지식을 넣는 도구가 아니라 행동을 바꾸는 도구입니다'),
            (1, '오늘 아침 RAG 로 시작해 방금 그 시스템에 학습 모델을 꽂아 본 이유입니다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '실무 순서 — 이 장의 결론',
         "bullets": [
            (0, '① 프롬프트로 해결되는가?  예 → 끝. 학습하지 않는다'),
            (0, '② 모델이 모르는 게 문제인가?  예 → RAG'),
            (0, '③ 입력–정답 쌍이 있는가?  예 → SFT'),
            (0, '④ 선호 쌍이 있는가?  예 → SFT 했으면 DPO, 안 했으면 ORPO'),
            (0, '⑤ 검증 함수를 짤 수 있는가?  예 → GRPO'),
            (0, '⑥ 전부 아니오 → 데이터부터 만든다 (2일차로 돌아간다)'),
            (1, '마지막 가지가 중요합니다 — 방법이 없는 게 아니라 데이터가 없는 것'),
            (1, '이 과정이 2일차 하루를 데이터에 쓴 이유입니다'),
        ]},
    ],
    "ragcheck": [
        {"header": '8.\t마무리 — RAG 개선 확인',
         "title": '아침의 시스템에 오후의 모델을 꽂는다',
         "bullets": [
            (0, '같은 BM25 인덱스 · 같은 질문 · 같은 프롬프트 — 바꾸는 것은 모델 하나'),
            (0, '학습 전 (Qwen3-0.6B-Base): 지시 개념이 없어 이어쓰기만 한다'),
            (1, '검색이 완벽한 문서를 찾아줘도 시스템은 동작하지 않는다'),
            (0, '학습 후 (오늘 만든 SFT): 질문을 받았다는 걸 알고 답하려 한다'),
            (1, '학습 데이터는 상품 요약 877건 — 그런데도 답하는 행동이 옮겨 왔다'),
            (0, '개선의 실체는 지식이 아니라 행동입니다'),
        ]},
        {"header": '8.\t마무리 — RAG 개선 확인',
         "title": '3일의 결론',
         "bullets": [
            (0, '지식은 검색으로 넣고, 행동은 학습으로 바꾼다'),
            (0, '프롬프트로 되는지 먼저 — 안 되면 그때 학습'),
            (0, '무엇을 하든, 바꾸기 전에 잴 수단부터 갖춘다'),
            (1, '오늘 평가 실습을 학습보다 먼저 한 이유'),
            (0, '그 판단을 스스로 내릴 수 있게 된 것 — 이 3일의 목표였습니다'),
        ]},
    ],
}


# ---------------------------------------------------------------------------
# 텍스트 채우기 (도너 복제 후)
# ---------------------------------------------------------------------------
def _para_templates(tf) -> dict:
    """상자의 레벨별 <a:p> 를 서식 템플릿으로 수집한다 (run 이 있는 첫 문단)."""
    tmpl = {}
    for para in tf.paragraphs:
        if para.level not in tmpl and para.runs:
            tmpl[para.level] = para._p
    if not tmpl:
        raise SystemExit("[중단] 도너 상자에 run 있는 문단이 없습니다 — 도너를 바꾸세요")
    return tmpl


def _make_para(tmpl_p, text: str, size_scale: float | None = None):
    """템플릿 문단을 복제해 텍스트만 갈아 끼운다. 서식(pPr·rPr)은 그대로."""
    p_el = copy.deepcopy(tmpl_p)
    runs = p_el.findall(qn("a:r"))
    for r in runs[1:]:
        p_el.remove(r)
    for br in p_el.findall(qn("a:br")):
        p_el.remove(br)
    t = runs[0].find(qn("a:t"))
    t.text = text
    if size_scale is not None:
        rPr = runs[0].find(qn("a:rPr"))
        if rPr is not None and rPr.get("sz"):
            rPr.set("sz", str(int(int(rPr.get("sz")) * size_scale)))
    return p_el


def _fill_box(tf, lines: list[tuple[int, str]], size_scale: float | None = None) -> None:
    """상자를 (level, text) 목록으로 다시 채운다 — 도너의 레벨별 서식을 유지한 채."""
    tmpl = _para_templates(tf)
    levels = sorted(tmpl)

    def pick(lv):
        return tmpl[lv] if lv in tmpl else tmpl[min(levels, key=lambda a: abs(a - lv))]

    new_ps = [_make_para(pick(lv), tx, size_scale) for lv, tx in lines]
    txBody = tf._txBody
    for old in txBody.findall(qn("a:p")):
        txBody.remove(old)
    for np_ in new_ps:
        txBody.append(np_)


def fill_new_slide(part, spec: dict) -> None:
    slide = Slide(part._element, part)
    boxes = [sh for sh in slide.shapes
             if sh.has_text_frame and BOILER not in sh.text_frame.text
             and sh.text_frame.text.strip()]

    if "toc" in spec:
        # 목차 도너: "목 차" 제목은 그대로, 본문 상자만 교체.
        # 항목이 많으면 원본(4항목) 대비 비율로 글자를 줄여 상자를 넘치지 않게 한다.
        body = max(boxes, key=lambda b: len(b.text_frame.paragraphs))
        lines = list(spec["toc"])
        # 2·3일차 원본 목차는 번호를 자동 번호(buAutoNum)로 그린다 — 텍스트에도
        # 번호를 넣으면 "1. 1. ..." 로 두 번 찍힌다 (사용자 스크린샷으로 확인).
        tmpl0 = _para_templates(body.text_frame)[min(_para_templates(body.text_frame))]
        pPr = tmpl0.find(qn("a:pPr"))
        autonum = pPr is not None and pPr.find(qn("a:buAutoNum")) is not None
        if autonum:
            tab = chr(9)
            lines = [ln.split(tab, 1)[1] if tab in ln and ln.split(tab, 1)[0].rstrip(".").isdigit()
                     else ln for ln in lines]
        n = len(lines)
        scale = None if n <= 5 else round(5.5 / n, 3)
        _fill_box(body.text_frame, [(0, line) for line in lines], scale)
        return

    if len(boxes) < 3:
        raise SystemExit(f"[중단] 도너 장의 텍스트 상자가 예상과 다릅니다: {len(boxes)}개")
    boxes.sort(key=lambda b: (b.top or 0))
    header = boxes[0]
    body = max(boxes[1:], key=lambda b: (b.width or 0) * (b.height or 0))
    tagline = next((b for b in boxes[1:] if b is not body), None)

    _fill_box(header.text_frame, [(0, spec["header"])])
    # 도너 본문의 레벨 구조: lvl0 = 소제목 스타일(불릿·24pt), lvl1/2 = 하위 불릿.
    # 우리의 title → lvl0, bullets 의 level+1 → lvl1/2 로 자연 대응된다.
    lines = [(0, spec["title"])] + [(lv + 1, tx) for lv, tx in spec["bullets"]]
    _fill_box(body.text_frame, lines)
    if tagline is not None:
        _fill_box(tagline.text_frame, [(0, "")])


# ---------------------------------------------------------------------------
# 빌드
# ---------------------------------------------------------------------------
def expected_counts() -> dict[str, int]:
    out = {}
    for day, items in PLACEMENT.items():
        n = 0
        for it in items:
            n += len(NEW_SLIDES[it.key]) if isinstance(it, NewSlide) else 1
        out[day] = n
    return out


def build(dst_dir: Path | None = None, verbose: bool = True) -> dict[str, Path]:
    dst_dir = dst_dir or WORK_PPT
    pc.guard_masters_identical(DECKS)

    srcs = {d: Presentation(str(p)) for d, p in DECKS.items()}
    zips = {d: zipfile.ZipFile(p) for d, p in DECKS.items()}
    for d, prs in srcs.items():
        if len(prs.slides) != DECK_PAGES[d]:
            raise SystemExit(f"[중단] {d} 원본 장수 {len(prs.slides)} ≠ 기대 {DECK_PAGES[d]}")

    rules_by_page: dict[tuple[str, int], list[SlideRule]] = {}
    for r in RULES:
        rules_by_page.setdefault((r.deck, r.page), []).append(r)
    global_hits = {id(g): 0 for g in GLOBAL_RULES}

    # PowerPoint 가 산출물을 열어 두면 저장이 막힌다 — 시작 전에 전부 확인한다
    for day in PLACEMENT:
        target = (dst_dir) / out_path(day).name
        if target.exists():
            try:
                target.rename(target)   # Windows: 열려 있으면 여기서 실패한다
            except PermissionError:
                raise SystemExit(
                    f"[중단] {target.name} 이 PowerPoint 에 열려 있습니다. "
                    f"파일을 닫고 다시 실행하세요.")

    outputs = {}
    for day, items in PLACEMENT.items():
        shell = Presentation(str(DECKS[day]))
        pc.drop_all_slides(shell)
        media = pc.MediaCache(shell.part.package)
        sid = 256
        n_ref = n_new = 0

        for it in items:
            if isinstance(it, SlideRef):
                def mutate(el, _ref=it):
                    for rule in rules_by_page.get((_ref.deck, _ref.page), []):
                        n = pc.replace_in_tree(el, rule.old, rule.new)
                        if n != rule.count:
                            raise SystemExit(
                                f"[중단] 규칙 불일치 — {_ref.deck} p{_ref.page}: "
                                f"{rule.old!r} 를 {n}회 치환 (기대 {rule.count})\n  사유: {rule.why}")
                    for g in GLOBAL_RULES:
                        global_hits[id(g)] += pc.replace_in_tree(el, g.old, g.new)
                part = pc.clone_slide(srcs[it.deck], zips[it.deck], it.page, shell,
                                      media, mutate_element=mutate)
                pc.append_slide_part(shell, part, sid)
                n_ref += 1
            else:
                for spec in NEW_SLIDES[it.key]:
                    dd, dp = spec.get("donor", DONOR)
                    part = pc.clone_slide(srcs[dd], zips[dd], dp, shell, media)
                    fill_new_slide(part, spec)
                    pc.append_slide_part(shell, part, sid)
                    sid += 1
                    n_new += 1
                sid -= 1
            sid += 1

        dst_dir.mkdir(parents=True, exist_ok=True)
        out = dst_dir / out_path(day).name
        shell.save(str(out))
        outputs[day] = out
        if verbose:
            print(f"{day}: 재사용 {n_ref} + 신규 {n_new} = {n_ref + n_new}장 → {out}")

    for g in GLOBAL_RULES:
        if global_hits[id(g)] != g.expect:
            raise SystemExit(
                f"[중단] 전역 규칙 {g.old!r}: {global_hits[id(g)]}회 치환 (기대 {g.expect})\n"
                f"  사유: {g.why}")
    if verbose and GLOBAL_RULES:
        print(f"전역 규칙 {len(GLOBAL_RULES)}건 — 전부 기대 횟수와 일치")
    return outputs


if __name__ == "__main__":
    build()
    print("\n검증: uv run --project tools python tools/validate_slides.py")
