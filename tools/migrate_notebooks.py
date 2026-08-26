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

from layout import DAY_OF, SRC, WORK, find_source, prune_stale, work_path  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent

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
class AppendCells:
    """노트북 **끝에** 셀을 덧붙인다.

    Rule 은 기존 셀 안의 문자열 치환만 하므로 새 단계를 추가할 수 없다.
    한 셀에 수십 줄을 욱여넣는 우회는 쓰지 않는다 — 중간 출력이 없어 수강생이
    어디서 잘못됐는지 못 본다.

    **끝에만** 허용한다. 중간 삽입을 허용하면 그 뒤 모든 Rule 의 `cell` 번호가
    밀려서 규칙표 전체가 조용히 어긋난다. 그건 이 도구가 막으려는 바로 그 실패다.
    """
    cells: list[tuple[str, str]]   # ("markdown" | "code", source)
    why: str
    after_cell: int                # 적용 시점의 마지막 셀 번호. 다르면 중단한다.


@dataclass
class Migration:
    name: str                      # 노트북 파일명만. 일자는 layout.py 가 정한다.
    rules: list[Rule] = field(default_factory=list)
    globals_: list[GlobalRule] = field(default_factory=list)
    appends: list[AppendCells] = field(default_factory=list)

    @property
    def rel(self) -> str:
        """로그 표시용 '일자/파일명'."""
        return f"{DAY_OF.get(self.name, '?')}/{self.name}"


# 공통 조각 -----------------------------------------------------------------

PIP_TRAIN = """# 2026 스택. datasets==3.5.1 핀을 제거했다 — 스크립트 데이터셋 지원이
# datasets 4.0 에서 사라져 핀을 유지하면 오히려 최신 데이터셋을 못 읽는다.
%pip install -q -U transformers datasets evaluate accelerate"""

WHY_PIP = "!pip/%pip/pip 혼용 정리 + datasets 3.5.1 핀 제거 (D-8, D-0)"
WHY_PROC = "transformers 5: Trainer(tokenizer=) 제거 → processing_class= (D-4 실측 확정)"
WHY_TOKATTR = "transformers 5: trainer.tokenizer 속성 제거 → processing_class (D-4 실측 확정)"
WHY_PEFT = "TRL 현행 권장: get_peft_model() 선감싸기 → peft_config= 인자 전달 (D-4)"

# transformers 5.x 에서 apply_chat_template(return_tensors="pt") 는 텐서가 아니라
# BatchEncoding 을 돌려준다. 그걸 generate(input_ids=...) 에 넣으면
# inputs_tensor.shape[0] 에서 AttributeError 가 난다.
# return_dict=True 로 명시하고 **inputs 로 펼쳐 넘기는 것이 현행 권장 형태다.
CHAT_TMPL_OLD = (
    'input_ids = tokenizer.apply_chat_template(messages, truncation=True, '
    'add_generation_prompt=True, return_tensors="pt").to("cuda")'
)
CHAT_TMPL_NEW = (
    '# transformers 5 에서 apply_chat_template 은 BatchEncoding 을 돌려준다.\n'
    '# 예전처럼 generate(input_ids=...) 로 넘기면 AttributeError 가 난다.\n'
    'inputs = tokenizer.apply_chat_template(\n'
    '    messages, add_generation_prompt=True,\n'
    '    return_tensors="pt", return_dict=True,\n'
    ').to("cuda")'
)
GEN_OLD = "outputs = model.generate(\n        input_ids=input_ids,"
GEN_NEW = "outputs = model.generate(\n        **inputs,"
WHY_CHAT = ("transformers 5: apply_chat_template 이 BatchEncoding 반환 → "
            "generate(input_ids=) 에서 AttributeError (실행 검증에서 확인)")


# ---------------------------------------------------------------------------
# Amazon 노트북 끝에 붙이는 **구조 확인** 셀.
#
# 왜 필요한가: summary_by_users / summary_by_subsection 의 **안쪽** 구조가
# 코드만 봐서는 확정되지 않는다. gen_text_featured_summary 가 돌려준 문자열을
# 다시 json.dumps 로 감싸는 이중 인코딩이라, 안쪽 키가 무엇인지는 프롬프트가
# 결정하고 스키마로 강제되지 않는다.
#
# 학습용 JSONL 변환 코드(D-3)를 실행 결과 없이 쓰면 조용히 0건이 나온다.
# 그래서 변환 셀보다 **먼저** 구조를 찍는 셀을 둔다. 이 출력을 보고 변환을 확정한다.
AMAZON_PROBE = AppendCells(
    after_cell=55,
    why="산출물 구조 확인 셀 — 학습 데이터 변환(D-3)을 쓰기 전에 실제 형태를 본다",
    cells=[
        ("markdown", '''
---

## 산출물 구조 확인

지금까지 7단계를 거쳐 `output_data` 를 만들었습니다.
이걸 **다음 실습의 학습 데이터로** 쓰려면 형태를 정확히 알아야 합니다.

특히 `summary_by_users` 와 `summary_by_subsection` 은 **JSON 문자열 안에 또 JSON 문자열**이
들어 있는 구조입니다. `json.loads` 를 두 번 해야 값에 닿습니다.

> 이 셀은 눈으로 확인하기 위한 것입니다. 출력이 예상과 다르면 앞 단계에서
> 무언가 실패한 것입니다 — 다음 단계로 넘어가기 전에 여기서 잡아야 합니다.
'''),
        ("code", '''
import json

print("컬럼:", output_data.column_names)
print("행 수:", len(output_data))
print()

row = output_data[0]

for col in ["summary_by_subsection", "summary_by_users"]:
    print("=" * 72)
    print(f"[{col}]")
    print("=" * 72)
    v = row[col]
    print(f"  1차 타입: {type(v).__name__}")

    try:
        outer = json.loads(v) if isinstance(v, str) else v
    except (json.JSONDecodeError, TypeError) as e:
        print(f"  ★ 1차 파싱 실패: {type(e).__name__} — 앞 단계가 실패했을 수 있습니다")
        print(f"     원문 앞부분: {str(v)[:200]}")
        continue

    print(f"  1차 파싱 후: {type(outer).__name__}, 키 {list(outer)[:6]}")

    if isinstance(outer, dict) and outer:
        k = next(iter(outer))
        inner_raw = outer[k]
        print(f"  값 타입: {type(inner_raw).__name__}")
        print(f"  값 원문(앞 200자): {str(inner_raw)[:200]}")

        # ★ 여기가 핵심. 안쪽이 또 JSON 문자열인지, 어떤 키를 갖는지.
        try:
            inner = json.loads(inner_raw) if isinstance(inner_raw, str) else inner_raw
            if isinstance(inner, dict):
                print(f"  2차 파싱 성공 → dict, 키 {list(inner)}")
                for ik, iv in list(inner.items())[:2]:
                    print(f"     {ik}: {str(iv)[:150]}")
            else:
                print(f"  2차 파싱 성공 → {type(inner).__name__}: {str(inner)[:150]}")
        except (json.JSONDecodeError, TypeError):
            # 'error' 문자열이 json.dumps 로 감싸이면 파싱은 되고 문자열이 나온다.
            # 여기 걸린다는 것은 애초에 JSON 이 아니라는 뜻이다.
            print("  2차 파싱 불가 — 안쪽은 평문입니다")
    print()

# 실패한 건이 얼마나 되는지 센다. 'error' 는 파싱을 통과하므로 문자열로 직접 본다.
bad = sum(1 for r in output_data
          if "error" in str(r["summary_by_users"])[:40]
          or "error" in str(r["summary_by_subsection"])[:40])
print(f"제외 대상(생성 실패로 보이는 행): {bad} / {len(output_data)}")
'''),
    ],
)


# ---------------------------------------------------------------------------
# DPO 노트북 뒤에 붙이는 ORPO 비교 섹션.
#
# 과정 전체가 "이런 방법이 있다" 의 나열로 끝나고 **언제 무엇을 고르는가** 가 없었다.
# ORPO 는 DPO 와 같은 데이터(prompt/chosen/rejected)를 그대로 쓰면서 전제가 다르다 —
# 그 대비가 선택 기준을 가르치기에 가장 좋다.
#
# 기본은 **실행하지 않는다.** 돌리면 DPO 만큼 GPU 시간이 더 든다.
# import 도 가드 안에 둔다. 밖에 두면 trl 버전에 따라 노트북 전체가 죽는다.
ORPO_SECTION = AppendCells(
    after_cell=23,
    why="ORPO 비교 섹션 추가 — 선호학습의 다른 선택지 (기본 미실행)",
    cells=[
        ("markdown", '''
## 다른 선택지 — ORPO

방금 한 DPO 에는 **숨은 전제**가 있습니다. 두 가지입니다.

1. **SFT 를 먼저 마친 모델이 있어야 합니다.** DPO 는 "이미 말은 할 줄 아는 모델"을
   사람 취향 쪽으로 미는 방법입니다. 아무것도 학습 안 된 베이스 모델에 바로 DPO 를
   걸면 잘 안 됩니다.
2. **참조 모델(reference model)이 필요합니다.** 학습 중인 모델이 원래 모델에서
   너무 멀어지지 않게 붙잡아 두는 역할입니다. 그래서 메모리에 모델이 **두 개** 뜹니다.

**ORPO** (Odds Ratio Preference Optimization) 는 이 두 전제를 모두 없앱니다.

| | SFT | DPO | ORPO |
|---|---|---|---|
| 필요한 데이터 | 입력–정답 | prompt / chosen / rejected | prompt / chosen / rejected |
| 시작 모델 | 베이스 | **SFT 완료 모델** | **베이스** |
| 참조 모델 | 불필요 | **필요** (메모리 2배) | 불필요 |
| 단계 수 | 1 | 2 (SFT → DPO) | **1** |

### 어떻게 한 단계로 합치나

ORPO 의 손실 함수는 두 항을 더한 것입니다.

```
loss = SFT 손실(chosen 을 그대로 따라 하기)
     + λ · 승산비 항(chosen 이 rejected 보다 얼마나 더 그럴듯한가)
```

앞 항이 SFT 를, 뒤 항이 선호 학습을 담당합니다. 그래서 **한 번에 끝납니다.**
뒤 항은 확률이 아니라 **승산(odds)** 의 비를 씁니다 — 확률로 직접 비교하는 것보다
"조금 더 좋은 답" 과 "많이 더 좋은 답" 을 덜 과격하게 벌립니다.

> 논문: Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model* (2024)
'''),
        ("code", '''
# ⚠️ 기본은 실행하지 않습니다. 돌리면 위 DPO 만큼 GPU 시간이 더 듭니다.
#    코드가 어떻게 달라지는지 보는 것이 목적입니다. 직접 돌려보려면 True 로 바꾸세요.
RUN_ORPO = False

if not RUN_ORPO:
    print("ORPO 학습은 건너뜁니다 (RUN_ORPO = False).")
    print()
    print("DPO 와 무엇이 다른지만 보세요:")
    print()
    print("  [DPO] SFT 된 모델에서 출발 + 참조 모델 필요")
    print("      trainer = DPOTrainer(model=model, args=DPOConfig(...), ...)")
    print()
    print("  [ORPO] 베이스 모델에서 바로 출발 + 참조 모델 없음")
    print("      trainer = ORPOTrainer(model=base, args=ORPOConfig(beta=0.1, ...), ...)")
    print()
    print("  데이터는 prompt / chosen / rejected 로 **완전히 같습니다.**")
else:
    # import 를 가드 안에 둔다. 밖에 두면 trl 버전이 안 맞을 때 노트북 전체가 죽는다.
    from trl import ORPOTrainer, ORPOConfig
    from transformers import AutoModelForCausalLM

    # ★ DPO 때 쓰던 model 을 재사용하지 않는다. 그건 이미 학습된 것이라
    #   "베이스에서 바로 시작한다" 는 ORPO 의 요점이 사라진다.
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto",
    )

    orpo_args = ORPOConfig(
        beta=0.1,                 # 승산비 항의 가중치 λ. 크게 줄수록 선호를 세게 반영한다
        output_dir="data/test_model_orpo",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=3.0e-04,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        report_to="none",
    )

    orpo_trainer = ORPOTrainer(
        model=base,                       # 참조 모델 인자가 아예 없다
        args=orpo_args,
        train_dataset=train_dataset,      # DPO 와 같은 데이터를 그대로 쓴다
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    orpo_trainer.train()
'''),
        ("markdown", '''
### 그래서 언제 무엇을 쓰나

정리하면 **가진 데이터가 방법을 정합니다.** 방법을 먼저 정하고 데이터를 맞추는 것이
아닙니다.

| 가진 것 | 방법 |
|---|---|
| 아무것도 없음 | 먼저 **프롬프트**를 끝까지 밀어붙인다 (2일차 프롬프트 최적화) |
| 입력–정답 쌍 | **SFT** |
| 좋은 답/나쁜 답 쌍 + SFT 완료 모델 | **DPO** |
| 좋은 답/나쁜 답 쌍만 있고 SFT 는 아직 | **ORPO** — 한 단계로 끝낸다 |
| 정답을 **검증하는 함수** (수학·코드·형식) | **GRPO** (다음 실습) |

그리고 순서가 있습니다. **위에서부터 시도합니다.**
프롬프트로 해결되면 학습하지 않습니다. 학습은 비싸고, 데이터를 모아야 하고,
한 번 하면 유지보수가 따라붙습니다.

> 2일차에 프롬프트만으로 점수를 얼마나 올렸는지 기억하시나요.
> 학습이 사주는 것은 **그 위에 얹히는 만큼**입니다. 그 차이가 데이터를 모으고
> GPU 를 돌릴 값어치가 있는지가 판단 기준입니다.
'''),
    ],
)


MIGRATIONS: list[Migration] = [
    # =====================================================================
    Migration("HPC_Classification실습.ipynb", [
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
    Migration("HPC_NER실습.ipynb", [
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
    Migration("HPC_SFT실습.ipynb", [
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
        Rule(27, CHAT_TMPL_OLD, CHAT_TMPL_NEW, WHY_CHAT),
        Rule(27, GEN_OLD, GEN_NEW, WHY_CHAT),
    ]),

    # =====================================================================
    Migration("HPC_DPO실습.ipynb", [
        Rule(2, "!pip install -q transformers[torch] datasets==3.5.1",
             "%pip install -q -U transformers datasets accelerate", WHY_PIP),
        Rule(3, "!pip install -q trl peft", "%pip install -q -U trl peft", WHY_PIP),
        Rule(15, "    overwrite_output_dir=True,\n", "",
             "transformers 5 에서 overwrite_output_dir 제거됨 → TypeError (실행 검증에서 확인)"),
        Rule(18, "trainer.tokenizer.save_pretrained(output_dir)",
             "trainer.processing_class.save_pretrained(output_dir)", WHY_TOKATTR),
        Rule(22, CHAT_TMPL_OLD, CHAT_TMPL_NEW, WHY_CHAT),
        Rule(22, GEN_OLD, GEN_NEW, WHY_CHAT),
    ], appends=[ORPO_SECTION]),

    # =====================================================================
    Migration("HPC_GRPO실습.ipynb", [
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
        # GRPO 만 줄바꿈 위치가 다르다
        Rule(25,
             'input_ids = tokenizer.apply_chat_template(messages, truncation=True,\n'
             '                                          add_generation_prompt=True, return_tensors="pt").to("cuda")',
             CHAT_TMPL_NEW, WHY_CHAT),
        Rule(25, GEN_OLD, GEN_NEW, WHY_CHAT),
    ]),

    # =====================================================================
    Migration("HPC_퓨샷실습.ipynb", [
        Rule(4, "pip install openai vllm datasets==3.5.1",
             "# vLLM 은 별도 venv 에서 서버로 띄운다 (verify/02_vllm_venv.sh 참조).\n"
             "# 이 노트북은 HTTP 로 붙기만 하므로 openai 클라이언트만 있으면 된다.\n"
             "# openai 는 3.x 로 올리지 않는다. llama-index-llms-openai 가 openai<3 을\n"
             "# 요구해서, 3일차 RAG 실습과 같은 환경을 쓰려면 2.x 여야 한다.\n"
             "%pip install -q -U 'openai<3' datasets", WHY_PIP),
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
    Migration("HPC_Amazon요약실습.ipynb", [
        Rule(2, "pip install openai vllm datasets==3.5.1",
             "# vLLM 은 별도 venv 에서 서버로 띄운다 (verify/02_vllm_venv.sh 참조).\n"
             "# openai<3 : llama-index-llms-openai 가 openai<3 을 요구한다(3일차와 같은 환경)\n"
             "%pip install -q -U 'openai<3' datasets pydantic", WHY_PIP),
        # ── 프롬프트 6종 한국어화 ────────────────────────────────────
        # 원본은 전부 영어이고 마지막 프롬프트는 "in English only" 로 못박혀 있었다.
        # 한국어 강의인데 수강생이 프롬프트를 읽지 않고 넘어가게 된다.
        # 입력(영문 상품 설명)은 그대로 두고 프롬프트와 출력만 한국어로 바꾼다.
        # → "영문 원본에서 한국어 데이터셋을 만드는" 실습이 되어 3단원 데이터 처리와 맞는다.
        Rule(12,
             'STEP_1_PROMPT = """Objective:\n'
             'Analyze the provided product description and identify the types of features that can be attributed to the product.\n'
             '\n'
             'Task:\n'
             '1. List the names of the feature types (e.g., color, size, material, etc.).\n'
             '2. For each feature type, provide a brief description explaining what it represents.\n'
             '\n'
             'Key Considerations:\n'
             '\n'
             '1. Explicit Mention: Only include feature types that are explicitly mentioned in the product description to ensure accuracy.\n'
             '2. Clear Descriptions: Ensure each description is concise and clearly explains the feature type without including specific values.\n'
             '3. JSON Validity: Verify the JSON format for proper syntax to ensure usability and avoid errors.\n'
             '\n'
             'Example of Desired Output:\n'
             '\n'
             '[\n'
             '    {"feature_type": "Color", "descript": "The available shades or hues of the product."},\n'
             '    {"feature_type": "Size", "descript": "The dimensions or measurements of the product."},\n'
             '    {"feature_type": "Material", "descript": "The substance used to make the product."},\n'
             '    {"feature_type": "Brand", "descript": "The manufacturer or brand name of the product."},\n'
             '    {"feature_type": "Release Year", "descript": "The year the product was released."},\n'
             ']\n'
             '"""',

             'STEP_1_PROMPT = """목표:\n'
             '주어진 상품 설명을 분석해서, 이 상품에 어떤 종류의 속성이 있는지 찾아내세요.\n'
             '\n'
             '할 일:\n'
             '1. 속성의 종류를 나열합니다 (예: 색상, 크기, 재질 등).\n'
             '2. 각 속성이 무엇을 뜻하는지 짧게 설명합니다.\n'
             '\n'
             '주의할 점:\n'
             '\n'
             '1. 명시된 것만: 상품 설명에 실제로 나온 속성만 포함하세요. 추측하지 마세요.\n'
             '2. 값이 아니라 종류: 설명에는 구체적인 값(예: "빨강")을 넣지 말고 종류만 설명하세요.\n'
             '3. JSON 형식: 문법이 올바른 JSON 으로 출력하세요.\n'
             '4. **모든 출력은 한국어로 작성하세요.** 입력이 영어여도 한국어로 답하세요.\n'
             '\n'
             '출력 예시:\n'
             '\n'
             '[\n'
             '    {"feature_type": "색상", "descript": "상품이 제공하는 색상."},\n'
             '    {"feature_type": "크기", "descript": "상품의 치수나 규격."},\n'
             '    {"feature_type": "재질", "descript": "상품을 만든 소재."},\n'
             '    {"feature_type": "브랜드", "descript": "제조사 또는 브랜드 이름."},\n'
             '    {"feature_type": "출시연도", "descript": "상품이 출시된 연도."},\n'
             ']\n'
             '"""',
             "프롬프트 한국어화 — Step 1 (속성 종류 찾기)"),

        Rule(16,
             'STEP_2_PROMPT = """Objective:\n'
             'Categorize the identified features into the following subsections\n'
             '\n'
             'Task:\n'
             '\n'
             '- Organize into Subsections: Group the identified feature types into the predefined subsections.\n'
             '- JSON Output: Present the final result in a structured JSON format, with each subsection containing an array of feature objects.\n'
             '\n'
             'Key Considerations:\n'
             '\n'
             '- Explicit Mention: Only include feature types that are explicitly mentioned in the product description to ensure accuracy.\n'
             '- Clear Descriptions: Ensure each description is concise and clearly explains the feature type without including specific values.\n'
             '- JSON Validity: Verify the JSON format for proper syntax to ensure usability and avoid errors.\n'
             '\n'
             'Example of Desired Output:',

             'STEP_2_PROMPT = """목표:\n'
             '앞에서 찾은 속성들을 비슷한 것끼리 묶어서 분류하세요.\n'
             '\n'
             '할 일:\n'
             '\n'
             '- 그룹으로 묶기: 속성 종류들을 의미가 비슷한 것끼리 몇 개의 그룹으로 나눕니다.\n'
             '- JSON 출력: 각 그룹이 속성 목록을 갖는 JSON 형태로 정리합니다.\n'
             '\n'
             '주의할 점:\n'
             '\n'
             '- 명시된 것만: 앞에서 나온 속성만 사용하세요. 새로 만들지 마세요.\n'
             '- 그룹 이름은 짧고 명확하게: 무엇을 묶은 그룹인지 알 수 있어야 합니다.\n'
             '- JSON 형식: 문법이 올바른 JSON 으로 출력하세요.\n'
             '- **모든 출력은 한국어로 작성하세요.**\n'
             '\n'
             '출력 예시:',
             "프롬프트 한국어화 — Step 2 (속성 그룹화)"),

        Rule(16,
             '        "subsection": "Product Specifications",\n'
             '        "features": ["Color", "Size", "Material"]',
             '        "subsection": "제품 사양",\n'
             '        "features": ["색상", "크기", "재질"]',
             "Step 2 출력 예시 한국어화"),
        Rule(16,
             '        "subsection": "Product Identification",\n'
             '        "features": ["Brand", "Release Year"]',
             '        "subsection": "제품 식별 정보",\n'
             '        "features": ["브랜드", "출시연도"]',
             "Step 2 출력 예시 한국어화"),

        Rule(20,
             'STEP_3_PROMPT = """Objective:\n'
             'Extract and list details from the source text using the provided features.\n'
             '\n'
             'Task:\n'
             '1. Extract relevant details from the source text.\n'
             '2. List the extracted details using the provided JSON formats.\n'
             '\n'
             'Key Considerations:\n'
             '1. Ensure the extracted details are accurate and complete.\n'
             '2. Organize the listed details clearly and properly using the specified JSON structure.\n'
             '\n'
             'Example of Desired Output:\n'
             '\n'
             '[\n'
             '    {"feature_type": "Color", "value": "yellow"},\n'
             '    {"feature_type": "Brand", "value": "APPLE"},\n'
             '    {"feature_type": "Release Year", "value": "2000"},\n'
             ']\n'
             '"""',

             'STEP_3_PROMPT = """목표:\n'
             '주어진 속성 목록을 이용해, 원문에서 실제 값을 뽑아내세요.\n'
             '\n'
             '할 일:\n'
             '1. 원문에서 각 속성에 해당하는 값을 찾습니다.\n'
             '2. 찾은 값을 JSON 형식으로 정리합니다.\n'
             '\n'
             '주의할 점:\n'
             '1. 원문에 없는 값은 만들어내지 마세요. 없으면 그 속성은 빼세요.\n'
             '2. 지정된 JSON 구조를 그대로 지키세요.\n'
             '3. **모든 출력은 한국어로 작성하세요.** 단, 브랜드명이나 모델명 같은 고유명사는 원문 그대로 두세요.\n'
             '\n'
             '출력 예시:\n'
             '\n'
             '[\n'
             '    {"feature_type": "색상", "value": "노란색"},\n'
             '    {"feature_type": "브랜드", "value": "APPLE"},\n'
             '    {"feature_type": "출시연도", "value": "2000"},\n'
             ']\n'
             '"""',
             "프롬프트 한국어화 — Step 3 (값 추출). 고유명사는 원문 유지하도록 명시"),

        Rule(26,
             'STEP_4_PROMPT = """Objective:\n'
             'Categorize potential buyers of this product based on their intended use or purposes.\n'
             'Limit 3 to 4 Categories\n'
             '\n'
             'Task:\n'
             '1. Identify distinct categories of users who are likely to purchase this product.\n'
             '2. List the user categories using the provided JSON format.\n'
             '\n'
             'Key Considerations:\n'
             '1. Use common sense and logical reasoning to group users into meaningful categories.\n'
             '2. Ensure the category explanations are clear and concise.\n'
             '3. Organize the listed categories in a clear and proper JSON structure, as shown below:',

             'STEP_4_PROMPT = """목표:\n'
             '이 상품을 살 만한 사람들을 사용 목적에 따라 분류하세요.\n'
             '3~4개 그룹으로 제한합니다.\n'
             '\n'
             '할 일:\n'
             '1. 이 상품을 구매할 만한 사용자 유형을 구분합니다.\n'
             '2. 각 유형을 JSON 형식으로 정리합니다.\n'
             '\n'
             '주의할 점:\n'
             '1. 상품 특성에 비추어 말이 되는 그룹으로 나누세요.\n'
             '2. 각 그룹 설명은 짧고 명확하게 쓰세요.\n'
             '3. **모든 출력은 한국어로 작성하세요.**\n'
             '4. 아래 JSON 구조를 그대로 지키세요:',
             "프롬프트 한국어화 — Step 4 (구매자 유형 분류)"),

        Rule(35,
             'FACTUAL_SUMMARY_PROMPT = """Given Json Format inputs, condense the product information into a concise, factual one line text. Please follow these example cases below.\n'
             '\n'
             '{"summary": "<Short Feature Express>: <Detailed One line Summary>"}\n'
             '\n'
             'Please output as a json format"""',

             'FACTUAL_SUMMARY_PROMPT = """JSON 형식의 상품 정보가 주어집니다.\n'
             '이를 사실 위주의 한 줄 문장으로 압축하세요.\n'
             '\n'
             '아래 형식을 지키세요.\n'
             '{"summary": "<핵심 특징 한마디>: <한 줄 상세 요약>"}\n'
             '\n'
             '**모든 출력은 한국어로 작성하세요.**\n'
             'JSON 형식으로만 출력하세요."""',
             "프롬프트 한국어화 — 사실 요약"),

        Rule(42,
             'FEATURED_SUMMARY_PROMPT = """Given inputs, make multiple lines of feature summaries in English only and provide only the summarized sentence as the output.\n'
             '\n'
             'Please output as a json format"""',

             'FEATURED_SUMMARY_PROMPT = """주어진 정보를 바탕으로 상품 특징 요약문을 여러 줄로 작성하세요.\n'
             '요약 문장만 출력하고 다른 설명은 붙이지 마세요.\n'
             '\n'
             '**모든 출력은 한국어로 작성하세요.**\n'
             'JSON 형식으로만 출력하세요."""',
             "프롬프트 한국어화 — 고객별 요약. 원본은 'in English only' 로 못박혀 있었다"),

        # ── 프롬프트에 넘기는 '내용' 도 한국어로 ──────────────────────
        # 프롬프트 6종은 한국어로 바꿨는데, 정작 그 프롬프트에 붙여 보내는 본문이
        # 영어 레이블이었다. 모델은 "Consumer Type:" 을 보고 영어로 답할 유인이 생긴다.
        # 한국어화가 절반만 된 상태였다. 두 곳에 있고 문구가 미묘하게 다르다.
        Rule(47,
             'content = f"Consumer Type: {k}; Consumer Objectives: {v}; '
             'Related Feature Summaries: {feature_summaries}"',
             'content = f"구매자 유형: {k}\\n구매 목적: {v}\\n관련 특징 요약:\\n{feature_summaries}"',
             "프롬프트에 넘기는 본문 레이블 한국어화 (단계별 확인용 셀)"),

        Rule(53,
             'content = f"Consumer Type: {k}; Consumer Objectives: {v}; '
             'Feature Summaries: {feature_summaries}"',
             'content = f"구매자 유형: {k}\\n구매 목적: {v}\\n제품 특징 요약:\\n{feature_summaries}"',
             "프롬프트에 넘기는 본문 레이블 한국어화 (전체 파이프라인 셀)"),

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
    ], appends=[AMAZON_PROBE]),

    # =====================================================================
    Migration("HPC_BM25_RAG실습.ipynb", [
        Rule(3,
             "%pip install llama-index\n%pip install llama-index-retrievers-bm25\n"
             "%pip install llama-index-llms-vllm\n%pip install datasets\n%pip install vllm",
             "# vLLM 은 별도 venv 에서 서버로 띄우고 여기서는 HTTP 로 붙는다.\n"
             "# llama-index-llms-vllm(in-process) 대신 openai-like 를 쓴다.\n"
             "# 주의: %pip 매직에서 백슬래시 줄바꿈은 불안정하다. 한 줄로 쓴다.\n"
             "%pip install -q -U llama-index llama-index-retrievers-bm25 "
             "llama-index-llms-openai-like datasets PyStemmer matplotlib",
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
        Rule(27,
             "sub_query_engine = SubQuestionQueryEngine.from_defaults(\n"
             "    query_engine_tools=query_engine_tools,\n"
             "    use_async=True,\n"
             ")",
             "# SubQuestionQueryEngine 은 기본값으로 OpenAI 전용 질문 생성기를 찾는다.\n"
             "# 없으면 ImportError 로 죽는데, 그 패키지(llama-index-question-gen-openai)는\n"
             "# llama-index-core<0.13 에 묶여 있어 설치하면 core 가 0.12 로 끌려 내려간다.\n"
             "# 대신 LLM 기반 범용 생성기를 직접 넘긴다. Settings.llm(로컬 vLLM)을 그대로 쓴다.\n"
             "from llama_index.core.question_gen import LLMQuestionGenerator\n\n"
             "sub_query_engine = SubQuestionQueryEngine.from_defaults(\n"
             "    query_engine_tools=query_engine_tools,\n"
             "    question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),\n"
             "    use_async=True,\n"
             ")",
             "SubQuestionQueryEngine 이 OpenAI 전용 생성기를 요구 → 범용 LLMQuestionGenerator "
             "직접 전달. 해당 패키지는 core 를 다운그레이드시켜 쓸 수 없다 (실행 검증에서 확인)"),
        Rule(29,
             "print(sub_query_prompts['question_gen:question_gen_prompt'].template)",
             "# 기본 프롬프트를 먼저 본다. 영어이고 예시도 Uber/Lyft 재무제표다.\n"
             "print(sub_query_prompts['question_gen:question_gen_prompt'].template)\n\n"
             "# ⚠️ 이대로 두면 한국어 질의가 영어 서브질문으로 바뀐다.\n"
             "#    실제로 \"맥스웰 방정식이 무엇인가요?\" 가 \"What is Maxwell's equation?\" 이 됐다.\n"
             "#    BM25 는 어휘를 그대로 매칭하는 검색이라, 영어로 물으면 한국어 위키에서\n"
             "#    아무것도 못 찾고 엉뚱한 문서를 가져온다.\n"
             "#    refine_template 을 바꿨던 것과 같은 방식으로 이것도 한국어로 바꾼다.\n"
             "from llama_index.core import PromptTemplate\n\n"
             "KO_QUESTION_GEN = \"\"\"사용자 질문과 도구 목록이 주어집니다.\n"
             "전체 질문에 답하는 데 필요한 하위 질문들을 JSON 으로 출력하세요.\n"
             "**하위 질문은 반드시 한국어로 작성하세요.**\n\n"
             "# 예시\n"
             "<도구>\n"
             "```json\n"
             "{{\n"
             '    "wiki_2020": "2020년 위키백과 문서를 제공한다",\n'
             '    "wiki_2021": "2021년 위키백과 문서를 제공한다"\n'
             "}}\n"
             "```\n\n"
             "<사용자 질문>\n"
             "2020년과 2021년의 인구 변화를 비교해줘\n\n"
             "<출력>\n"
             "```json\n"
             "{{\n"
             '    "items": [\n'
             '        {{"sub_question": "2020년 인구는 얼마인가", "tool_name": "wiki_2020"}},\n'
             '        {{"sub_question": "2021년 인구는 얼마인가", "tool_name": "wiki_2021"}}\n'
             "    ]\n"
             "}}\n"
             "```\n\n"
             "# 실제 질문\n"
             "<도구>\n"
             "```json\n"
             "{tools_str}\n"
             "```\n\n"
             "<사용자 질문>\n"
             "{query_str}\n\n"
             "<출력>\n"
             "\"\"\"\n\n"
             "sub_query_engine.update_prompts(\n"
             '    {"question_gen:question_gen_prompt": PromptTemplate(KO_QUESTION_GEN)}\n'
             ")\n"
             "print(sub_query_engine.get_prompts()['question_gen:question_gen_prompt'].get_template()[:200])",
             "서브질문이 영어로 생성돼 BM25 한국어 검색이 실패한다 → 질문 생성 프롬프트도 "
             "한국어로 교체 (실행 검증에서 확인)"),
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


def make_cell(kind: str, source: str) -> dict:
    """노트북 셀 dict 를 만든다. build_*_notebook.py 와 같은 형태를 유지한다."""
    cell = {"cell_type": kind, "metadata": {},
            "source": source.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def apply(mig: Migration, dry: bool) -> tuple[int, int, list[str]]:
    src_path = find_source(mig.name)
    out_path = work_path(mig.name)

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
                f"\n[중단] {mig.rel} 전역 치환 대상을 찾지 못했습니다.\n"
                f"  사유: {g.why}\n  찾던 문자열:\n    {g.old[:200]!r}\n"
                f"  → 원본이 바뀌었는지 확인하세요."
            )
        if hits != g.expect:
            raise SystemExit(
                f"\n[중단] {mig.rel} 전역 치환 횟수 불일치.\n"
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
            raise SystemExit(f"[중단] {mig.rel} 셀 {r.cell} 범위 초과 (총 {len(cells)})")
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
                f"\n[중단] {mig.rel} 셀 {r.cell} 에서 대상 문자열을 찾지 못했습니다.\n"
                f"  사유: {r.why}\n"
                f"  찾던 문자열:\n    {r.old[:200]!r}\n"
                f"  실제 셀 내용:\n    {src[:400]!r}\n"
                f"  → 규칙표를 실제 내용에 맞게 고치세요. 조용히 넘어가지 않습니다."
            )

    # --- 셀 추가는 맨 마지막 ---
    # Rule 을 다 적용한 뒤에 붙인다. 순서를 바꾸면 Rule 의 cell 번호가 밀린다.
    for ap_ in mig.appends:
        if ap_.after_cell != len(cells):
            raise SystemExit(
                f"\n[중단] {mig.rel} 셀 추가 위치가 맞지 않습니다.\n"
                f"  사유: {ap_.why}\n"
                f"  예상 마지막 셀 {ap_.after_cell}, 실제 {len(cells)}\n"
                f"  → 원본이 바뀌었습니다. after_cell 을 확인하세요."
            )
        for kind, source in ap_.cells:
            cells.append(make_cell(kind, source))
        applied += 1
        log.append(f"  [추가] 셀 {len(ap_.cells)}개 · {ap_.why}")

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

    # 일자를 재배치했으면 옛 위치에 사본이 남는다. 그대로 두면 같은 노트북이 둘이 되어
    # 수강생이 어느 쪽을 열지 알 수 없다. 배치는 layout.py 가 단독으로 정한다.
    if not args.dry_run:
        print("\n── 배치 정리")
        if not prune_stale():
            print("  (정리할 것 없음)")

    total = 0
    for mig in MIGRATIONS:
        a, _s, log = apply(mig, args.dry_run)
        total += a
        note = "" if log else "  (변경 없음 — 통과 복사)"
        print(f"\n── {mig.rel}  (적용 {a}){note}")
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
