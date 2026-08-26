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
from nbcommon import DATA_DIR_CODE

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
class InsertCells:
    """지정한 셀 **앞에** 셀을 끼워넣는다.

    Rule 은 셀 안의 치환만, AppendCells 는 끝에만 붙는다. 그런데 헤딩이 거의 없는
    노트북(분류 2셀 · NER 2셀 · RAG 4셀)에는 설명을 **중간에** 넣어야 한다.

    중간 삽입은 그동안 일부러 막아뒀다 — 뒤 셀 번호가 밀려 Rule 의 규칙표가
    조용히 어긋나기 때문이다. 그래서 AppendCells 와 같은 규율을 쓴다:

        **모든 Rule 이 끝난 뒤, 위치 역순으로** 적용한다.

    그러면 `before_cell` 을 원본 번호로 적을 수 있고, Rule 의 `cell` 번호도
    영향을 받지 않는다. 역순이라 앞쪽 삽입이 뒤쪽 위치를 밀지도 않는다.

    `expect_head` 는 앵커가 맞는지 확인하는 안전장치다. `AppendCells.after_cell`
    이 실제로 해줬던 역할과 같다 — 원본이 바뀌면 조용히 엉뚱한 데 붙지 않고 멈춘다.
    """
    before_cell: int               # **원본 노트북 기준** 셀 번호. 이 셀 앞에 넣는다
    cells: list[tuple[str, str]]   # ("markdown" | "code", source)
    why: str
    expect_head: str               # 앵커 셀의 첫 줄이 이걸로 시작해야 한다


@dataclass
class Migration:
    name: str                      # 노트북 파일명만. 일자는 layout.py 가 정한다.
    rules: list[Rule] = field(default_factory=list)
    globals_: list[GlobalRule] = field(default_factory=list)
    appends: list[AppendCells] = field(default_factory=list)
    inserts: list[InsertCells] = field(default_factory=list)
    drops: list[int] = field(default_factory=list)   # 지울 셀 번호(원본 기준). 빈 셀 정리용

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
    why="경로 규약 + 산출물 구조 확인 셀 — 학습 데이터 변환(D-3)을 쓰기 전에 실제 형태를 본다",
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
        ("code", DATA_DIR_CODE),
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
# Amazon 산출물 -> 3일차 SFT 학습 데이터 (D-3).
#
# 구조는 verify/06 amazon 실행 결과로 확정했다 (2026-08-26):
#   summary_by_users      -> json.loads -> {구매자유형(한국어): '{"summary": "..."}'}
#   summary_by_subsection -> json.loads -> {속성그룹명(한국어): '{"summary": "..."}'}
# 값이 **또 JSON 문자열**이라 json.loads 를 두 번 해야 한다.
AMAZON_TO_SFT = AppendCells(
    after_cell=58,   # AMAZON_PROBE 가 3셀(경로+확인2)을 먼저 붙인 뒤
    why="학습 데이터 변환 셀 — instruction/output JSONL (D-3)",
    cells=[
        ("markdown", '''
---

## 학습 데이터로 바꾸기

지금까지 만든 것은 **요약**입니다. 이걸 내일 파인튜닝(SFT)의 학습 데이터로 씁니다.

SFT 는 `instruction`(무엇을 하라) 과 `output`(정답) 쌍을 먹습니다.
우리에게는 상품 설명과 요약이 있으니 이렇게 짝지으면 됩니다.

```
instruction : "다음 상품 설명을 읽고 '<구매자 유형>' 관점에서 요약하세요."  + 상품 설명
output      : 그 유형에 대해 만든 요약
```

상품 하나에서 구매자 유형 3~4개 + 속성 그룹 3~5개가 나오므로,
**10개 상품이 수십 건의 학습 예시**가 됩니다.

### 세 가지를 조심해야 합니다

1. **JSON 이 두 겹입니다.** 위 구조 확인에서 봤듯이 `json.loads` 를 두 번 해야
   `summary` 에 닿습니다.
2. **`'error'` 는 파싱을 통과합니다.** 실패한 호출이 돌려준 `'error'` 가
   `json.dumps` 로 감싸이면 `'"error"'` 가 되어 파싱은 성공하고 **문자열**이 나옵니다.
   그러면 `obj["summary"]` 에서 `KeyError` 가 아니라 **`TypeError`** 가 납니다.
3. **상품 설명을 잘라야 합니다.** SFT 의 `max_length` 에 걸리면 **정답이 잘려나가고**
   loss 는 낮은데 아무것도 안 배우는 상태가 됩니다. 조용히 실패하는 종류입니다.
4. **원본의 고유명사가 깨져 있습니다.** `- Brand: F, a, t,  , S, h, a, r, k` 처럼
   글자가 쉼표로 쪼개져 있습니다. 이대로 학습시키면 모델이 그 패턴을 배웁니다.
   붙여서 복원하고 넣습니다.
'''),
        ("code", '''
MAX_SRC = 1200   # 상품 설명을 이만큼만 쓴다. 내일 SFT 의 max_length 에 여유를 둔다


def restore_split_names(text):
    """글자가 쉼표로 쪼개진 값을 붙인다.

        - Brand: F, a, t,  , S, h, a, r, k   ->   - Brand: Fat Shark

    원본 데이터의 결함이다. 이대로 학습시키면 모델이
    "브랜드명은 글자를 쉼표로 나눠 쓴다" 를 배운다.

    조각이 3개 이상이고 **전부 한 글자 이하**일 때만 고친다.
    'color, size, weight' 같은 정상 목록은 건드리지 않는다.
    """
    out = []
    for line in text.splitlines(keepends=True):
        head, sep, val = line.partition(":")
        if sep and head.lstrip().startswith("-"):
            parts = [q.strip() for q in val.split(",")]
            if len(parts) >= 3 and all(len(q) <= 1 for q in parts):
                tail = val[len(val.rstrip()):]          # 줄 끝 개행을 보존한다
                line = head + ": " + "".join(q or " " for q in parts).strip() + tail
        out.append(line)
    return "".join(out)


def unwrap(raw):
    """이중 인코딩을 풀어 summary 문자열을 꺼낸다. 못 꺼내면 None."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    # 'error' 가 감싸이면 obj 가 dict 가 아니라 str 이 된다.
    # 그때 obj["summary"] 는 TypeError 이므로 dict 인지 먼저 본다.
    return obj.get("summary") if isinstance(obj, dict) else None


# 이스케이프를 쓰지 않으려고 삼중따옴표 템플릿으로 둔다.
# (여러 겹의 문자열을 거치면서 백슬래시가 한 겹씩 사라져 실제로 두 번 깨졌다)
INSTRUCTION = """{task}

[상품 설명]
{src}"""

TEMPLATES = [
    ("summary_by_users",
     "다음 상품 설명을 읽고 '{k}' 관점에서 핵심을 요약하세요."),
    ("summary_by_subsection",
     "다음 상품 설명에서 '{k}' 에 해당하는 내용을 한 줄로 요약하세요."),
]

records, dropped = [], 0

for row in output_data:
    src = restore_split_names(row["text"])[:MAX_SRC]
    for col, tmpl in TEMPLATES:
        try:
            outer = json.loads(row[col])
        except (json.JSONDecodeError, TypeError):
            dropped += 1
            continue
        if not isinstance(outer, dict):
            dropped += 1
            continue

        for k, v in outer.items():
            try:
                summary = unwrap(v)
            except (json.JSONDecodeError, TypeError):
                summary = None
            if not summary:
                dropped += 1          # 'error' 나 빈 값
                continue
            records.append({
                "instruction": INSTRUCTION.format(task=tmpl.format(k=k), src=src),
                "output": summary,
            })

# 출력에 'error' 라는 단어를 쓰지 않는다 — 검증 스크립트가 그 단어를 세기 때문이다.
print(f"학습 예시 {len(records)}건 · 제외 {dropped}건")
print(f"  상품 {len(output_data)}개에서 나왔습니다.")
'''),
        ("code", '''
if records:
    print("=" * 72)
    print("[instruction]")
    print(records[0]["instruction"][:300], "...")
    print()
    print("[output]")
    print(records[0]["output"])
    print("=" * 72)
    print()

    lens = sorted(len(r["instruction"]) + len(r["output"]) for r in records)
    print(f"글자 수 — 중앙값 {lens[len(lens)//2]:,} · 최대 {lens[-1]:,}")
    print("  내일 SFT 에서 토큰 길이 분포를 다시 확인합니다.")
'''),
        ("code", '''
out_path = DATA_DIR / "amazon_ko_sft.mine.jsonl"
with out_path.open("w", encoding="utf-8") as f:
    for r in records:
        # ensure_ascii=False 가 없으면 한글이 escape 되어 눈으로 확인할 수 없다
        print(json.dumps(r, ensure_ascii=False), file=f)

print(f"저장: {out_path}  ({out_path.stat().st_size/1024:.0f} KB)")
print()
print("내일 SFT 실습이 이 파일을 읽습니다.")
print("강사가 미리 만들어 둔 assets/amazon_ko_sft.jsonl.gz 와 합쳐서 학습합니다.")
'''),
        ("markdown", '''
### 여기서 만든 것이 내일로 이어집니다

| 오늘 만든 것 | 내일 쓰는 곳 |
|---|---|
| `amazon_ko_sft.mine.jsonl` | 3일차 **SFT 학습 데이터** |

수강생이 직접 만든 것은 10개 상품 분량이라 학습에는 적습니다.
그래서 강사가 미리 300개 상품으로 돌려둔 파일과 **합쳐서** 씁니다.
직접 만든 것이 그 안에 들어가 있다는 점이 중요합니다 —
**남의 데이터가 아니라 내가 만든 데이터로 학습**하게 됩니다.

> 어제 프롬프트만으로 점수를 얼마나 올렸는지 기억하세요.
> 내일은 같은 일을 **학습으로** 해보고, 그 차이가 값어치가 있는지 따집니다.
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


TRUNC_OLD = "        return completion.choices[0].message.content\n    except Exception as e:\n        print(e)\n        return 'error'"
TRUNC_NEW = '        # 길이 상한에 걸리면 JSON 이 중간에서 끊긴다. 그런데 API 는 **성공으로**\n        # 응답하므로 아래 except 에 걸리지 않고, 몇 셀 뒤에서 json.loads 가\n        # "Unterminated string" 으로 죽는다. 원인에서 먼 곳에서 터지는 것이 가장 나쁘다.\n        if completion.choices[0].finish_reason == "length":\n            print("[잘림] 출력이 max_tokens 에 걸렸습니다. "\n                  "스키마의 max_length 나 max_tokens 를 늘려야 합니다.")\n            return \'error\'\n        return completion.choices[0].message.content\n    except Exception as e:\n        print(e)\n        return \'error\''


# 중복 제거 — 실제 개행이 든 상수로 둔다 (백슬래시 이스케이프 회피, CLAUDE.md §12)
SFT_DEDUP_OLD = 'raw_datasets = load_dataset("json", data_files=[str(p) for p in paths])'

SFT_DEDUP_NEW = SFT_DEDUP_OLD + """

    # 어제 직접 만든 10건은 강사 사전생성본 안에 이미 들어 있다.
    # 그대로 합치면 앞쪽 상품이 두 번 학습된다. 같은 (instruction, output) 은 하나만 남긴다.
    from datasets import Dataset, DatasetDict

    _rows, _seen = [], set()
    for r in raw_datasets["train"]:
        key = (r["instruction"], r["output"])
        if key not in _seen:
            _seen.add(key)
            _rows.append({"instruction": r["instruction"], "output": r["output"]})
    _dup = len(raw_datasets["train"]) - len(_rows)
    raw_datasets = DatasetDict({"train": Dataset.from_list(_rows)})
    if _dup:
        print(f"  겹치는 {_dup}건은 하나로 합쳤습니다")"""

# 토큰 길이 확인 — 백슬래시 이스케이프를 쓰지 않으려고 실제 개행이 든 상수로 둔다.
# (여러 겹 문자열을 거치며 백슬래시가 사라지는 사고를 여러 번 겪었다. CLAUDE.md §12)
SFT_SPLIT_OLD = """# create the splits
train_dataset = raw_datasets["train"]
eval_dataset = raw_datasets["test"]"""

SFT_SPLIT_NEW = SFT_SPLIT_OLD + """

# ── 학습 전에 토큰 길이를 본다 ──────────────────────────────────
# ★ max_length 에 걸리면 뒤가 잘린다. 그런데 잘리는 쪽은 **정답**이다 —
#   instruction 이 앞에 오기 때문이다. 그러면 loss 는 낮게 나오면서
#   아무것도 안 배우는 상태가 되고, 에러도 나지 않는다.
#   조용히 실패하는 종류라 반드시 확인하고 넘어간다.
_lens = sorted(len(tokenizer.encode(t)) for t in train_dataset["text"])
_limit = tokenizer.model_max_length
_over = sum(1 for n in _lens if n > _limit)

print()
print(f"토큰 길이 — 중앙값 {_lens[len(_lens)//2]:,} · "
      f"상위10% {_lens[int(len(_lens)*0.9)]:,} · 최대 {_lens[-1]:,}")
print(f"상한 {_limit:,} 초과: {_over}건 ({_over/len(_lens):.1%})")
if _over:
    print("  ★ 초과분은 뒤가 잘립니다. instruction 이 앞이므로 정답이 날아갑니다.")
    print("    Amazon 노트북의 MAX_SRC 를 줄이거나 model_max_length 를 늘리세요.")
else:
    print("  → 잘리는 예시 없음")"""


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
        # ── Phase E: 어제 만든 데이터로 학습한다 ────────────────────
        # 2일차 Amazon 실습이 만든 것 + 강사 사전생성본을 합쳐 읽는다.
        # 둘 다 없으면 kullm-v2 로 폴백하되 **조용히 넘어가지 않고 안내한다** —
        # 폴백인 줄 모르고 "내 데이터로 학습했다" 고 오해하면 안 된다.
        Rule(5,
             'from datasets import load_dataset\n'
             '\n'
             'raw_datasets = load_dataset("beomi/KoAlpaca-v1.1a")',

             'import os\n'
             'from pathlib import Path\n'
             '\n'
             'from datasets import load_dataset\n'
             '\n'
             '\n'
             'def _repo_root():\n'
             '    """cwd 에서 위로 올라가며 repo 루트를 찾는다. setup_vessl.sh 를 표지로 쓴다."""\n'
             '    p = Path.cwd().resolve()\n'
             '    return next((c for c in [p, *p.parents] if (c / "setup_vessl.sh").exists()), None)\n'
             '\n'
             '\n'
             '_root = _repo_root()\n'
             'DATA_DIR = Path(os.environ["HPC_DATA"]) if "HPC_DATA" in os.environ else (\n'
             '    _root / "data" if _root else None)\n'
             'ASSETS_DIR = (_root / "assets") if _root else None\n'
             '\n'
             '# 어제 직접 만든 것 + 강사가 미리 만들어 둔 것\n'
             'paths = [p for p in [\n'
             '    ASSETS_DIR / "amazon_ko_sft.jsonl.gz" if ASSETS_DIR else None,\n'
             '    DATA_DIR / "amazon_ko_sft.mine.jsonl" if DATA_DIR else None,\n'
             '] if p is not None and p.exists()]\n'
             '\n'
             'if paths:\n'
             '    raw_datasets = load_dataset("json", data_files=[str(p) for p in paths])\n'
             '    print(f"내가 만든 데이터로 학습합니다 — {len(raw_datasets[\'train\']):,}건")\n'
             '    for p in paths:\n'
             '        print(f"  · {p.name}")\n'
             'else:\n'
             '    # ★ 조용히 넘어가면 폴백인 줄 모른다. 크게 알린다.\n'
             '    print("=" * 60)\n'
             '    print("★ 생성 데이터가 없어 남의 데이터(kullm-v2)로 대체합니다.")\n'
             '    print("  2일차 HPC_Amazon요약실습 을 먼저 돌리면")\n'
             '    print("  직접 만든 데이터로 학습할 수 있습니다.")\n'
             '    print("=" * 60)\n'
             '    raw_datasets = load_dataset("nlpai-lab/kullm-v2")',
             "어제 만든 데이터로 학습하게 한다. 없으면 kullm-v2 폴백 (Phase E)"),

        Rule(6,
             'indices = range(0,1000)\n'
             'test_indices = range(1000, 1050)',

             '# 데이터가 얼마나 있을지 모르므로 건수로 자르지 않고 비율로 나눈다.\n'
             '# (고정 인덱스로 잘랐다가 평가셋이 0건이 된 적이 있다)\n'
             'n_all = len(raw_datasets["train"])\n'
             'n_train = min(1000, int(n_all * 0.9))\n'
             'n_test = min(50, n_all - n_train)\n'
             '\n'
             'indices = range(0, n_train)\n'
             'test_indices = range(n_train, n_train + n_test)\n'
             'print(f"전체 {n_all:,}건 → 학습 {n_train:,} / 평가 {n_test}")',
             "건수 하드코딩 제거 — 데이터 크기가 바뀌면 select 가 깨진다 (Phase E)"),
        Rule(5, SFT_DEDUP_OLD, SFT_DEDUP_NEW,
             "직접 만든 10건이 사전생성본에 이미 들어 있어 중복 학습된다 (Phase E)"),

        Rule(15, SFT_SPLIT_OLD, SFT_SPLIT_NEW,
             "학습 전 토큰 길이 확인 — 잘리면 정답이 날아가고 조용히 실패한다 (Phase E)"),

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

        # ── 구조화 출력 안전장치 (2026-08-26 실행 검증에서 드러남) ──────────
        # 증상: 셀 53 summarize_by_users 에서
        #   JSONDecodeError: Unterminated string starting at: line 4 column 24 (char 58)
        # char 58 은 Step 4 출력의 **첫 user_category 값** 위치와 정확히 일치한다.
        # 제약 디코딩이 그 문자열을 쓰다가 max_tokens 에 걸려 JSON 이 통째로 잘린 것이다.
        #
        # 원인이 둘 겹쳐 있었다.
        #   1. 모든 str 필드에 **길이 상한이 없다.** 문법상 무한히 길어질 수 있다
        #   2. temperature 를 지정하지 않아 **기본 1.0** 이다. 추출 작업에 너무 높다
        # 영어로 돌 때는 우연히 넘어갔고, 한국어화로 출력이 길어지면서 터졌다.
        Rule(3, "from pydantic import BaseModel",
             "from pydantic import BaseModel, Field",
             "스키마 길이 상한을 걸기 위해 Field 를 들여온다"),
        Rule(3, "from typing import List",
             "from typing import Annotated, List",
             "리스트 원소의 길이 상한에는 Annotated 가 필요하다"),

        Rule(13,
             "class FeatureType(BaseModel):\n"
             "    feature_type: str\n"
             "    descript: str\n"
             "\n"
             "class FeatureTypeList(BaseModel):\n"
             "    feature_type_list: List[FeatureType]",
             "# 길이 상한이 없으면 제약 디코딩이 문자열을 끝없이 늘릴 수 있고,\n"
             "# 그러면 max_tokens 에서 잘려 JSON 자체가 깨진다 (실행 검증에서 확인).\n"
             "class FeatureType(BaseModel):\n"
             "    feature_type: str = Field(max_length=40)\n"
             "    descript: str = Field(max_length=200)\n"
             "\n"
             "class FeatureTypeList(BaseModel):\n"
             "    feature_type_list: List[FeatureType] = Field(max_length=20)",
             "스키마 길이 상한 — Step 1 (D-11)"),
        Rule(13, "              temperature=0.8,\n              top_p=0.95,",
             "              temperature=0.2,   # 추출 작업이다. 높으면 문자열이 늘어진다\n"
             "              max_tokens=2048,   # 원본은 상한이 아예 없었다\n"
             "              top_p=0.95,",
             "temperature 하향 + max_tokens 부여 — Step 1 (D-11)"),

        Rule(17,
             "class Subsection(BaseModel):\n"
             "    subsection: str\n"
             "    features: List[str]\n"
             "\n"
             "class SubsectionList(BaseModel):\n"
             "    subsection_list: List[Subsection]",
             "class Subsection(BaseModel):\n"
             "    subsection: str = Field(max_length=40)\n"
             "    features: List[Annotated[str, Field(max_length=60)]] = Field(max_length=20)\n"
             "\n"
             "class SubsectionList(BaseModel):\n"
             "    subsection_list: List[Subsection] = Field(max_length=10)",
             "스키마 길이 상한 — Step 2 (D-11)"),
        Rule(17, "              max_tokens=1024,\n              top_p=0.95,\n              seed=777,",
             "              max_tokens=2048,\n              temperature=0.2,\n"
             "              top_p=0.95,\n              seed=777,",
             "temperature 하향 + max_tokens 상향 — Step 2 (D-11)"),

        Rule(21,
             "class ExtractedFeature(BaseModel):\n"
             "    feature_type: str\n"
             "    value: str\n"
             "\n"
             "class ExtractedFeatureList(BaseModel):\n"
             "    feature_list: List[ExtractedFeature]",
             "class ExtractedFeature(BaseModel):\n"
             "    feature_type: str = Field(max_length=40)\n"
             "    value: str = Field(max_length=200)\n"
             "\n"
             "class ExtractedFeatureList(BaseModel):\n"
             "    feature_list: List[ExtractedFeature] = Field(max_length=30)",
             "스키마 길이 상한 — Step 3 (D-11)"),
        Rule(21, "              max_tokens=1024,\n              top_p=0.95,\n              seed=0,",
             "              max_tokens=2048,\n              temperature=0.2,\n"
             "              top_p=0.95,\n              seed=0,",
             "temperature 하향 + max_tokens 상향 — Step 3 (D-11)"),

        Rule(27,
             "class ConsumerCategory(BaseModel):\n"
             "    user_category: str\n"
             "    describe: str\n"
             "\n"
             "class ConsumerCategoryList(BaseModel):\n"
             "    consuber_categories: List[ConsumerCategory]",
             "# ★ 여기가 실제로 터진 곳이다. user_category 에 상한이 없어서\n"
             "#   제약 디코딩이 그 문자열을 쓰다가 max_tokens 에 걸렸다.\n"
             "class ConsumerCategory(BaseModel):\n"
             "    user_category: str = Field(max_length=40)\n"
             "    describe: str = Field(max_length=200)\n"
             "\n"
             "class ConsumerCategoryList(BaseModel):\n"
             "    # 프롬프트가 3~4개라고 했으니 스키마로도 못박는다\n"
             "    consuber_categories: List[ConsumerCategory] = Field(max_length=6)",
             "스키마 길이 상한 — Step 4. 실패 지점 (D-11)"),
        Rule(27, "              max_tokens=1024, top_p=0.95, seed=1234,",
             "              max_tokens=1024, temperature=0.2, top_p=0.95, seed=1234,",
             "temperature 하향 — Step 4 (D-11)"),

        Rule(36,
             "class SummaryDescription(BaseModel):\n    summary: str",
             "class SummaryDescription(BaseModel):\n"
             "    summary: str = Field(max_length=400)   # 한 줄 요약이다",
             "스키마 길이 상한 — 사실 요약 (D-11)"),
        Rule(36, "              max_tokens=1024, top_p=0.95, seed=1234,",
             "              max_tokens=1024, temperature=0.2, top_p=0.95, seed=1234,",
             "temperature 하향 — 사실 요약 (D-11)"),

        Rule(43,
             "class SummaryDescription(BaseModel):\n    summary: str",
             "class SummaryDescription(BaseModel):\n"
             "    summary: str = Field(max_length=1000)   # 여러 줄이라 더 길게 준다",
             "스키마 길이 상한 — 고객별 요약 (D-11)"),
        Rule(43, "              max_tokens=1024, top_p=0.95, seed=1234,",
             "              max_tokens=2048, temperature=0.2, top_p=0.95, seed=1234,",
             "temperature 하향 + max_tokens 상향 — 고객별 요약 (D-11)"),
    ], [
        GlobalRule(OLD_EXAONE, SERVE_LM,
                   "EXAONE NC 라이선스 → Qwen3 교체 (F-1)", 2),
        GlobalRule(TRUNC_OLD, TRUNC_NEW,
                   "잘린 응답은 예외가 아니라 정상 응답이라 except 에 안 걸린다. "
                   "finish_reason 으로 잡아야 원인 가까이서 드러난다 (D-11)", 6),
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
    ], appends=[AMAZON_PROBE, AMAZON_TO_SFT]),

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

    # --- 중간 삽입은 **맨 마지막에, 위치 역순으로** ---
    # 역순이 아니면 앞쪽 삽입이 뒤쪽 before_cell 을 밀어버린다.
    for ins in sorted(mig.inserts, key=lambda x: -x.before_cell):
        if not (1 <= ins.before_cell <= len(cells)):
            raise SystemExit(
                f"\n[중단] {mig.rel} 삽입 위치 {ins.before_cell} 이 범위를 "
                f"벗어납니다 (총 {len(cells)}).\n  사유: {ins.why}"
            )
        anchor = cell_source(cells[ins.before_cell - 1])
        head = anchor.lstrip().splitlines()[0] if anchor.strip() else ""
        if not head.startswith(ins.expect_head):
            raise SystemExit(
                f"\n[중단] {mig.rel} 셀 {ins.before_cell} 이 예상한 앵커가 아닙니다.\n"
                f"  사유: {ins.why}\n"
                f"  예상 시작: {ins.expect_head!r}\n"
                f"  실제 첫 줄: {head[:120]!r}\n"
                f"  → 원본이 바뀌었거나 앞선 규칙이 셀을 옮겼습니다."
            )
        for off, (kind, source) in enumerate(ins.cells):
            cells.insert(ins.before_cell - 1 + off, make_cell(kind, source))
        applied += 1
        log.append(f"  [삽입] 셀 {ins.before_cell} 앞에 {len(ins.cells)}개 · {ins.why}")

    # --- 삭제도 맨 마지막에, 역순으로 ---
    # 빈 셀처럼 지워도 되는 것만 대상이다. 내용이 있으면 중단한다 —
    # 실수로 코드를 날리는 것이 이 도구에서 가장 위험한 실패다.
    for idx in sorted(mig.drops, reverse=True):
        if not (1 <= idx <= len(cells)):
            raise SystemExit(f"[중단] {mig.rel} 삭제 위치 {idx} 범위 초과 (총 {len(cells)})")
        body = cell_source(cells[idx - 1]).strip()
        if body:
            raise SystemExit(
                f"\n[중단] {mig.rel} 셀 {idx} 는 비어 있지 않습니다. 지우지 않습니다.\n"
                f"  내용: {body[:120]!r}"
            )
        del cells[idx - 1]
        applied += 1
        log.append(f"  [삭제] 빈 셀 {idx}")

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
