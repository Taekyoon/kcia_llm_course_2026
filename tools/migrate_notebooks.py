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
from nbcommon import DATA_DIR_CODE, save_notebook

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

> 이다음 실습(프롬프트 최적화)에서는 학습 없이 프롬프트만으로 점수를 끌어올립니다.
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
        output_dir="data/orpo_model",
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


# ── 분류 실습 설명 (강의 설명 보강) ──────────────────────────────────
# 마크다운이 2셀(48자)뿐이라 InsertCells 로 끼워넣는다.
# 백슬래시 이스케이프를 쓰지 않으려고 실제 개행이 든 상수로 둔다 (CLAUDE.md §12).

CLS_INTRO = """
# 영화 리뷰 감성 분류

문장을 읽고 **긍정인지 부정인지 맞히는** 모델을 만듭니다.

## 무엇을 하게 되나

1. **토크나이저**가 한국어를 어떻게 쪼개는지 봅니다
2. **NSMC**(네이버 영화 리뷰) 데이터를 불러옵니다
3. 문장을 모델이 읽을 수 있는 형태로 **전처리**합니다
4. 인코더 모델에 **분류 헤드**를 붙여 학습시킵니다
5. 학습한 모델로 새 문장을 **분류**해 봅니다

## 왜 인코더인가

앞서 **"Encoder 는 BERT, Decoder 는 GPT"** 를 배웠습니다.
분류는 문장 **전체를 읽고 하나의 답**을 내는 일이라 인코더가 맞습니다.
GPT 처럼 다음 단어를 이어 쓰는 구조가 아니라, 문장을 양방향으로 훑어
**하나의 벡터로 압축**한 뒤 그 위에 분류기를 얹습니다.

여기서 쓰는 `mmBERT-base` 는 다국어로 학습된 인코더입니다.
""".strip()

CLS_TOKENIZER = """
## 1. 토크나이저가 한국어를 어떻게 다루나

모델은 글자를 모릅니다. **숫자(토큰 ID)** 만 다룹니다.
그 변환표가 토크나이저이고, 모델마다 **다릅니다** — 그래서 모델과 토크나이저는
항상 짝으로 씁니다.

아래 두 셀에서 볼 것:

- 한 문장이 **몇 개로 쪼개지는가**. 한국어는 조사와 어미가 붙어 늘어나기 쉽습니다
- `[CLS]` 와 `[SEP]` 가 자동으로 붙는 것. `[CLS]` 자리의 출력이 나중에
  **문장 전체를 대표하는 벡터**로 쓰입니다. 분류 헤드가 붙는 곳이 바로 여기입니다
- 같은 뜻인데 **한국어와 영어의 토큰 수가 다른 것**. 다국어 모델이라도
  언어별로 효율이 같지 않습니다
""".strip()

CLS_DATA = """
## 2. 데이터 — NSMC

**NSMC**(Naver Sentiment Movie Corpus)는 네이버 영화 리뷰에 긍정/부정 라벨을 붙인
한국어 감성 분석의 표준 데이터셋입니다. 학습 15만 · 평가 5만 건입니다.

컬럼은 셋입니다.

| 컬럼 | 내용 |
|---|---|
| `id` | 리뷰 번호 |
| `document` | 리뷰 본문 — 이걸 모델에 넣습니다 |
| `label` | 0 = 부정, 1 = 긍정 |

아래에서 실제 샘플을 하나 열어봅니다. **데이터를 눈으로 보는 것**을 건너뛰지 마세요.
컬럼 이름이 무엇인지, 문장이 얼마나 긴지, 라벨이 어떻게 들어 있는지를 알아야
그다음 전처리를 쓸 수 있습니다.
""".strip()

CLS_PREPROCESS = """
## 3. 전처리 — 토큰화와 패딩

문장을 토큰 ID 로 바꿉니다. 두 가지가 눈에 띌 겁니다.

**`truncation=True` 는 있는데 `padding` 이 없습니다.**
길이를 여기서 맞추지 않고 **배치를 만들 때** 맞추기 때문입니다.
그 일을 하는 것이 다음 셀의 `DataCollatorWithPadding` 입니다.

미리 전체를 가장 긴 문장에 맞춰 패딩하면, 짧은 문장이 대부분인 데이터에서
쓸데없는 패딩 토큰을 잔뜩 계산하게 됩니다. **배치 안에서만** 맞추면
그 낭비가 사라집니다. 실무에서 쓰는 방식입니다.

`batched=True` 는 map 을 한 건씩이 아니라 묶음으로 돌립니다. 훨씬 빠릅니다.
""".strip()

CLS_METRIC = """
## 4. 무엇으로 잘했다고 할 것인가

학습을 시작하기 전에 **평가 기준**을 정합니다. 이게 없으면 학습은 돌아가도
좋아졌는지 알 수 없습니다.

분류는 정답이 하나로 정해지므로 **정확도(accuracy)** 를 씁니다.

`compute_metrics` 가 받는 `eval_pred` 는 `(예측, 정답)` 튜플입니다. Trainer 의 관례입니다.
예측은 아직 확률 이전의 **로짓**이라 `argmax` 로 가장 큰 쪽을 고릅니다.

> 정확도가 늘 적절한 것은 아닙니다. 긍정이 95% 인 데이터라면 전부 "긍정" 이라고
> 답해도 95% 가 나옵니다. NSMC 는 긍정·부정이 거의 반반이라 정확도로 충분합니다.
""".strip()

CLS_LABEL = """
## 5. 라벨 이름 붙이기

모델 내부에서 라벨은 `0`, `1` 입니다. 그대로 두면 나중에 추론 결과가
`LABEL_0` 이라고 나옵니다 — 이게 긍정인지 부정인지 알 수 없습니다.

그래서 **번호와 이름의 대응표**를 만들어 모델에 함께 넣습니다.
그러면 마지막 셀의 `pipeline` 이 `POSITIVE` / `NEGATIVE` 로 답합니다.

사소해 보이지만, 모델을 남에게 넘길 때 이 대응표가 없으면
**받은 사람이 0과 1의 의미를 알 수 없습니다.** 모델 파일에 같이 저장됩니다.
""".strip()

CLS_TRAIN = """
## 7. 학습

인코더 위에 **분류 헤드**(2차원 출력층)를 새로 얹고 전체를 파인튜닝합니다.
헤드는 무작위로 초기화된 상태라 학습이 필요합니다.

### 눈여겨볼 값

| 값 | 왜 이렇게 |
|---|---|
| `learning_rate=2e-5` | 사전학습된 가중치를 **망가뜨리지 않을 만큼** 작게. 파인튜닝 관례값 |
| `num_train_epochs=0.1` | **실습 시간을 줄이려고 일부러 줄인 것.** 15만 건의 10%만 봅니다 |
| `weight_decay=0.01` | 과적합 억제 |
| `load_best_model_at_end` | 평가 점수가 가장 좋았던 체크포인트를 되돌려 놓습니다 |

`num_train_epochs=0.1` 이 가장 이상해 보일 겁니다. **1 에폭도 안 돌립니다.**
그래도 정확도가 꽤 나옵니다 — 사전학습된 모델이 이미 언어를 알고 있어서
분류 헤드만 맞추면 되기 때문입니다. **이것이 파인튜닝의 요점입니다.**

> 실무에서는 1~3 에폭을 돌립니다. 값을 올려보면 정확도가 어떻게 변하는지
> 직접 확인해 보세요. 대신 시간이 그만큼 늘어납니다.
""".strip()

CLS_INFER = """
## 8. 써보기

학습이 끝났으니 실제로 문장을 넣어봅니다.

`pipeline` 은 토큰화 → 모델 → 후처리를 한 번에 묶어주는 편의 도구입니다.
직접 쓰면 앞에서 한 전처리를 다시 손으로 해야 합니다.

체크포인트 경로를 **하드코딩하지 않는 것**에 주의하세요. 배치 크기나 에폭이
조금만 달라져도 체크포인트 번호가 바뀝니다. 방금 학습한 `trainer` 에서
경로를 받아오면 그런 문제가 없습니다.

여러 문장을 바꿔 넣어보세요. **반어법이나 부정문**에서 모델이 어떻게 반응하는지
보면 한계가 드러납니다.
""".strip()

CLS_OUTRO = """
## 마무리

인코더 모델로 한국어 문장 분류기를 만들었습니다.

- **토크나이저와 모델은 짝**입니다. `[CLS]` 자리가 문장 전체를 대표합니다
- **패딩은 배치 단위로** 합니다. 미리 맞추면 계산이 낭비됩니다
- **평가 기준을 먼저 정합니다.** 없으면 좋아졌는지 알 수 없습니다
- 사전학습 모델은 **0.1 에폭만으로도** 쓸 만해집니다. 파인튜닝의 요점입니다

다음 실습에서는 같은 인코더로 **개체명 인식(NER)** 을 합니다.
문장 하나에 답 하나였던 분류와 달리, **토큰마다 답을 내야** 합니다.
그 차이가 코드에서 어떻게 드러나는지 보게 됩니다.
""".strip()


CLS_MODEL = """
## 6. 모델 — 인코더에 분류 헤드를 얹는다

`mmBERT-base` 는 문장을 벡터로 바꿀 줄만 알지, 긍정·부정을 판단할 줄은 모릅니다.
그래서 그 위에 **2차원 출력층**을 새로 붙입니다. `num_labels=2` 가 그 뜻입니다.

| | 상태 |
|---|---|
| 인코더 본체 (307M) | **사전학습됨** — 언어를 이미 안다 |
| 분류 헤드 (2차원) | **무작위 초기화** — 아무것도 모른다 |

학습은 둘 다 건드립니다. 다만 본체는 이미 잘 되어 있으니 **조심스럽게**(작은 학습률로)
건드리고, 헤드는 바닥부터 배웁니다. 이것이 파인튜닝입니다.

> `device_map="auto"` 를 쓰지 않는 것에 주의하세요. 수십 GB 짜리 모델을 여러 GPU 에
> 쪼개 올릴 때 쓰는 옵션인데, 307M 은 한 장에 넉넉히 들어갑니다. 오히려 Trainer 와
> 함께 쓰면 문제가 생길 수 있습니다.
""".strip()


# ── NER 실습 설명 ─────────────────────────────────────────────────────
NER_INTRO = """
# 개체명 인식 (NER)

문장에서 **사람·장소·기관·날짜** 같은 것을 찾아내는 모델을 만듭니다.

```
"안녕하세요 대한민국 서울에 사는 홍길동입니다."
                └ 장소 ┘ └장소┘      └인물┘
```

## 분류와 무엇이 다른가

바로 앞 실습은 문장 하나에 **답이 하나**였습니다. NER 은 **토큰마다 답**을 냅니다.

| | 감성 분류 | 개체명 인식 |
|---|---|---|
| 입력 | 문장 | 문장 |
| 출력 | 라벨 1개 | **토큰 수만큼** |
| 모델 | `...ForSequenceClassification` | `...ForTokenClassification` |

인코더 본체는 **똑같습니다.** 위에 얹는 헤드만 다릅니다.
분류는 문장 전체를 대표하는 `[CLS]` 자리 하나를 보고, NER 은 **모든 자리**를 봅니다.

이 차이 하나가 전처리를 꽤 까다롭게 만듭니다. 그게 이 실습의 핵심입니다.
""".strip()

NER_LABELS = """
## 2. 라벨 체계 — BIO 태그

개체명은 **여러 단어에 걸칩니다.** `대한민국 서울` 은 두 어절이지만 한 덩어리입니다.
그래서 "이 토큰이 개체의 **시작**인지 **이어지는 중**인지" 를 구분해야 합니다.

그 방식이 **BIO 태그**입니다.

| 태그 | 뜻 |
|---|---|
| `B-LC` | 장소(LoCation)의 **B**eginning — 여기서 시작 |
| `I-LC` | 장소의 **I**nside — 앞에서 이어짐 |
| `O` | **O**utside — 개체가 아님 |

KLUE-NER 은 6종을 다룹니다.

| 코드 | 뜻 | | 코드 | 뜻 |
|---|---|---|---|---|
| `PS` | 인물 | | `DT` | 날짜 |
| `LC` | 장소 | | `TI` | 시간 |
| `OG` | 기관 | | `QT` | 수량 |

`B-`/`I-` 각각에 6종 + `O` = **13개**입니다. 아래 셀에서 실제 목록을 확인하세요.

> `.features["ner_tags"].feature.names` 에서 `.feature` 가 한 번 더 들어가는 것은,
> `ner_tags` 가 **리스트**(토큰마다 하나)이기 때문입니다. 바깥은 리스트, 안쪽이 라벨입니다.
""".strip()

NER_ALIGN = """
## 3. 라벨 정렬 — 이 실습에서 가장 까다로운 곳

문제가 하나 있습니다. **라벨은 어절 단위인데 토크나이저는 더 잘게 쪼갭니다.**

```
어절:   홍길동  입니다
라벨:   B-PS    O
토큰:   홍 ##길 ##동 ##입 ##니다        ← 5개
라벨:   ???
```

라벨이 3개인데 토큰이 5개입니다. **다시 맞춰줘야** 합니다.
그 일을 하는 것이 `align_labels_with_tokens` 입니다. 세 가지 규칙이 들어 있습니다.

### 1. 특수 토큰은 `-100`

`[CLS]`, `[SEP]`, 패딩에는 정답이 없습니다. 그렇다고 아무 라벨이나 주면
모델이 그걸 배웁니다.

`-100` 은 PyTorch `CrossEntropyLoss` 의 **`ignore_index` 기본값**입니다.
이 값이 든 자리는 손실 계산에서 **통째로 빠집니다.** 마법의 숫자가 아니라 약속입니다.

### 2. 같은 어절의 두 번째 조각부터는 `B-` 를 `I-` 로

`홍길동` 이 `홍`/`##길`/`##동` 으로 쪼개졌다면, 첫 조각만 `B-PS` 이고
나머지는 `I-PS` 여야 합니다. **개체는 한 번만 시작**하니까요.

```python
if label % 2 == 1:
    label += 1
```

이 한 줄이 그 일을 합니다. 왜 되는지는 **라벨 목록의 배열 순서** 때문입니다.

```
0: O   1: B-DT  2: I-DT   3: B-LC  4: I-LC   5: B-OG  6: I-OG  ...
       └ 홀수 ┘ └ 짝수 ┘
```

`B-` 는 전부 홀수, `I-` 는 전부 짝수이고 **바로 다음 번호**입니다.
그래서 1을 더하면 `B-` → `I-` 가 됩니다.

> **이건 KLUE 의 라벨 순서에 기댄 코드입니다.** 다른 데이터셋에서 순서가 다르면
> 조용히 틀립니다. 위 셀에서 `label_names` 를 눈으로 확인한 이유가 이것입니다.

### 3. 입력이 이미 어절로 나뉘어 있다는 표시

입력이 이미 어절로 나뉜 리스트라는 뜻입니다. 그래야 `word_ids()` 로
"이 토큰이 몇 번째 어절에서 나왔는지" 를 되물을 수 있습니다.
이 기능은 **fast 토크나이저에만** 있습니다.
""".strip()

NER_METRIC = """
## 4. 평가 — 정확도로는 부족하다

분류에서는 정확도를 썼습니다. NER 에서 정확도를 쓰면 **속습니다.**

문장의 대부분 토큰은 `O`(개체 아님)입니다. 전부 `O` 라고 답해도 정확도가
90% 를 넘습니다. 정작 찾아야 할 개체는 하나도 못 찾았는데도요.

그래서 **개체 단위**로 잽니다.

| 지표 | 뜻 |
|---|---|
| 정밀도 | 찾았다고 한 것 중 **진짜**가 얼마나 |
| 재현율 | 진짜 있는 것 중 **찾아낸 것**이 얼마나 |
| F1 | 둘의 조화평균 — 보통 이걸 본다 |

`seqeval` 이 이 계산을 해줍니다. **부분 일치를 인정하지 않는 것**이 핵심입니다 —
`대한민국 서울` 을 `서울` 만 찾았다면 맞힌 것이 아닙니다.

계산 전에 `-100` 자리를 걷어내는 것도 잊지 마세요. 앞에서 넣은 그 값입니다.
""".strip()

NER_TRAIN = """
## 6. 학습

분류 실습과 거의 같습니다. 다른 점만 봅니다.

| | 분류 | NER |
|---|---|---|
| 모델 | `ForSequenceClassification` | `ForTokenClassification` |
| 콜레이터 | `WithPadding` | `ForTokenClassification` |
| 에폭 | 0.1 | **1** |
| 평가셋 | `test` | **`validation`** |

**콜레이터가 다른 이유**가 있습니다. NER 은 라벨도 토큰 수만큼 있어서,
패딩할 때 **라벨 쪽도 함께** 늘려야 합니다. 채우는 값도 `-100` 입니다.
`DataCollatorWithPadding` 은 입력만 맞추므로 여기서는 못 씁니다.

**KLUE 에는 `test` 스플릿이 없습니다.** 정답이 공개되지 않은 벤치마크라
`validation` 으로 평가합니다.

에폭이 1인 것은 데이터가 작기 때문입니다(약 2만 1천 문장). 분류의 15만 건과 다릅니다.
""".strip()

NER_INFER = """
## 7. 써보기

`aggregation_strategy="simple"` 이 눈에 띌 겁니다.

앞에서 **어절을 토큰으로 쪼개며** 라벨을 늘렸던 것을 기억하세요.
추론 결과도 그대로 나오면 `홍`/`##길`/`##동` 이 각각 따로 나옵니다.
이 옵션이 그것을 **다시 하나로 합쳐** `홍길동 → PS` 로 돌려줍니다.

전처리에서 쪼갠 것을 후처리에서 되돌리는 셈입니다. 대칭입니다.

문장을 바꿔 넣어보세요. **처음 보는 이름**이나 **띄어쓰기가 틀린 문장**에서
어떻게 반응하는지 보면 한계가 드러납니다.
""".strip()

NER_OUTRO = """
## 마무리

같은 인코더로 **토큰마다 답을 내는** 모델을 만들었습니다.

- **BIO 태그**로 여러 단어에 걸친 개체를 표현합니다
- 어절 라벨을 토큰에 **다시 맞춰야** 합니다. 여기가 가장 까다롭습니다
- `-100` 은 "손실에서 빼라" 는 **약속된 값**입니다
- NER 에 정확도를 쓰면 속습니다. **개체 단위 F1** 을 봅니다
- 전처리에서 쪼갠 것을 추론에서 **다시 합칩니다**

여기까지가 **인코더**로 하는 일입니다. 문장을 이해해서 라벨을 붙이는 것이지,
새로운 글을 쓰지는 못합니다.

내일은 **글을 만들어내는 쪽** — 디코더로 갑니다. 학습에 쓸 데이터를 정제하는 것부터
시작해, GPT 를 밑바닥부터 만들며 왜 이 구조가 생성에 쓰이는지 직접 보게 됩니다.
""".strip()


# 원본 HF 튜토리얼(ultrachat/Shopify)의 잔재. 현재 데이터와 아무 상관이 없다.
SHOPIFY = ("In this case, it looks like the instructions are about enabling "
           "certain features in Shopify. Interesting!")

# ── SFT 실습 설명 ────────────────────────────────────────────────────
SFT_INTRO = """
# 지시를 따르게 만들기 (SFT)

사전학습된 모델은 **다음 단어를 이어 쓸 줄만** 압니다. "요약해줘" 라고 하면
요약하는 게 아니라 그 문장 뒤에 그럴듯한 말을 잇습니다.

**지시를 따르게** 만드는 것이 Supervised Fine-Tuning 입니다.
방법은 단순합니다 — **"이렇게 물으면 이렇게 답한다"** 는 예시를 잔뜩 보여줍니다.

## 무엇을 하게 되나

1. **어제 만든 데이터**를 불러옵니다 (2일차 Amazon 실습 산출물)
2. 데이터를 **대화 형식**으로 바꿉니다
3. **LoRA** 로 일부만 학습시킵니다
4. 학습한 모델에 실제로 물어봅니다

## 어제와 오늘이 이어집니다

2일차에 상품 설명에서 요약을 뽑아 `instruction`/`output` 쌍을 만들었습니다.
그게 오늘의 학습 데이터입니다. **남의 데이터가 아니라 직접 만든 것**으로 학습합니다.

그리고 어제 프롬프트만으로 점수를 얼마나 올렸는지 기억해 두세요.
오늘은 같은 일을 **학습으로** 합니다. 마지막에 그 둘을 비교하게 됩니다.

## 환경 세팅
""".strip()

SFT_DATA = """
## 1. 데이터 — 어제 만든 것을 불러온다

두 파일을 합쳐 읽습니다.

| 파일 | 무엇 |
|---|---|
| `assets/amazon_ko_sft.jsonl.gz` | 강사가 미리 만든 것 (상품 100개, 약 800건) |
| `data/amazon_ko_sft.mine.jsonl` | **여러분이** 2일차에 만든 것 (상품 10개) |

두 파일에는 **같은 상품이 들어 있습니다** — 배포본이 상품 100개, 여러분 것이 그중 앞 10개.
그래서 완전히 같은 쌍은 하나만 남깁니다.

다만 실제로 걸리는 것은 거의 없습니다. **생성이 비결정적**이라 같은 상품이라도
문장이 조금씩 다르게 나오기 때문입니다. 그건 오히려 도움이 됩니다 —
같은 입력에 대한 여러 표현을 보는 셈이니까요.

파일이 없으면 공개 데이터셋(KULLM)으로 대체하되 **크게 알립니다.**
폴백인 줄 모르고 "내 데이터로 학습했다" 고 오해하면 안 되니까요.

> 경로를 계산하는 코드가 붙어 있는 이유가 있습니다. Jupyter 는 노트북이 있는 폴더를
> 작업 디렉터리로 잡습니다. 2일차 노트북이 쓴 파일을 3일차 노트북이 읽으려면
> **양쪽이 같은 곳을 봐야** 합니다.
""".strip()

SFT_MESSAGES = """
## 2. 대화 형식으로 바꾸기

`instruction` / `output` 쌍을 **주고받는 대화**로 바꿉니다.

```
{"instruction": "요약해줘...", "output": "..."}
        ↓
[{"role": "user",      "content": "요약해줘..."},
 {"role": "assistant", "content": "..."}]
```

왜 이렇게 하느냐면, 실제로 쓸 때도 이 형태로 물어보기 때문입니다.
**학습할 때와 쓸 때의 형식이 같아야** 모델이 헷갈리지 않습니다.

이 원칙은 앞으로 계속 나옵니다. 형식이 어긋나면 학습은 정상적으로 끝나는데
막상 써보면 이상한 답이 나옵니다. **조용히 실패하는** 종류입니다.
""".strip()

SFT_TEMPLATE = """
## 3. 챗 템플릿 — Base 모델에는 대화 형식이 없다

모델 이름 끝의 **`-Base`** 를 보세요. 사전학습만 하고 대화 학습은 안 한 모델입니다.
그래서 `<|user|>` 같은 **대화 표시를 모릅니다.**

우리가 직접 정해서 넣어줍니다. 그것이 `DEFAULT_CHAT_TEMPLATE` 입니다.
Jinja 문법이라 한 줄로 붙어 있어 읽기 어렵지만, 하는 일은 이겁니다.

```
[{"role":"user","content":"안녕"},
 {"role":"assistant","content":"반가워"}]
        ↓  템플릿을 거치면
<|user|>
안녕<|endoftext|>
<|assistant|>
반가워<|endoftext|>
```

**역할을 표시하는 특수 문자열로 감싸는 것**이 전부입니다.
모델은 이 표시를 보고 "여기서부터 내가 답할 차례" 를 배웁니다.

### 나머지 두 줄

| 코드 | 왜 |
|---|---|
| `pad_token_id = eos_token_id` | Base 모델엔 패딩 토큰이 없다. 문장 끝 토큰을 대신 쓰는 관례 |
| `model_max_length > 100_000` 이면 2048 | Qwen3 의 기본 컨텍스트가 지나치게 길다. 실습용으로 줄인다 |

> **`-Instruct` 모델은 이미 템플릿을 갖고 있습니다.** 그때는 이 셀이 필요 없습니다.
> 오히려 덮어쓰면 모델이 학습한 것과 어긋나 성능이 떨어집니다.
""".strip()

SFT_LORA = """
## 4. LoRA — 전부 학습시키지 않는다

0.6B 모델도 전부 학습시키려면 메모리가 만만치 않습니다. 그리고 대부분의 경우
**그럴 필요도 없습니다.**

**LoRA** 는 원래 가중치를 얼려두고, 옆에 **작은 행렬 두 개**를 붙여 그것만 학습합니다.
학습이 끝나면 그 작은 것만 저장하면 됩니다 — 수백 MB 가 아니라 수십 MB 입니다.

```
원래 가중치 W  (얼림, 학습 안 함)
      +
   A × B      (작다. 이것만 학습)
```

### 값 읽는 법

| 값 | 뜻 |
|---|---|
| `r=64` | 붙이는 행렬의 **크기**. 클수록 표현력이 늘고 무거워진다 |
| `lora_alpha=16` | 그 결과를 얼마나 **세게 반영**할지 |
| `target_modules` | 어디에 붙일지. 여기서는 **어텐션 4곳**(q·k·v·o) |
| `lora_dropout=0.1` | 과적합 억제 |

`print_trainable_parameters()` 를 꼭 보세요. **전체의 몇 %만 학습하는지** 나옵니다.
그 숫자가 LoRA 를 쓰는 이유를 한눈에 보여줍니다.

> 어텐션에만 붙이고 피드포워드(`gate/up/down_proj`)는 뺐습니다. 관례적인 선택이고,
> 붙이면 표현력은 늘지만 무거워집니다. 정답이 있는 값은 아닙니다.
""".strip()

SFT_TRAINER = """
## 5. 학습 설정

1일차 분류 실습과 값이 꽤 다릅니다. 이유가 있습니다.

| | 분류 (1일차) | SFT |
|---|---|---|
| 학습률 | `2e-5` | **`5e-4`** — 25배 |
| 배치 | 16 | 2 × 누적 8 = **유효 16** |

**학습률이 큰 것은 LoRA 때문입니다.** 원래 가중치는 얼려 뒀고 새로 붙인 작은 행렬만
바닥부터 배우므로, 조심스럽게 갈 이유가 없습니다.

**배치를 쪼개는 것은 메모리 때문입니다.** 한 번에 16개를 올리면 GPU 메모리가 넘칩니다.
2개씩 8번 계산해서 **기울기를 모았다가 한 번에 갱신**합니다. 결과는 배치 16과 같고
메모리만 1/8 입니다. `gradient_accumulation_steps` 가 그것입니다.

`gradient_checkpointing` 도 같은 목적입니다 — 중간 계산 결과를 저장하지 않고
필요할 때 다시 계산합니다. 메모리를 아끼고 시간을 씁니다.

> `use_reentrant: False` 는 PEFT 와 gradient checkpointing 을 함께 쓸 때 필요한
> 관용구입니다. 빼면 경고가 나거나 기울기가 흐르지 않습니다.
""".strip()

SFT_OUTRO = """
## 마무리

지시를 따르는 모델을 만들었습니다.

- 사전학습 모델은 **이어 쓸 줄만** 압니다. 지시를 따르게 하려면 예시를 보여줘야 합니다
- **Base 모델에는 대화 형식이 없습니다.** 우리가 정해서 넣어줍니다
- **학습할 때와 쓸 때의 형식이 같아야** 합니다. 어긋나면 조용히 실패합니다
- **LoRA** 로 전체의 일부만 학습해도 충분합니다
- 학습 전에 **토큰 길이를 확인**합니다. 잘리면 정답이 날아갑니다

### 여기서 끝이 아닙니다

SFT 는 **"이렇게 답해라"** 를 가르칩니다. 그런데 정답이 하나로 정해지지 않는
작업에서는 이것만으로 부족합니다. 요약은 여러 가지가 다 맞을 수 있으니까요.

그럴 때 쓰는 것이 **선호 학습**입니다 — "이 답이 저 답보다 낫다" 를 가르칩니다.
다음 실습(DPO)에서 **방금 학습한 이 모델을 이어받아** 그것을 해봅니다.
""".strip()

SFT_OBSERVE = """
### 형식이 제대로 만들어졌는지 봅니다

`role` 과 `content` 가 짝을 이루는지, `user` 다음에 `assistant` 가 오는지 확인합니다.

여기가 어긋나면 뒤에서 전부 어긋납니다. 그런데 **에러가 나지 않아서** 학습이
끝날 때까지 모릅니다. 그래서 눈으로 한 번 봅니다.
""".strip()

SFT_APPLY = """
## 챗 템플릿 적용과 길이 확인

위에서 정한 형식을 실제로 씌웁니다. `messages` 가 한 덩어리 문자열(`text`)이 됩니다.

`tokenize=False` 인 것에 주의하세요. 여기서는 **문자열까지만** 만듭니다.
토큰으로 바꾸는 것은 학습할 때 TRL 이 알아서 합니다.

빈 `system` 메시지를 앞에 끼워 넣는 것도 보입니다. 템플릿이 세 역할을 다 기대하는데
데이터에는 `system` 이 없어서, **형식을 맞추려고** 넣습니다.

그리고 **토큰 길이를 잽니다.** 아래 셀의 주석을 꼭 읽어보세요 —
이 확인을 건너뛰면 어떻게 조용히 실패하는지 적혀 있습니다.
""".strip()

SFT_RUN = """
## 학습

돌려놓고 기다립니다. 몇 분 걸립니다.

`loss` 가 내려가는지 보세요. 내려가지 않으면 학습률이나 데이터 형식을 의심합니다.
**너무 빨리 0 에 가까워지는 것도 좋은 신호가 아닙니다** — 데이터가 너무 적거나
같은 것을 반복해서 외우는 중일 수 있습니다.
""".strip()

SFT_GEN = """
## 써보기

학습한 어댑터를 다시 불러와 실제로 물어봅니다.

**챗 템플릿을 다시 넣는 것**에 주의하세요. 템플릿은 우리가 코드로 지정한 것이라
모델 파일에 자동으로 따라가지 않습니다. 불러온 뒤 다시 지정해야
**학습할 때와 같은 형식**으로 물어볼 수 있습니다.

앞에서 말한 "학습할 때와 쓸 때의 형식이 같아야 한다" 가 여기서 지켜집니다.

질문을 바꿔 넣어보세요. 학습 데이터가 **상품 요약**이었으니 그쪽은 잘하고,
전혀 다른 질문에는 어색할 겁니다. 그것이 파인튜닝의 성질입니다.
""".strip()


# ── DPO 실습 설명 ────────────────────────────────────────────────────
DPO_INTRO = """
# 더 나은 답을 고르게 만들기 (DPO)

SFT 는 **"이렇게 답해라"** 를 가르쳤습니다. 정답이 하나로 정해지는 일에는 충분합니다.

그런데 요약이나 설명처럼 **정답이 여럿인** 작업에서는 부족합니다.
문법도 맞고 내용도 맞는 답 두 개 중에 **어느 쪽이 더 나은지** 는 SFT 로 못 가르칩니다.

그래서 방식을 바꿉니다. 정답 하나를 보여주는 대신 **"이 답이 저 답보다 낫다"** 를
보여줍니다. 이것이 **선호 학습**(preference learning)이고,
그중 가장 단순한 방법이 **DPO**(Direct Preference Optimization)입니다.

## 무엇을 하게 되나

1. **선호 쌍** 데이터를 봅니다 — 같은 질문에 좋은 답과 나쁜 답
2. **앞 실습에서 학습한 모델을 이어받습니다**
3. DPO 로 선호를 학습시킵니다
4. 끝에서 **ORPO** 와 비교하고, 언제 무엇을 쓸지 정리합니다

> **DPO 는 SFT 를 전제합니다.** 말은 할 줄 아는 모델을 취향 쪽으로 미는 방법이지,
> 아무것도 모르는 모델을 가르치는 방법이 아닙니다. 그래서 앞 실습의 결과를 이어받습니다.

## 환경 세팅
""".strip()

DPO_DATA = """
## 1. 데이터 — 선호 쌍

SFT 데이터는 `instruction` / `output` 두 열이었습니다. DPO 는 **세 열**입니다.

| 열 | 내용 |
|---|---|
| `prompt` | 질문 |
| `chosen` | **더 나은** 답 |
| `rejected` | **덜 나은** 답 |

같은 질문에 답이 둘 있고, **어느 쪽이 나은지 표시**돼 있습니다.
모델은 `chosen` 쪽 확률을 올리고 `rejected` 쪽을 내리는 방향으로 학습합니다.

여기서 쓰는 `ko-dpo-mix-7k-trl-style` 은 한국어 선호 데이터를 모은 것입니다.
이름의 **`trl-style`** 은 TRL 라이브러리가 기대하는 세 열 구조로 정리돼 있다는 뜻입니다.

> **이 데이터를 어떻게 만드나** 가 실무의 진짜 문제입니다. 사람이 일일이 고르면
> 비싸고, 큰 모델에게 시키면 그 모델의 편향이 딸려옵니다.
> 2일차에 다룬 **LLM-as-judge** 가 여기에 쓰입니다.
""".strip()

DPO_SPLIT = """
### 얼마나 쓸 것인가

데이터가 7천 건이지만 실습에서는 일부만 씁니다. 전부 돌리면 시간이 오래 걸립니다.

건수를 고정해서 자르지 않고 **비율로** 나눕니다. 데이터 크기가 바뀌면
고정 인덱스는 조용히 깨지기 때문입니다 — 실제로 다른 노트북에서 평가셋이
0건이 된 적이 있습니다.
""".strip()

DPO_FORMAT = """
## 3. 형식 맞추기 — 여기가 SFT 와 다릅니다

SFT 에서는 `apply_chat_template` 이 알아서 형식을 씌웠습니다.
여기서는 **손으로 조립**합니다. 이유가 있습니다.

DPO 는 `prompt` / `chosen` / `rejected` 를 **각각 따로** 받습니다.
하나로 합쳐진 대화가 아니라 세 조각이 필요하므로, 템플릿을 통째로 씌울 수 없습니다.

```
prompt   = <|system|>…<|user|>…<|assistant|>     ← 여기까지가 질문
chosen   = 좋은 답<|endoftext|>
rejected = 나쁜 답<|endoftext|>
```

**주의할 것이 하나 있습니다.** 손으로 조립하다 보니 위 토크나이저 셀에서 정한
템플릿과 **어긋날 수 있습니다.** 어긋나면 학습은 정상적으로 끝나는데
막상 써보면 이상한 답이 나옵니다. 두 곳의 `<|user|>` 표기가 같은지 확인해 보세요.
""".strip()

DPO_MODEL = """
## 4. 모델 — 앞 실습을 이어받는다

DPO 는 **SFT 된 모델에서 출발**해야 합니다. 아래 셀은 앞 실습(`HPC_SFT실습`)이
저장한 어댑터를 찾아 이어받습니다. 없으면 베이스에서 시작하되 **크게 알립니다.**

### 참조 모델은 어디 있나

DPO 의 원래 정의에는 **참조 모델**이 필요합니다. 학습 중인 모델이 원래 모델에서
너무 멀어지지 않게 붙잡아 두는 역할입니다. 그래서 보통 메모리에 모델이 두 개 뜹니다.

그런데 아래 코드에는 `ref_model` 인자가 **없습니다.** 빠뜨린 게 아닙니다.

**LoRA 를 쓰면 참조 모델이 따로 필요 없습니다.** 원래 가중치는 얼려 둔 채
어댑터만 학습하므로, **어댑터를 잠깐 끄면 그게 곧 원래 모델**입니다.
TRL 이 내부에서 그렇게 처리합니다. 모델 하나 분량의 메모리를 아끼는 셈입니다.

> 이 노트북 끝의 ORPO 섹션에서 "DPO 는 참조 모델이 필요하다" 고 말합니다.
> **개념상 필요한 것**과 **LoRA 덕에 따로 안 띄워도 되는 것**은 다른 얘기입니다.
""".strip()

DPO_TRAINER = """
## 5. 학습 설정

SFT 와 값이 조금 다릅니다.

| | SFT | DPO |
|---|---|---|
| 학습률 | `5e-4` | **`3e-4`** — 조금 낮게 |
| 에폭 | 3 | **2** |

**학습률을 낮추는 이유**가 있습니다. DPO 는 이미 SFT 된 모델을 다듬는 단계라
크게 흔들면 앞에서 배운 것을 잃습니다. 2일차 CPT 에서 본 **치명적 망각**과 같은 이야기입니다.

`beta` 는 명시하지 않아 기본값(0.1)이 쓰입니다. 이 값이 클수록 참조 모델에서
**멀어지지 않으려는 힘**이 세집니다. 작게 주면 선호를 세게 반영하지만
원래 능력을 잃을 위험이 커집니다.
""".strip()

DPO_RUN = """
## 학습

`rewards/chosen` 과 `rewards/rejected` 가 로그에 나옵니다. **둘의 차이가 벌어지는지**
보세요. 그것이 "좋은 답과 나쁜 답을 구분하게 되고 있다" 는 신호입니다.

`rewards/accuracies` 는 chosen 을 rejected 보다 높게 준 비율입니다. 올라가야 정상입니다.
""".strip()

DPO_GEN = """
## 써보기

SFT 때와 같은 방식으로 물어봅니다. 챗 템플릿을 다시 넣는 것도 같습니다.

**SFT 결과와 비교해 보세요.** 선호 학습이 붙으면 보통 답이 더 정돈되고 길어집니다.
다만 실습 규모(500건 · 2에폭)에서는 차이가 크지 않을 수 있습니다.
""".strip()

# DPO 가 SFT 를 이어받게 한다. 실제 개행이 든 상수로 둔다 (백슬래시 회피).
DPO_RESUME_OLD = """model = get_peft_model(model, lora_config)"""

DPO_RESUME_NEW = """# ★ DPO 는 SFT 를 전제한다. 앞 실습이 남긴 어댑터가 있으면 이어받는다.
#   없으면 베이스에서 시작하되 조용히 넘어가지 않고 알린다 —
#   이 노트북 끝의 ORPO 섹션이 "베이스에 바로 걸면 잘 안 된다" 고 말하기 때문이다.
from pathlib import Path

from peft import PeftModel

_sft = Path("data/sft_model")
if (_sft / "adapter_config.json").exists():
    print(f"SFT 결과를 이어받습니다 — {_sft}")
    # is_trainable=True 가 없으면 어댑터가 얼어붙어 학습이 되지 않는다.
    model = PeftModel.from_pretrained(model, str(_sft), is_trainable=True)
else:
    print("=" * 60)
    print("★ SFT 산출물이 없어 베이스 모델에서 시작합니다.")
    print("  앞의 HPC_SFT실습 을 먼저 돌리면 이어받을 수 있습니다.")
    print("  베이스에 바로 DPO 를 거는 것은 권장되지 않습니다 —")
    print("  이 노트북 끝의 ORPO 섹션에서 그 이유를 다룹니다.")
    print("=" * 60)
    model = get_peft_model(model, lora_config)"""

DPO_SPLIT_OLD = """# remove this when done debugging
indices = range(0,500)
test_indices = range(500,550)"""

DPO_SPLIT_NEW = """# 건수를 고정해서 자르면 데이터 크기가 바뀔 때 조용히 깨진다. 비율로 나눈다.
n_all = len(raw_datasets["train"])
n_train = min(500, int(n_all * 0.9))
n_test = min(50, n_all - n_train)

indices = range(0, n_train)
test_indices = range(n_train, n_train + n_test)
print(f"전체 {n_all:,}건 → 학습 {n_train:,} / 평가 {n_test}")"""


# ── GRPO 실습 설명 ───────────────────────────────────────────────────
GRPO_INTRO = """
# 채점 함수로 학습시키기 (GRPO)

지금까지 두 가지를 봤습니다.

| | 필요한 것 |
|---|---|
| SFT | 입력–**정답** 쌍 |
| DPO | 좋은 답 / 나쁜 답 **쌍** |

둘 다 **사람이 만든 데이터**가 있어야 합니다. 그런데 어떤 일은 정답을
**코드로 확인**할 수 있습니다. 수학 문제가 그렇습니다 — 답이 맞는지 계산하면 됩니다.

그럴 때는 데이터를 모으는 대신 **채점 함수**를 짜면 됩니다.
모델이 답을 여러 개 만들면, 함수가 점수를 매기고, 높은 쪽으로 밀어냅니다.

이것이 **GRPO**(Group Relative Policy Optimization)입니다.

## 무엇을 하게 되나

1. 한국어 **수학 문제** 데이터를 봅니다
2. 모델에게 **생각하는 형식**을 정해줍니다
3. **채점 함수 두 개**를 만듭니다 — 형식과 정답
4. 학습시키고, 실제로 그 형식으로 답하는지 봅니다

> **GRPO 는 가장 비쌉니다.** SFT·DPO 는 정답을 한 번 보고 손실을 계산하지만,
> GRPO 는 **매 스텝마다 답을 여러 개 생성**하고 각각 채점합니다.
> 생성이 학습 안에 들어가 있어서 스텝당 비용이 몇 배입니다.

## 환경 세팅
""".strip()

GRPO_DATA = """
## 1. 데이터 — 답을 검증할 수 있는 것

`numina_math_ko_verifiable_540k` 는 한국어 수학 문제 모음입니다.
이름의 **`verifiable`** 이 핵심입니다 — 답이 맞는지 **기계가 확인할 수 있다**는 뜻입니다.

| 열 | 내용 |
|---|---|
| `problem` | 문제 |
| `answer` | 정답 (수식) |

GRPO 를 쓸 수 있는지 판단하는 기준이 여기 있습니다.
**"이 작업의 정답을 코드로 확인할 수 있는가?"**

| 가능 | 어려움 |
|---|---|
| 수학 — 답이 맞나 | 요약 — 좋은 요약인가 |
| 코드 — 테스트가 통과하나 | 번역 — 자연스러운가 |
| 형식 — 스키마를 지켰나 | 상담 — 공감이 되나 |

오른쪽은 GRPO 대상이 아닙니다. 그쪽은 SFT 나 DPO 로 갑니다.
""".strip()

GRPO_FORMAT = """
## 2. 생각하는 형식을 정해준다

시스템 프롬프트로 **출력 형식**을 못박습니다.

```
<think>
...생각하는 과정...
</think>
<answer>
...최종 답...
</answer>
```

왜 이렇게 나누느냐면, **답만 뽑아내야** 채점할 수 있기 때문입니다.
생각 과정과 답이 뒤섞여 있으면 어디가 답인지 알 수 없습니다.

그리고 생각을 쓰게 하는 것 자체가 정답률을 올립니다.
바로 답하는 것보다 단계를 밟는 편이 낫다는 것은 잘 알려져 있습니다.

**이 형식이 다음 섹션의 채점 함수와 짝입니다.** 프롬프트에서 형식을 요구하고,
채점 함수가 그 형식을 지켰는지 봅니다. 둘 중 하나만 바꾸면 어긋납니다.
""".strip()

GRPO_REWARD = """
## 3. 채점 함수 — 이 실습의 핵심

GRPO 에는 **정답 라벨이 없습니다.** 대신 **답을 채점하는 함수**가 있습니다.

두 개를 씁니다.

| 함수 | 무엇을 보나 | 왜 |
|---|---|---|
| `format_reward` | `<think>…</think><answer>…</answer>` 형식을 지켰나 | 형식이 깨지면 답을 꺼낼 수 없다 |
| `accuracy_reward` | 답이 정답과 **수학적으로 같은가** | 이게 진짜 목표다 |

### 두 번째가 중요합니다

문자열로 비교하면 `1/2` 와 `0.5` 를 다른 답으로 봅니다. 둘 다 맞는데도요.

`math_verify` 는 수식을 **파싱해서** 비교합니다. 표현이 달라도 값이 같으면
맞다고 판정합니다. 그래서 채점이 실제 정답률에 가까워집니다.

**채점 함수의 품질이 곧 학습의 품질입니다.** 함수가 엉성하면 모델은
그 엉성함을 파고듭니다 — 형식만 그럴듯하게 맞추고 답은 틀리는 식으로요.
`format_reward` 하나만 썼다면 정확히 그렇게 됐을 겁니다.

> **이것이 GRPO 와 PPO 의 차이이기도 합니다.** PPO 는 사람의 선호를 배운
> **보상 모델**(또 하나의 신경망)을 씁니다. GRPO 는 그냥 **함수**를 씁니다.
> 채점을 코드로 짤 수 있으면 모델 하나를 통째로 아낍니다.
""".strip()

GRPO_TRAIN = """
## 4. 학습 설정

`num_generations=4` 가 GRPO 의 정의에 해당하는 값입니다.

**문제 하나에 답을 4개 만듭니다.** 그리고 그 4개를 **서로 비교**합니다 —
평균보다 잘한 답은 확률을 올리고, 못한 답은 내립니다.
이름의 **G**roup **R**elative 가 이 뜻입니다. 그룹 안에서 상대적으로 평가합니다.

절대 점수가 아니라 상대 비교라서, 별도의 기준선 모델이 필요 없습니다.
그룹의 평균이 기준선 역할을 합니다.

| 값 | 뜻 |
|---|---|
| `num_generations=4` | **그룹 크기.** 크면 비교가 안정되고 그만큼 느려진다 |
| `max_completion_length=128` | 답 길이 상한. 생성이 학습 안에 있어 시간에 직결된다 |
| `gradient_accumulation_steps=16` | 메모리를 아끼려고 기울기를 모았다 갱신 |
| LoRA `r=8` | SFT 의 64 보다 작다. 생성 부담이 커서 가볍게 간다 |

> 학습 로그의 `reward` 가 올라가는지 보세요. 두 채점 함수의 합입니다.
> `format_reward` 가 먼저 1.0 에 가까워지고, `accuracy_reward` 가 천천히 따라옵니다 —
> 형식을 지키는 것이 답을 맞히는 것보다 쉽기 때문입니다.
""".strip()

GRPO_GEN = """
## 5. 써보기 — 템플릿이 학습 때와 다릅니다

아래 셀에서 챗 템플릿을 다시 넣는데, **학습 때와 한 군데가 다릅니다.**

```
학습:  ... <|assistant|>
생성:  ... <|assistant|>
       <think>              ← 이 줄이 추가됨
```

모델이 답을 시작하는 자리에 **`<think>` 를 미리 넣어줍니다.**
그러면 모델은 생각부터 이어 쓸 수밖에 없습니다. 형식을 강제하는 셈입니다.

이런 기법을 **프리필**(prefill)이라고 합니다. 학습으로 형식을 가르치되,
생성할 때 한 번 더 못을 박는 것입니다.

출력에 `<think>` 와 `<answer>` 가 제대로 나오는지 보세요.
나오지 않으면 학습이 부족한 것이고, 실습 규모에서는 흔히 그렇습니다.
""".strip()

GRPO_OUTRO = """
## 마무리

채점 함수로 학습시키는 방법을 봤습니다.

- GRPO 는 정답 라벨이 아니라 **채점 함수**를 씁니다
- 그래서 **답을 코드로 확인할 수 있는 일**에만 쓸 수 있습니다
- 문제 하나에 답을 여러 개 만들어 **서로 비교**합니다 (Group Relative)
- **채점 함수의 품질이 곧 학습의 품질**입니다. 엉성하면 모델이 파고듭니다
- 생성이 학습 안에 있어 **가장 비쌉니다**

### 세 가지를 다 봤습니다

| 방법 | 필요한 것 | 비용 |
|---|---|---|
| SFT | 입력–정답 쌍 | 보통 |
| DPO / ORPO | 좋은 답 / 나쁜 답 쌍 | 보통 |
| GRPO | **채점 함수** | 가장 비쌈 |

무엇을 고를지는 **가진 데이터가 정합니다.** 방법을 먼저 정하고 데이터를
맞추는 것이 아닙니다. 그리고 그 위에 **학습하지 않는 선택지**가 있습니다 —
2일차에 본 프롬프트 최적화입니다.

**프롬프트로 되면 학습하지 않습니다.** 그것이 이 과정 전체의 결론입니다.

이제 마지막 실습이 남았습니다 — 오늘 아침 만든 RAG 시스템에 방금까지 학습한
모델을 꽂아, **포스트트레이닝이 시스템에서 무엇을 바꾸는지** 눈으로 확인합니다.
""".strip()


# ── BM25 RAG 실습 설명 ───────────────────────────────────────────────
RAG_INTRO = """
# 찾아서 답하기 (RAG)

3일차는 **시스템을 하나 만드는 것**으로 시작합니다.

모델을 쓸모 있게 만드는 길이 학습만 있는 것은 아닙니다.
문제가 **"모델이 모른다"** 라면 학습이 답이 아닐 때가 많습니다.

- 사내 문서는 모델이 본 적이 없습니다
- 어제 나온 소식은 학습 시점 이후입니다
- 내용이 바뀔 때마다 다시 학습시킬 수는 없습니다

이럴 때는 **찾아서 넣어줍니다.** 질문이 오면 관련 문서를 검색해 프롬프트에 붙이고,
모델은 그것을 보고 답합니다. **RAG**(Retrieval-Augmented Generation)입니다.

## 무엇을 하게 되나

1. 한국어 위키를 **잘라서 색인**합니다
2. **BM25** 로 검색합니다 — 임베딩도 벡터DB도 쓰지 않습니다
3. 검색 결과로 답을 만듭니다
4. **프롬프트를 한국어로** 고칩니다 (기본값이 영어라 생기는 문제)
5. 복합 질문을 **쪼개서** 검색해 봅니다

> **학습하지 않습니다.** 가중치는 그대로 두고 입력만 바꿉니다.
> 그래서 문서가 바뀌면 색인만 다시 만들면 됩니다.

## 오늘 하루가 이 시스템을 축으로 돕니다

```
지금:      RAG 시스템을 만든다 (모델은 기성품 4B)
이후:      평가 수단을 갖추고 → 프롬프트로, 그다음 학습으로 모델을 개선한다
하루의 끝:  학습한 모델을 이 시스템에 다시 꽂아 무엇이 달라졌는지 본다
```

그러니 이 실습에서 저장하는 **검색 인덱스(`./bm25_retriever`)를 지우지 마세요.**
마지막 실습이 그걸 다시 씁니다.
""".strip()

RAG_SETUP = """
## 1. 환경 세팅

vLLM 서버에 HTTP 로 붙습니다. 3일차의 첫 실습이므로 서버를 여기서 띄웁니다 —
이어지는 평가·퓨샷 실습도 같은 서버를 계속 씁니다.

```bash
source /opt/vllm-env/bin/activate
nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000     --gpu-memory-utilization 0.80 --max-model-len 16384 > /tmp/vllm.log 2>&1 &
```

`Settings.embed_model = None` 이 눈에 띌 겁니다. **일부러 끄는 것**입니다.

LlamaIndex 는 기본으로 OpenAI 임베딩을 쓰려 하고, 그러면 API 키를 요구합니다.
그런데 **BM25 는 임베딩이 필요 없습니다.** 다음 절에서 이유를 봅니다.
""".strip()

RAG_DATA = """
## 2. 문서 준비 — 자르는 것이 중요하다

한국어 위키 1,000건을 가져와 **조각(chunk)** 으로 자릅니다.

왜 자르느냐면, 문서 하나가 너무 길면 **검색이 뭉뚝해지기** 때문입니다.
위키 문서 하나에는 생애·업적·평가가 다 들어 있습니다. 통째로 색인하면
어느 질문에나 어중간하게 걸립니다.

반대로 너무 잘게 자르면 **문맥이 끊깁니다.** "그는" 이 누구인지 모르게 됩니다.

`chunk_size=512` 는 그 사이의 타협입니다. **정답이 있는 값이 아닙니다** —
문서 성격에 따라 조정합니다. RAG 품질이 안 나올 때 가장 먼저 손보는 곳이기도 합니다.
""".strip()

RAG_BM25 = """
## 3. BM25 검색기 — 임베딩 없이 찾는다

**BM25 는 단어가 겹치는 정도**로 문서를 고릅니다. 의미를 이해하지 않습니다.

| | BM25 (어휘 검색) | 임베딩 (의미 검색) |
|---|---|---|
| 기준 | **단어 일치** | 벡터 거리 |
| 준비물 | 없음 | 임베딩 모델 + 벡터DB |
| 강한 곳 | 고유명사·전문용어·숫자 | 바꿔 말한 질문 |
| 약한 곳 | "영화 만든 사람" ↔ "감독" | 드문 고유명사 |

낡은 방법처럼 들리지만 **실무에서 아직 많이 씁니다.**
사람 이름이나 제품 코드처럼 **정확히 그 단어**를 찾아야 할 때는 오히려 임베딩보다 낫습니다.
그래서 실제 시스템은 둘을 **같이** 쓰는 경우가 많습니다(하이브리드 검색).

여기서는 BM25 만으로 어디까지 되는지 봅니다.

아래에서 두 가지 질문을 던져봅니다. **하나는 잘 되고 하나는 덜 됩니다.**
어느 쪽이 왜 그런지 생각해 보세요 — 위 표에 답이 있습니다.
""".strip()

RAG_ENGINE = """
## 4. 답을 만드는 단계 — `refine` 모드

검색까지 했으니 이제 그 문서들로 **답을 씁니다.**

`response_mode="refine"` 을 씁니다. 이름 그대로 **고쳐 쓰는** 방식입니다.

```
문서1 → 초안 작성
문서2 → 초안을 보면서 고쳐 씀
문서3 → 또 고쳐 씀
```

문서를 한꺼번에 넣지 않고 **하나씩 순차로** 넣습니다.
그래서 프롬프트가 두 개입니다 — 처음 쓸 때(`text_qa`)와 고칠 때(`refine`).
`refine` 프롬프트에 `existing_answer` 라는 자리가 있는 이유가 이것입니다.

> 문서가 많아도 컨텍스트를 넘기지 않는다는 장점이 있고, LLM 을 문서 수만큼
> 호출한다는 단점이 있습니다. 문서가 적으면 `compact` 가 더 빠릅니다.

### 그런데 답이 영어로 나옵니다

LlamaIndex 의 기본 프롬프트는 **영어**입니다. 한국어로 물어도 영어 지시를 받은
모델이 영어로 답하기 쉽습니다. 그래서 다음 셀들에서 프롬프트를 한국어로 바꿉니다.

**프롬프트를 바꾸는 방법에 함정이 하나 있습니다.** 코드 주석을 꼭 읽어보세요 —
겉보기에 바뀐 것 같은데 실제로는 안 바뀌는 종류의 함정입니다.
""".strip()

RAG_QUERY = """
## 5. 실제로 물어보기

이제 검색과 생성을 붙여서 답을 받습니다.

**답만 보지 말고 `source_nodes` 를 같이 보세요.** 어떤 문서를 근거로 답했는지 나옵니다.

RAG 에서 이게 중요합니다. 모델이 그럴듯하게 답해도 **근거 문서가 엉뚱하면**
그 답은 지어낸 것입니다. 검색이 실패했는지 생성이 실패했는지 구분하려면
근거를 봐야 합니다.

| 증상 | 어디가 문제인가 |
|---|---|
| 근거 문서가 질문과 무관 | **검색** — 청킹이나 질의를 손본다 |
| 근거는 맞는데 답이 틀림 | **생성** — 프롬프트나 모델을 손본다 |
| 근거가 아예 없음 | 색인에 그 내용이 없다 |

실무에서 RAG 를 고칠 때 이 구분부터 합니다.
""".strip()

RAG_SUBQ = """
## 6. 복합 질문 쪼개기 — SubQuestionQueryEngine

한 번의 검색으로 안 되는 질문이 있습니다.

```
"백남준과 맥스웰은 각각 어떤 분야의 인물인가요?"
```

이걸 통째로 검색하면 두 인물이 **같이 나오는** 문서를 찾게 됩니다. 그런 문서는 없습니다.

**질문을 쪼개면** 됩니다.

```
원 질문 → LLM 이 하위 질문으로 분해
            ├ "백남준은 어떤 분야의 인물인가?"  → 검색 → 답
            └ "맥스웰은 어떤 분야의 인물인가?"  → 검색 → 답
                                                    ↓
                                              합쳐서 최종 답
```

하위 질문을 **LLM 이 만들어냅니다.** 그래서 여기에도 프롬프트가 있고,
그것도 기본이 영어입니다.

> **이게 실제로 문제가 됐습니다.** 영어로 하위 질문이 생성되면 한국어 위키를
> BM25 로 검색해서 아무것도 못 찾습니다. 단어가 안 겹치니까요.
> 아래 셀의 주석에 그 내용이 있습니다.

마지막 두 셀에서 **단일 개체 질문**도 넣어봅니다.
쪼갤 필요가 없는 질문에 이 엔진을 쓰면 어떻게 되는지 — LLM 호출만 늘고
결과는 나아지지 않을 수 있습니다. **항상 좋은 도구는 없습니다.**
""".strip()

RAG_OUTRO = """
## 마무리

학습하지 않고 **찾아서 답하는** 방법을 봤습니다.

- 문제가 "모델이 **모른다**" 일 때는 학습보다 **검색**이 맞습니다
- **청킹**이 검색 품질을 좌우합니다. 너무 크면 뭉뚝하고 너무 작으면 문맥이 끊깁니다
- **BM25 는 단어 일치**로 찾습니다. 고유명사에 강하고 바꿔 말한 질문에 약합니다
- **근거 문서를 보세요.** 검색 실패와 생성 실패를 구분하는 유일한 방법입니다
- 라이브러리 **기본 프롬프트는 영어**입니다. 한국어로 쓰려면 바꿔야 합니다

### 언제 RAG 이고 언제 학습인가

| 문제 | 방법 |
|---|---|
| 모델이 **모른다** (사내 문서, 최신 정보) | **RAG** |
| 모델이 알긴 아는데 **원하는 대로 안 한다** | SFT · DPO |
| 도메인 **언어 자체**가 다르다 | CPT |

자주 하는 실수가 **"우리 회사 문서를 학습시키자"** 입니다.
대개는 RAG 가 맞습니다. **파인튜닝은 지식을 넣는 도구가 아니라
행동을 바꾸는 도구**입니다.

표의 가운데 줄 — "알긴 아는데 원하는 대로 안 한다" — 가 **오늘 이 뒤에서 배우는 것**입니다.
그리고 하루의 끝에, 학습한 모델을 지금 만든 이 시스템에 다시 꽂아 봅니다.
`./bm25_retriever` 를 지우지 말라고 한 이유입니다.
""".strip()


# ── 퓨샷 실습 설명 ───────────────────────────────────────────────────
# 64셀이지만 반복 구조다. 평가 루프(30~35 / 43~46), 스키마 강제(57~61 / 62~63),
# 파라미터 실험(22~27)이 각각 같은 패턴이라 대표 지점에만 설명을 넣는다.
FEW_INTRO = """
# 프롬프트만으로 어디까지 되나

학습을 하지 않고 **말로만** 시켜서 어디까지 되는지 봅니다.

이것을 먼저 하는 데는 이유가 있습니다. 학습은 비쌉니다. 데이터를 모으고, 평가 체계를
만들고, GPU 를 돌리고, 그 다음에도 유지보수가 따라붙습니다.
**프롬프트로 되면 그걸 다 안 해도 됩니다.**

## 무엇을 하게 되나

1. **지시만으로** 감정을 분류해 봅니다 (zero-shot)
2. 출력이 제멋대로인 것을 보고 **형식을 잡아갑니다**
3. **예시를 몇 개 보여주고**(few-shot) 얼마나 나아지는지 잽니다
4. 문장에서 **정보를 뽑아내는** 일에도 같은 방법을 씁니다

## 이 실습의 결론은 숫자입니다

3번에서 **zero-shot 정확도와 few-shot 정확도를 비교**하게 됩니다.
그 두 숫자의 차이가 "예시를 보여주는 것이 값어치가 있는가" 에 대한 답입니다.

> 학습은 하지 않습니다. GPU 는 추론에만 씁니다.

## 환경 준비
""".strip()

FEW_API = """
### 두 가지 호출 방식

바로 아래 두 셀이 거의 같아 보이지만 **API 가 다릅니다.**

| | `completions` | `chat.completions` |
|---|---|---|
| 입력 | 문자열 하나 | **역할이 있는 메시지 목록** |
| 성격 | 이어 쓰기 | 주고받기 |

`chat` 쪽은 `system` / `user` / `assistant` 로 역할을 나눕니다.
`system` 에 "너는 무엇이다" 를 두고 `user` 에 실제 질문을 두는 것이 관례입니다.

요즘 모델은 대부분 `chat` 형식으로 학습돼 있어서, **`chat` 을 쓰는 것이 기본**입니다.
`completions` 는 이어 쓰기가 필요한 특수한 경우에만 씁니다.

이 역할 형식을 문자열로 바꿔 주는 장치가 **챗 템플릿**입니다.
잠시 뒤 SFT 실습에서 직접 다룹니다.
""".strip()

FEW_MESSY = """
### 출력이 제멋대로인 것을 보세요

바로 위 셀에서 10건을 돌렸습니다. 결과가 지저분할 겁니다.

- 어떤 건 `긍정`, 어떤 건 `이 리뷰는 긍정적입니다`
- 어떤 건 이유까지 설명
- 어떤 건 앞에 인사말

**모델이 틀린 게 아닙니다.** 우리가 형식을 말해주지 않았을 뿐입니다.

이 상태로는 프로그램에서 쓸 수 없습니다. 결과를 세려면 `긍정`인지 아닌지
**기계가 판별**할 수 있어야 하니까요.

아래에서 형식을 잡아갑니다. 세 단계로 점점 세게 조입니다.

| 방법 | 강제력 |
|---|---|
| 프롬프트로 **부탁**한다 | 약함 — 대체로 따르지만 보장 없음 |
| `seed` 로 **재현성**을 확보한다 | 흔들림만 줄임 |
| **제약 디코딩**으로 막는다 | **강함 — 다른 답이 나올 수 없음** |
""".strip()

FEW_PARAM = """
### 여기부터는 같은 입력에 파라미터만 바꿉니다

아래 몇 셀은 **같은 질문을 반복**합니다. 바뀌는 것은 파라미터뿐입니다.

| 파라미터 | 하는 일 |
|---|---|
| `seed` | 난수를 고정한다. 같은 seed 면 같은 답 |
| `n` | 한 번에 여러 개를 뽑는다. **얼마나 흔들리는지** 보는 용도 |
| `temperature` | 낮으면 보수적, 높으면 과감 |
| `top_p` | 확률 상위 몇 %에서만 고른다 |

**보려는 것은 답 자체가 아니라 답이 흔들리는 폭입니다.**

같은 입력인데 답이 매번 다르다면, 그 프롬프트로 만든 결과는 신뢰하기 어렵습니다.
2일차 프롬프트 최적화에서 "judge 점수에 노이즈가 있다" 고 했던 것도 같은 이야기입니다.

분류처럼 답이 정해진 일에는 `temperature` 를 낮게 둡니다.
""".strip()

FEW_CONSTRAIN = """
### 제약 디코딩 — 다른 답이 나올 수 없게 막는다

지금까지는 프롬프트로 **부탁**했습니다. 아래는 다릅니다.

```python
extra_body={"structured_outputs": {"choice": ["긍정", "부정"]}}
```

이렇게 주면 모델은 **둘 중 하나밖에 출력할 수 없습니다.**
부탁이 아니라 **문법으로 막는** 것입니다. 생성 단계에서 다른 토큰의 확률을 0으로 만듭니다.

프롬프트를 아무리 잘 써도 가끔 형식을 어깁니다. 1,000건 중 3건이면 그 3건이
파이프라인을 멈춥니다. 제약 디코딩은 그 가능성을 없앱니다.

**아래 평가 루프부터는 전부 이 방식을 씁니다.** 그래야 세는 것이 가능합니다.

> 뒤쪽 QA 절에서는 같은 원리를 **JSON 스키마**로 씁니다.
> 선택지가 아니라 구조를 강제하는 형태입니다.
""".strip()

FEW_EVAL = """
## 재보기 — 여기서부터 숫자가 나옵니다

10건을 돌려 정확도를 잽니다. 흐름은 이렇습니다.

```
10건 반복 호출 → 긍정/부정 수집 → 1/0 으로 변환 → 정답과 비교 → 정확도
```

> **10건은 통계적으로 의미가 없습니다.** 한 건만 달라져도 10%가 움직입니다.
> 실습이라 작게 잡은 것이고, 실제로는 최소 수백 건을 씁니다.
> 그래도 **비교의 방향**은 볼 수 있습니다.

이 숫자를 기억해 두세요. 아래 few-shot 절에서 **같은 루프를 다시 돌려**
두 숫자를 비교하는 것이 이 노트북의 결론입니다.
""".strip()

FEW_SHOT = """
### 같은 루프를 few-shot 으로 다시 돌립니다

아래는 위 평가 루프와 **거의 같은 코드**입니다. 프롬프트만 바뀝니다.

```
zero-shot :  지시            → 분류
few-shot  :  지시 + 예시 5개  → 분류
```

예시를 `train[100:105]` 에서 그냥 잘라 씁니다. **라벨이 균형 잡혀 있는지 확인하지 않습니다.**
5개가 전부 긍정이면 모델이 긍정 쪽으로 기웁니다 — 실무에서는 이걸 확인합니다.

**두 정확도를 비교하세요.** 그것이 이 노트북의 결론입니다.

| 결과 | 뜻 |
|---|---|
| few-shot 이 높다 | 예시가 도움이 됐다 |
| 비슷하다 | 이 작업에는 지시만으로 충분하다 |
| few-shot 이 낮다 | 예시가 편향됐거나 문제를 헷갈리게 했다 |

**어느 쪽이든 배울 것이 있습니다.** few-shot 이 항상 낫지는 않습니다.
""".strip()

FEW_QA = """
### 정보 추출 — 형식을 어떻게 강제하나

앞에서는 **분류**였습니다. 답이 둘 중 하나라 `choice` 로 막을 수 있었습니다.

여기서는 **뽑아내기**입니다. 지문에서 인물 이름을 찾아 목록으로 만듭니다.
답이 몇 개일지 모르고 내용도 미리 알 수 없으니 `choice` 를 쓸 수 없습니다.

세 단계로 진행합니다. **차이를 보는 것이 요점입니다.**

| | 방법 | 결과 |
|---|---|---|
| 1 | 그냥 물어본다 | 자유 문장. 파싱 불가 |
| 2 | 프롬프트에 JSON 예시를 넣는다 | 대체로 JSON. **보장은 없음** |
| 3 | **스키마로 강제**한다 | 항상 그 구조 |

2번과 3번의 차이가 핵심입니다. 2번은 잘 되는 것처럼 보이다가 가끔 깨집니다.
그 "가끔" 이 운영에서 사고가 됩니다.

3번은 pydantic 모델을 JSON 스키마로 바꿔 `response_format` 에 넘깁니다.
2일차 Amazon 실습에서 쓴 것과 같은 방법입니다.

> **스키마에 길이 상한을 두는 것**을 잊지 마세요. 문자열 필드에 제한이 없으면
> 모델이 끝없이 쓰다가 `max_tokens` 에서 잘리고, JSON 이 깨집니다.
> 2일차에 실제로 그것 때문에 파이프라인이 죽은 적이 있습니다.
""".strip()

FEW_OUTRO = """
## 마무리

학습 없이 프롬프트만으로 어디까지 되는지 봤습니다.

- **형식을 말해주지 않으면** 모델은 제멋대로 답합니다. 모델 탓이 아닙니다
- 프롬프트로 **부탁**하는 것과 **제약 디코딩으로 막는** 것은 다릅니다.
  운영에서는 후자가 필요합니다
- **예시를 보여주는 것**(few-shot)이 항상 낫지는 않습니다. 재봐야 압니다
- 같은 입력에 답이 흔들리는 폭을 보세요. 그게 신뢰도입니다

### 그래서 학습은 언제 하나

여기까지가 **학습하지 않고 할 수 있는 것**입니다.
2일차에는 이것을 자동화하는 방법(프롬프트 최적화)도 봤습니다.

그래도 부족할 때 학습으로 갑니다. 다음 실습부터가 그것입니다 —
**SFT · DPO · GRPO**. 각각 필요한 데이터가 다르고, 비용도 다릅니다.

프롬프트로 얻은 이 숫자를 기억해 두세요. **학습이 그 위에 얼마를 더 얹어주는지**
가 판단 기준이 됩니다.
""".strip()


# ── Amazon 요약 실습 설명 보강 ───────────────────────────────────────
# 단계마다 짧은 설명은 이미 있다. 영어 헤딩을 한국어로 바꾸고,
# 빠진 개념 네 곳(스키마 강제 · 체이닝 · LLM 없는 정리 · 스케일 전환)을 채운다.
# 셀 55~62(구조 확인·변환·연결)는 이미 충실하므로 손대지 않는다.

AMZ_SCHEMA = """
### 출력을 JSON 으로 **강제**합니다

바로 아래 셀에 이 실습 전체를 관통하는 패턴이 처음 나옵니다.

```
pydantic 모델  →  model_json_schema()  →  response_format
```

프롬프트로 "JSON 으로 답해줘" 라고 **부탁**할 수도 있습니다. 대체로 따릅니다.
그런데 10건 중 1건이 어기면, 그 1건이 다음 단계를 통째로 무너뜨립니다.
7단계가 사슬로 엮여 있으니까요.

**스키마를 주면 다른 형태가 나올 수 없습니다.** 생성 단계에서 문법으로 막습니다.

### 길이 상한을 반드시 겁니다

`Field(max_length=...)` 가 붙어 있는 것을 보세요. **이게 없으면 실제로 터집니다.**

문자열 필드에 상한이 없으면 문법상 무한히 길어질 수 있습니다. 모델이 늘어지다가
`max_tokens` 에 걸려 JSON 이 **중간에서 잘리고**, 몇 셀 뒤에서
`Unterminated string` 으로 죽습니다. 원인에서 가장 먼 곳에서 터지는 셈입니다.

> 이 실습을 한국어로 바꾸면서 실제로 겪은 일입니다. 영어일 때는 우연히 넘어갔는데
> 출력이 길어지면서 드러났습니다. **"영어에서 됐으니 한국어도 된다" 가 성립하지 않습니다.**

### 실패를 삼키지 않습니다

각 함수 끝에 `finish_reason == "length"` 검사가 있습니다.
**잘린 응답은 에러가 아니라 정상 응답**이라 `try/except` 에 걸리지 않습니다.
여기서 잡지 않으면 원인에서 멀리 떨어진 곳에서 터집니다.
""".strip()

AMZ_CHAIN = """
### 단계를 사슬로 잇습니다 — 이 실습의 방법론

앞서 Step 1 에서 뽑은 **속성 종류 목록**을 Step 3 의 프롬프트에 **같이 넣습니다.**

```
Step 1:  "이 상품에 어떤 속성이 있나?"     →  [색상, 용량, 무게, ...]
                                                    │
Step 3:  "이 속성들의 **값**을 뽑아라"  ←──────────┘
         + 원문
```

한 번에 "상품 정보를 전부 뽑아줘" 라고 하면 모델이 무엇을 뽑을지 스스로 정합니다.
상품마다 다른 것을 뽑고, 형식도 제각각이 됩니다.

**나눠서 물으면** 각 단계가 단순해지고 결과가 안정됩니다.
그리고 중간 산출물을 눈으로 확인할 수 있어서, **어디서 틀렸는지** 알 수 있습니다.

> 이것이 이 실습의 방법론입니다. 큰 작업을 LLM 하나에 통째로 맡기지 않고,
> **작은 단계로 쪼개 사슬로 잇습니다.** 대신 호출 횟수가 늘어 비용이 커집니다.
""".strip()

AMZ_ORGANIZE = """
### 여기는 LLM 을 쓰지 않습니다

아래 두 셀을 보세요. **LLM 호출이 없습니다.** 순수 파이썬입니다.

앞에서 뽑은 속성값(Step 3)을 앞에서 만든 그룹(Step 2)에 맞춰 **재배치**하는 일입니다.
이건 규칙이 정해져 있어서 코드로 하면 됩니다.

**모든 것을 LLM 에 시키지 않는 것**도 설계입니다.

| | LLM 에 맡길 일 | 코드로 할 일 |
|---|---|---|
| 성격 | 판단·해석이 필요 | 규칙이 정해져 있다 |
| 예 | "이 속성은 어느 그룹인가" | 그룹에 맞춰 재배치 |
| 비용 | 호출 시간 · 돈 | 거의 0 |
| 신뢰도 | 흔들린다 | **항상 같다** |

LLM 을 한 번 덜 부르면 그만큼 빨라지고 싸지고, **결과가 흔들리지 않습니다.**
파이프라인을 설계할 때 매번 물어야 하는 질문입니다 — **"이건 꼭 LLM 이어야 하나?"**
""".strip()

AMZ_PIPELINE = """
### 하나씩 하던 것을 전부에 적용합니다

지금까지는 **상품 하나**로 각 단계를 확인했습니다. 이제 **10개 전부**에 돌립니다.

바뀌는 것은 `data.map(...)` 으로 감싸는 것뿐입니다. 함수는 그대로입니다.

```
지금까지:  gen_text_step_1(data[0]['text'])        ← 1개
여기부터:  data.map(lambda x: gen_text_step_1(...))  ← 전부
```

**상품 1개에 LLM 호출이 16번쯤** 들어갑니다. 10개면 160번입니다.
그래서 이 셀들이 몇 분 걸립니다.

> 이 숫자를 기억해 두세요. 강사가 배포한 데이터는 **상품 100개**로 만든 것이고,
> 호출이 1,600번 가까이 들어갔습니다. **강의 시간에 할 수 없는 분량**이라
> 미리 만들어 둔 것입니다.
>
> 단계를 쪼개면 결과가 안정되지만 **비용이 그만큼 늡니다.** 그 트레이드오프가
> 여기서 시간으로 드러납니다.
""".strip()


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

        # 기존 헤딩 2개는 교체한다 (영어 제목 + 설명 없는 제목)
        Rule(4, "## 토크나이저 테스트", CLS_TOKENIZER,
             "헤딩만 있던 것을 설명으로 확장 (강의 설명 보강)"),
        Rule(7, "# Korean Movie Review Classification", CLS_DATA,
             "영어 제목 → 한국어 + 데이터 설명 (강의 설명 보강)"),
    ], inserts=[
        InsertCells(1, [("markdown", CLS_INTRO)], "도입 — 무엇을 왜 하는가",
                    expect_head="# 2026 스택"),
        InsertCells(11, [("markdown", CLS_PREPROCESS)], "전처리 — 패딩을 왜 미루나",
                    expect_head="def preprocess_function"),
        InsertCells(14, [("markdown", CLS_METRIC)], "평가 기준을 먼저 정한다",
                    expect_head="accuracy = evaluate.load"),
        InsertCells(16, [("markdown", CLS_LABEL)], "라벨 이름을 왜 붙이나",
                    expect_head="id2label = "),
        InsertCells(17, [("markdown", CLS_MODEL)], "모델 — 분류 헤드를 얹는다",
                    expect_head="# 307M 인코더라"),
        InsertCells(19, [("markdown", CLS_TRAIN)], "학습 — num_train_epochs=0.1 의 의미",
                    expect_head="training_args = TrainingArguments"),
        InsertCells(20, [("markdown", CLS_INFER)], "추론 — pipeline 과 체크포인트 경로",
                    expect_head="from transformers import pipeline"),
    ], appends=[
        AppendCells([("markdown", CLS_OUTRO)], "마무리 — 되짚기 + 다음 실습 연결",
                    after_cell=21),
    ], drops=[21]),

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

        Rule(4, "## 토크나이저 테스트", "## 1. 토크나이저 — 분류 실습과 같은 것을 씁니다\n\n인코더 본체가 같으므로 토크나이저도 같습니다.\n앞 실습에서 본 것을 다시 확인하고 넘어갑니다.",
             "헤딩만 있던 것을 짧게 확장. 분류와 중복이라 길게 쓰지 않는다"),
        Rule(7, "# NER Classification", NER_LABELS,
             "영어 제목 → 한국어 + BIO 태그 설명 (강의 설명 보강)"),
    ], inserts=[
        InsertCells(1, [("markdown", NER_INTRO)], "도입 — 분류와 무엇이 다른가",
                    expect_head="# 2026 스택"),
        InsertCells(13, [("markdown", NER_ALIGN)], "라벨 정렬 — 이 실습의 핵심",
                    expect_head="def align_labels_with_tokens"),
        InsertCells(16, [("markdown", NER_METRIC)], "평가 — 정확도로는 속는다",
                    expect_head="import evaluate"),
        InsertCells(21, [("markdown", NER_TRAIN)], "학습 — 분류와 다른 점만",
                    expect_head="training_args = TrainingArguments"),
        InsertCells(22, [("markdown", NER_INFER)], "추론 — 쪼갠 것을 다시 합친다",
                    expect_head="from transformers import pipeline"),
    ], appends=[
        AppendCells([("markdown", NER_OUTRO)], "마무리 — 되짚기 + 다음 실습 연결",
                    after_cell=22),
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

        # ── 강의 설명 보강 ──────────────────────────────────────
        Rule(1, "## 환경 세팅", SFT_INTRO, "도입 — 왜 SFT 인가 (강의 설명)"),
        Rule(4, "## 데이터셋 불러오기", SFT_DATA, "데이터 — 어제 만든 것 (강의 설명)"),
        Rule(11, SHOPIFY, SFT_OBSERVE,
             "원본 HF 튜토리얼(Shopify) 잔재. 현재 데이터와 무관하다 → 교체"),
        Rule(12, "## 토크나이저 불러오기", SFT_TEMPLATE,
             "챗 템플릿 — Base 모델엔 대화 형식이 없다 (강의 설명)"),
        Rule(14, "## 챗 템플릿 적용하기", SFT_APPLY, "적용과 길이 확인 (강의 설명)"),
        Rule(17, "## 모델 학습 준비하기", SFT_LORA, "LoRA — 전부 학습시키지 않는다 (강의 설명)"),
        Rule(21, "## 학습해보기!", SFT_RUN, "학습 — loss 를 어떻게 읽나 (강의 설명)"),
        Rule(24, "## 학습한 모델로 생성해보기", SFT_GEN, "써보기 — 템플릿을 다시 넣는 이유 (강의 설명)"),
    ], inserts=[
        InsertCells(7, [("markdown", SFT_MESSAGES)], "대화 형식으로 바꾸기",
                    expect_head="def build_messages"),
        InsertCells(20, [("markdown", SFT_TRAINER)], "학습 설정 — 1일차와 값이 다른 이유",
                    expect_head="from trl import SFTTrainer"),
    ], appends=[
        AppendCells([("markdown", SFT_OUTRO)], "마무리 — 되짚기 + DPO 연결", after_cell=28),
    ], drops=[28], globals_=[
        GlobalRule("'data/test_model'", "'data/sft_model'",
                   "SFT·DPO·GRPO 가 전부 같은 폴더에 저장해 서로 덮어쓰고 있었다. "
                   "DPO 가 SFT 결과를 이어받으려면 갈라야 한다", 2),
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

        # ── SFT 를 이어받는다 (설명과 실습의 모순 해소) ──────────
        Rule(14, DPO_RESUME_OLD, DPO_RESUME_NEW,
             "DPO 는 SFT 를 전제하는데 베이스에서 시작하고 있었다. "
             "같은 노트북의 ORPO 섹션이 그걸 경고한다"),
        Rule(6, DPO_SPLIT_OLD, DPO_SPLIT_NEW,
             "영문 TODO 주석 제거 + 고정 인덱스를 비율 분할로 (SFT 와 일관)"),

        # ── 강의 설명 보강 ──────────────────────────────────────
        Rule(1, "## 환경 세팅", DPO_INTRO, "도입 — 왜 선호 학습인가 (강의 설명)"),
        Rule(4, "## 데이터셋 불러오기", DPO_DATA, "데이터 — 선호 쌍 세 열 (강의 설명)"),
        Rule(10, "## 챗 템플릿 적용하기", DPO_FORMAT,
             "형식 — SFT 와 달리 손으로 조립하는 이유 (강의 설명)"),
        Rule(13, "## 모델 학습 준비하기", DPO_MODEL,
             "모델 — 이어받기 + 참조 모델이 왜 안 보이나 (강의 설명)"),
        Rule(16, "## 학습해보기!", DPO_RUN, "학습 — rewards 를 어떻게 읽나 (강의 설명)"),
        Rule(19, "## 학습한 모델로 생성해보기", DPO_GEN, "써보기 (강의 설명)"),
        Rule(8, "## 토크나이저 불러오기",
             """## 토크나이저 불러오기

SFT 실습과 같은 설정입니다 — 같은 토크나이저, 같은 챗 템플릿.
**학습할 때와 쓸 때의 형식이 같아야 한다**는 원칙이 실습을 건너서도 이어집니다.""",
             "알맹이 없는 헤딩 보강 (산문 정비)"),
    ], [
        GlobalRule("'data/test_model'", "'data/dpo_model'",
                   "SFT·DPO·GRPO 가 전부 같은 폴더에 저장해 서로 덮어쓰고 있었다", 2),
    ], appends=[ORPO_SECTION], inserts=[
        InsertCells(6, [("markdown", DPO_SPLIT)], "얼마나 쓸 것인가",
                    expect_head="from datasets import DatasetDict"),
        InsertCells(15, [("markdown", DPO_TRAINER)], "학습 설정 — SFT 와 다른 값",
                    expect_head="from trl import DPOTrainer"),
    ], drops=[23]),

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

        # ── 강의 설명 보강 ──────────────────────────────────────
        Rule(1, "## 환경 세팅", GRPO_INTRO, "도입 — 왜 채점 함수인가 (강의 설명)"),
        Rule(4, "## 데이터셋 불러오기", GRPO_DATA,
             "데이터 — verifiable 이 무슨 뜻인가 (강의 설명)"),
        Rule(10, "## 챗 템플릿 적용하기", GRPO_FORMAT,
             "생각하는 형식 — 채점 함수와 짝이다 (강의 설명)"),
        Rule(13, "## 보상 함수 만들기", GRPO_REWARD,
             "채점 함수 — 이 실습의 핵심 (강의 설명)"),
        Rule(16, "## 모델 학습 준비하기", GRPO_TRAIN,
             "학습 설정 — num_generations 가 GRPO 의 정의 (강의 설명)"),
        Rule(22, "## 학습한 모델로 생성해보기", GRPO_GEN,
             "생성 — 템플릿이 학습 때와 다른 이유 (강의 설명)"),
        Rule(8, "## 토크나이저 불러오기",
             """## 토크나이저 불러오기

SFT·DPO 와 같은 설정입니다. 생성할 때 템플릿이 한 군데 달라지는데,
그 이유는 아래 생성 절에서 다룹니다.""",
             "알맹이 없는 헤딩 보강 (산문 정비)"),
        Rule(19, "## 학습해보기!",
             """## 학습

시간이 꽤 걸립니다 — 매 스텝 답을 4개씩 **생성하면서** 학습하기 때문입니다.""",
             "알맹이 없는 헤딩 보강 (산문 정비)"),
    ], inserts=[], appends=[
        AppendCells([("markdown", GRPO_OUTRO)], "마무리 — 세 방법 정리", after_cell=26),
    ], drops=[26], globals_=[
        GlobalRule("'data/test_model'", "'data/grpo_model'",
                   "SFT·DPO·GRPO 가 전부 같은 폴더에 저장해 서로 덮어쓰고 있었다", 2),
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

        # ── 강의 설명 보강 ──────────────────────────────────────
        # 64셀이지만 반복 구조라 대표 지점에만 넣는다.
        Rule(3, "## 환경 준비", FEW_INTRO, "도입 — 학습 전에 어디까지 되나 (강의 설명)"),
        Rule(3, "(Google Colab 환경에서 사용하세요)", "",
             "VESSL 에서 하므로 Colab 안내는 맞지 않는다"),
        Rule(20, "### 분석 결과를 형식에 맞게 작성해보기", FEW_MESSY,
             "출력이 제멋대로인 것이 의도된 실패라는 것 (강의 설명)"),
        Rule(25, "### 파라메터를 이용하여 다양한 생성 만들어보기", FEW_PARAM,
             "파라미터 실험 — 보려는 것은 흔들리는 폭 (강의 설명. '파라메터' 표기도 정정)"),
        Rule(29, "### 모델 성능 예측해보기", FEW_EVAL,
             "평가 루프 — 10건은 통계적으로 무의미하다는 것도 (강의 설명)"),
        Rule(36, "### Few-Shot Prompt 만들어보기", FEW_SHOT,
             "같은 루프를 다시 — 두 숫자 비교가 결론 (강의 설명)"),
        Rule(47, "### QA 정답 단어 추출", FEW_QA,
             "형식 강제 3단계 — 부탁 vs 스키마 (강의 설명)"),

        # ── 25년 원본 문장 재작성 (산문 정비) ─────────────────────
        Rule(1, """#### 주의!!

이 실습은 가급적 NVIDIA GPU가 설치된 컴퓨터 환경이거나 Google Colab에서 진행해주세요.""",
             "> 이 실습은 GPU 가 있는 VESSL 워크스페이스에서 진행합니다.",
             "Colab 안내는 현행 환경과 맞지 않는다 (산문 정비)"),
        Rule(2, "https://huggingface.co/LGAI-EXAONE",
             "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
             "모델 교체(F-1)에 맞춰 레퍼런스 링크도 갱신"),
        Rule(5, "여러분이 실습할 모델을 실행하는 명령어 입니다. 실행후 강사님의 지시가 있을때 까지 다른 코드들을 실행하지 마세요.",
             """모델 서버(vLLM)를 띄우는 명령입니다.
실행 후 강사 안내가 있을 때까지 다음 코드는 실행하지 말아 주세요.""",
             "맞춤법 (산문 정비)"),
        Rule(10, "### 감정 분석", "### 감정 분류 — 먼저 한 건 물어봅니다",
             "알맹이 없는 헤딩에 방향 제시 (산문 정비)"),
        Rule(16, "### 간단한 감정분석", "### 같은 질문을 10건 돌려봅니다",
             "표기 불일치(감정 분석/감정분석) 해소 + 방향 제시 (산문 정비)"),
    ], inserts=[
        InsertCells(8, [("markdown", FEW_API)], "completions 와 chat 의 차이",
                    expect_head="# 모델이 잘 실행되는지"),
        InsertCells(28, [("markdown", FEW_CONSTRAIN)], "제약 디코딩 — 개념 도입점이었다",
                    expect_head="completion = client.chat.completions.create"),
    ], appends=[
        AppendCells([("markdown", FEW_OUTRO)], "마무리 — 그래서 학습은 언제 하나",
                    after_cell=64),
    ], drops=[64], globals_=[
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

        # ── 강의 설명 보강 ──────────────────────────────────────
        # 셀 55~62(구조 확인·변환·연결)는 이미 충실하므로 손대지 않는다.
        Rule(11, "### Step 1", "### 1단계 — 모델이 무엇을 아는지 묻는다",
             "영어 헤딩 → 한국어 (강의 설명 보강)"),
        Rule(15, "### Step 2", "### 2단계 — 정보를 그룹으로 묶는다",
             "영어 헤딩 → 한국어 (강의 설명 보강)"),
        Rule(19, "### Step 3", "### 3단계 — 실제 값을 뽑는다",
             "영어 헤딩 → 한국어 (강의 설명 보강)"),
        Rule(25, "### Step 4", "### 4단계 — 구매자 유형을 묻는다",
             "영어 헤딩 → 한국어 (강의 설명 보강)"),
        Rule(29, "### Step 5", "### 5단계 — 뽑은 정보를 정리한다",
             "영어 헤딩 → 한국어 (강의 설명 보강)"),
        Rule(34, "### Step 6", "### 6단계 — 속성 그룹별로 요약한다",
             "영어 헤딩 → 한국어 (강의 설명 보강)"),
        Rule(41, "### Step 7", "### 7단계 — 구매자 유형별로 요약한다",
             "영어 헤딩 → 한국어 (강의 설명 보강)"),
        Rule(49, "### Step Final", "### 마지막 — 하나의 파이프라인으로",
             "영어 헤딩 → 한국어 (강의 설명 보강)"),
        Rule(22, "앞서 Step 1에서 추출한 내용을 Prompt에 같이 넣어볼까요?", AMZ_CHAIN,
             "체이닝 — 이 실습의 방법론인데 한 줄뿐이었다 (강의 설명 보강)"),

        # ── 25년 원본 문장 재작성 (산문 정비) ─────────────────────
        Rule(1, """이번 실습은 아마존 상품 정보 데이터를 추출해서 사용자에게 적합한 정보를 전달하는 요약문을 만들어 볼겁니다.

오픈소스 GPT를 이용하여 데이터 생성 파이프라인이 어떤것이고 어떻게 요약문을 만들어가는지 살펴보려 합니다.""",
             """# 상품 정보 추출과 요약 — 데이터 생성 파이프라인

아마존 상품 설명에서 정보를 뽑아, 구매자에게 맞는 요약문을 만들어 봅니다.
오픈소스 LLM 으로 **데이터 생성 파이프라인**을 한 단계씩 쌓아 가는 실습입니다.

## 무엇을 하게 되나

1. 모델에게 상품에 대해 **무엇을 아는지** 묻습니다
2. 그 답을 다음 단계의 입력으로 넘기며, 작은 단계 7개를 **사슬로 잇습니다**
3. 마지막에 전체 상품에 돌려 **학습 데이터**를 만듭니다 — 내일 SFT 가 이걸 씁니다

왜 큰 작업을 LLM 하나에 통째로 맡기지 않고 작게 쪼개는지는,
파이프라인을 만들어 가면서 직접 확인하게 됩니다.""",
             "도입 재작성 — 제목·맞춤법·개요 (산문 정비)"),
        Rule(8, """여러분들이 실습한 GPT 모델을 실행해보겠습니다.
실행후에 강사님의 지시가 있을 때 까지 다른 코드 실행을 하지 말아주세요.""",
             """모델 서버(vLLM)가 떠 있는지 먼저 확인합니다.
확인 후 강사 안내가 있을 때까지 다음 코드는 실행하지 말아 주세요.""",
             "맞춤법·현행 실행 방식 (산문 정비)"),
        Rule(11, """모델이 이해하는 제품 정보가 무엇인지 알아보기

- 데이터 추출 시에는 내가 요구하는것을 모델에 요청하는 것보다 모델이 무엇을 알고있는지를 물어보는게 문제해결에 도움이 됩니다.""",
             """- 무엇을 뽑을지 우리가 정하기 전에, **모델이 이 상품에서 무엇을 알아볼 수 있는지**부터 묻습니다.
- 원하는 것을 바로 시키는 것보다 모델이 아는 것을 먼저 묻는 쪽이 문제 해결에 도움이 됩니다.""",
             "옛 제목 중복 제거 + 맞춤법 (산문 정비)"),
        Rule(15, """모델이 이해하는 제품 정보를 묶어보기

- 모델이 알고있는 정보가 많을 경우에는 정보를 그룹으로 만들어서 사람 입장에서도 보기 쉬운 데이터 구성을 해둡니다.""",
             """- 항목이 많으면 **그룹으로 묶어** 사람이 보기 쉬운 구조로 정리해 둡니다.
- 이 그룹이 뒤 단계(5·6단계)에서 요약의 뼈대가 됩니다.""",
             "옛 제목 중복 제거 + 맞춤법 (산문 정비)"),
        Rule(19, """제품 정보 추출하기

- 정보를 추출할 때는 내가 추출할 대상을 명시적으로 언급해주는 것이 도움이 됩니다.""",
             """- 이제 속성의 **실제 값**을 뽑습니다.
- 추출할 대상을 명시적으로 나열해 주면 결과가 안정됩니다 — 아래에서 1단계의 답을 그 목록으로 씁니다.""",
             "옛 제목 중복 제거 (산문 정비)"),
        Rule(25, """제품을 기대하는 고객에 대한 인사이트 물어보기

- 때로는 GPT 모델이 생성한 인사이트 결과를 실험적으로 활용하기에 용이할 때가 있습니다.
- 이번 실습에서는 사람이 직접 인사이트를 분석할 시간은 대신해서 GPT에게 물어봅시다.""",
             """- 이 상품을 살 만한 **구매자 유형**을 모델에게 묻습니다.
- 원래는 사람이 리뷰와 시장을 보고 얻는 인사이트지만, 여기서는 그 초안을 모델에게 맡깁니다.
  **실험적 용도**라는 것은 기억해 두세요.""",
             "옛 제목 중복 제거 + 어색한 문장 재작성 (산문 정비)"),
        Rule(29, """요약문을 만들기 위해 추출한 정보 정리하기

- 요약문을 만들기 위해 앞서 추출한 데이터를 Subsection별로 묶어보겠습니다.""",
             "- 앞서 뽑은 값들을 그룹(subsection)별로 다시 묶어, 요약문을 만들 재료로 정리합니다.",
             "옛 제목 중복 제거 (산문 정비)"),
        Rule(34, """제품 정보만 모아 요약하기

- 사용자에게 전달할 문장을 만들기 전에 제품 정보만 미리 문장으로 구성해 보겠습니다.""",
             "- 구매자 이야기를 붙이기 전에, **제품 정보만으로** 먼저 문장을 만들어 둡니다.",
             "옛 제목 중복 제거 (산문 정비)"),
        Rule(41, """고객에 대한 특징을 첨가하여 요약하기

- 구매할 사용자에게 전달할 내용을 앞서 만든 제품 정보 요약과 합쳐보겠습니다.""",
             "- 구매자 유형별 관심사를 앞의 제품 요약과 합쳐 **최종 요약문**을 만듭니다.",
             "옛 제목 중복 제거 (산문 정비)"),
        Rule(49, """지금까지 만들어온 과정을 하나의 파이프라인으로 만들어보겠습니다.

- datasets 라이브러리를 활용하여 데이터 생성 파이프라인을 만들어 봅시다.""",
             "- 지금까지 한 단계씩 확인한 과정을, 한 번에 실행되는 **하나의 파이프라인**으로 묶습니다.",
             "옛 제목 중복 제거 (산문 정비)"),
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
    ], inserts=[
        InsertCells(13, [("markdown", AMZ_SCHEMA)], "스키마 강제 — 이 실습의 핵심 패턴",
                    expect_head="# 길이 상한이"),
        InsertCells(30, [("markdown", AMZ_ORGANIZE)], "여기는 LLM 을 쓰지 않는다",
                    expect_head="# Feature list"),
        InsertCells(50, [("markdown", AMZ_PIPELINE)], "스케일 전환 — 1개에서 전부로",
                    expect_head="# Step 1"),
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

        # ── 강의 설명 보강 ──────────────────────────────────────
        Rule(1, "# BM25 Retriever RAG 실습", RAG_INTRO,
             "도입 — 학습이 아니라 검색으로 푸는 문제 (강의 설명)"),
        Rule(2, "## Setup", RAG_SETUP,
             "영어 헤딩 → 한국어 + embed_model=None 이유 (강의 설명)"),
        Rule(5, "## Load Data", RAG_DATA,
             "영어 헤딩 → 한국어 + 청킹이 왜 중요한가 (강의 설명)"),
        Rule(12, "## BM25 Retriever + Disk Persistance", RAG_BM25,
             "영어 헤딩(철자 오류 포함) → 한국어 + 어휘검색 vs 의미검색 (강의 설명)"),
    ], inserts=[
        InsertCells(17, [("markdown", RAG_ENGINE)], "refine 모드 — 개념이 완전 공백이었다",
                    expect_head="from llama_index.core import get_response_synthesizer"),
        InsertCells(23, [("markdown", RAG_QUERY)], "근거 문서를 보라 — 검색 실패와 생성 실패 구분",
                    expect_head="response = query_engine.query"),
        InsertCells(27, [("markdown", RAG_SUBQ)], "SubQuestionQueryEngine — 개념이 완전 공백이었다",
                    expect_head="from llama_index.core.query_engine import SubQuestionQueryEngine"),
    ], appends=[
        AppendCells([("markdown", RAG_OUTRO)], "마무리 — 언제 RAG 이고 언제 학습인가",
                    after_cell=31),
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

    # --- 구조 변경(삽입·삭제)은 맨 마지막에, **위치 내림차순 한 번에** ---
    # 삽입과 삭제를 따로 돌리면 서로의 위치를 밀어버린다. 실제로 그렇게 만들었다가
    # drops 가 엉뚱한 셀을 가리켰다. 한 리스트로 합쳐 높은 번호부터 처리하면
    # 모든 위치를 **원본 번호 그대로** 적을 수 있다.
    ops = ([(d, "drop", None) for d in mig.drops]
           + [(i.before_cell, "insert", i) for i in mig.inserts])
    for pos, kind, payload in sorted(ops, key=lambda t: -t[0]):
        if not (1 <= pos <= len(cells)):
            raise SystemExit(
                f"\n[중단] {mig.rel} {kind} 위치 {pos} 가 범위를 벗어납니다 "
                f"(총 {len(cells)})."
            )
        if kind == "drop":
            body = cell_source(cells[pos - 1]).strip()
            if body:
                raise SystemExit(
                    f"\n[중단] {mig.rel} 셀 {pos} 는 비어 있지 않습니다. 지우지 않습니다.\n"
                    f"  내용: {body[:120]!r}"
                )
            del cells[pos - 1]
            applied += 1
            log.append(f"  [삭제] 빈 셀 {pos}")
        else:
            anchor = cell_source(cells[pos - 1])
            head = anchor.lstrip().splitlines()[0] if anchor.strip() else ""
            if not head.startswith(payload.expect_head):
                raise SystemExit(
                    f"\n[중단] {mig.rel} 셀 {pos} 가 예상한 앵커가 아닙니다.\n"
                    f"  사유: {payload.why}\n"
                    f"  예상 시작: {payload.expect_head!r}\n"
                    f"  실제 첫 줄: {head[:120]!r}\n"
                    f"  → 원본이 바뀌었거나 앞선 규칙이 셀을 옮겼습니다."
                )
            for off, (kind_, source) in enumerate(payload.cells):
                cells.insert(pos - 1 + off, make_cell(kind_, source))
            applied += 1
            log.append(f"  [삽입] 셀 {pos} 앞에 {len(payload.cells)}개 · {payload.why}")

    if not dry:
        save_notebook(out_path, nb)   # 마크다운 조사 붙여쓰기가 여기서 일괄 적용된다

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
