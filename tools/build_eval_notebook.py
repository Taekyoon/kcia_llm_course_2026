"""⑪ "태스크 정의와 평가" 실습 노트북을 만든다.

    uv run python tools/build_eval_notebook.py

왜 새로 만드는가
----------------
커리큘럼 4단원(LLM Post-training)의 **첫 항목**이 "태스크 정의와 평가" 인데,
대응 자료가 3일차 p73-82 의 **서술형 10장뿐이고 실습이 0건**이었다.
BLEU·ROUGE 를 코드 한 줄 없이 설명만 한다 (review/01_대조표.md ⑪ 부분충족).

게다가 4단원 첫 항목인데 **과정 최종일 마지막 섹션**에 배치돼 있다.
"무엇을 만들지 정하고 어떻게 잴지 정한 다음에 학습 방법을 배운다" 가 자연스러운데
지금은 반대다.

설계 방침
---------
- **2일차 SFT 실습 결과물을 평가한다.** 앞 실습과 이어져야 "왜 재는지" 가 와닿는다.
  SFT 모델이 없어도 돌아가도록 폴백을 둔다.
- **자동 지표의 한계를 직접 보여준다.** BLEU·ROUGE 는 표면 일치를 재기 때문에
  뜻이 같아도 표현이 다르면 점수가 낮다. 숫자를 믿기 전에 이걸 알아야 한다.
- **LLM-as-judge** 를 함께 다룬다. 2026년 실무에서 가장 많이 쓰는 방식인데
  기존 자료에는 없었다.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "notebook" / "2일차" / "HPC_평가실습.ipynb"

MD, CODE = "markdown", "code"
CELLS: list[tuple[str, str]] = []


def md(t: str) -> None:
    CELLS.append((MD, t.strip("\n")))


def code(t: str) -> None:
    CELLS.append((CODE, t.strip("\n")))


# =====================================================================
md("""
# 태스크 정의와 평가

모델을 학습시키기 전에 정해야 하는 것이 있습니다. **무엇을 잘하게 만들 것인가**, 그리고
**잘하는지 어떻게 확인할 것인가**입니다.

이게 정해지지 않으면 학습은 할 수 있어도 **좋아졌는지 알 수 없습니다.**
"느낌상 나아진 것 같다" 로는 다음 결정을 내릴 수 없습니다.

## 무엇을 하게 되나

1. **태스크를 정의**합니다 — 입력·출력·성공 기준
2. **자동 지표**를 계산합니다 — BLEU, ROUGE
3. **자동 지표의 한계**를 직접 확인합니다 ← 이게 핵심입니다
4. **LLM-as-judge** 로 평가합니다 — 사람 대신 모델이 채점
5. 어떤 상황에 어떤 방법을 쓸지 정리합니다

## 앞 실습과의 연결

2일차에 SFT·DPO·GRPO 로 모델을 학습시켰습니다. 그런데 **정말 좋아졌는지** 는
확인하지 않았습니다. 여기서 그걸 합니다.
""")

code("""
%pip install -q -U evaluate datasets transformers rouge_score sacrebleu 'openai<3'
""")

code("""
import json
import re
from statistics import mean

import evaluate
""")

# ---------------------------------------------------------------------
md("""
## 1. 태스크 정의

평가는 지표를 고르는 것부터 시작하지 않습니다. **태스크를 적는 것**부터입니다.

아래 네 가지가 정해져야 지표를 고를 수 있습니다.

| 항목 | 이번 실습의 예 |
|---|---|
| 입력 | 사용자 질문 (한국어) |
| 출력 | 질문에 답하는 한국어 문장 |
| 성공 기준 | 사실이 맞고, 질문에 답하고, 한국어로 자연스러울 것 |
| 실패 사례 | 영어로 답함 · 질문과 무관 · 사실 오류 · 답을 회피 |

**성공 기준이 여러 개면 지표도 여러 개 필요합니다.** 하나로 뭉뚱그리면
무엇이 나빠졌는지 알 수 없습니다.
""")

code("""
# 평가 데이터. 실제로는 수백~수천 건을 쓰지만 실습이라 소규모로 만든다.
# 중요한 건 개수가 아니라 **레퍼런스가 태스크를 제대로 대표하는가** 다.
eval_set = [
    {
        "question": "대한민국의 수도는 어디인가요?",
        "reference": "대한민국의 수도는 서울입니다.",
    },
    {
        "question": "광합성이 무엇인지 간단히 설명해주세요.",
        "reference": "광합성은 식물이 빛 에너지를 이용해 이산화탄소와 물로 포도당을 만드는 과정입니다.",
    },
    {
        "question": "김치는 어떻게 만드나요?",
        "reference": "배추를 소금에 절인 뒤 고춧가루, 마늘, 생강 등의 양념을 버무려 발효시켜 만듭니다.",
    },
    {
        "question": "파이썬에서 리스트와 튜플의 차이는 무엇인가요?",
        "reference": "리스트는 생성 후 값을 바꿀 수 있고 튜플은 바꿀 수 없습니다.",
    },
]
print(f"평가 데이터 {len(eval_set)}건")
""")

# ---------------------------------------------------------------------
md("""
## 2. 모델 응답 만들기

2일차에서 학습시킨 SFT 모델이 있으면 그것을 씁니다.
없으면 미리 준비한 예시 응답으로 진행합니다 — **평가 방법 자체를 익히는 것이 목적**이라
모델이 무엇이든 흐름은 같습니다.
""")

code("""
SFT_DIR = "data/test_model"     # 2일차 SFT 실습의 출력 경로

predictions = None
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SFT_DIR)
    model = AutoModelForCausalLM.from_pretrained(SFT_DIR, device_map="auto")

    preds = []
    for ex in eval_set:
        msgs = [{"role": "user", "content": ex["question"]}]
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)
        out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(text.strip())
    predictions = preds
    print(f"SFT 모델({SFT_DIR})로 {len(preds)}건 생성했습니다.")
except Exception as e:
    print(f"SFT 모델을 불러오지 못했습니다: {type(e).__name__}")
    print("→ 준비된 예시 응답으로 진행합니다. 평가 흐름은 동일합니다.\\n")
""")

code("""
if predictions is None:
    # 일부러 성격이 다른 응답을 섞었다. 지표가 이것들을 어떻게 다르게 보는지가 관전 포인트다.
    predictions = [
        "서울이 대한민국의 수도입니다.",                          # 뜻은 같고 표현이 다름
        "광합성은 식물이 햇빛을 받아 양분을 만드는 작용입니다.",      # 맞지만 더 짧고 표현이 다름
        "배추를 절이고 양념을 발라 숙성시킵니다.",                  # 맞지만 훨씬 축약됨
        "리스트는 변경 가능하고 튜플은 변경 불가능합니다.",          # 사실상 정답, 어휘만 다름
    ]

for ex, p in zip(eval_set, predictions):
    print(f"Q: {ex['question']}")
    print(f"  정답: {ex['reference']}")
    print(f"  응답: {p}")
    print()
""")

# ---------------------------------------------------------------------
md("""
## 3. 자동 지표 — BLEU 와 ROUGE

두 지표 모두 **모델 응답과 정답이 표면적으로 얼마나 겹치는가**를 잽니다.

| 지표 | 무엇을 보나 | 주로 쓰는 곳 |
|---|---|---|
| **BLEU** | 응답의 n-gram 중 정답에도 있는 비율 (정밀도 중심) | 기계 번역 |
| **ROUGE** | 정답의 n-gram 중 응답에도 있는 비율 (재현율 중심) | 요약 |

`ROUGE-1` 은 단어 단위, `ROUGE-2` 는 2단어 연속, `ROUGE-L` 은 가장 긴 공통 부분열입니다.
""")

code("""
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

references = [ex["reference"] for ex in eval_set]

bleu_score = bleu.compute(predictions=predictions,
                          references=[[r] for r in references])
rouge_score = rouge.compute(predictions=predictions, references=references)

print(f"BLEU     {bleu_score['score']:.2f}")
for k in ["rouge1", "rouge2", "rougeL"]:
    print(f"{k:<9}{rouge_score[k]:.4f}")
""")

code("""
# 건별로 봐야 무슨 일이 일어나는지 보인다. 평균만 보면 안 된다.
print(f"{'질문':<34}{'BLEU':>8}{'ROUGE-L':>10}")
print("-" * 54)
for ex, p in zip(eval_set, predictions):
    b = bleu.compute(predictions=[p], references=[[ex["reference"]]])["score"]
    r = rouge.compute(predictions=[p], references=[ex["reference"]])["rougeL"]
    print(f"{ex['question'][:32]:<34}{b:>8.1f}{r:>10.3f}")
""")

# ---------------------------------------------------------------------
md("""
## 4. ★ 자동 지표의 한계

여기가 이 실습에서 가장 중요한 부분입니다.

BLEU 와 ROUGE 는 **글자와 단어가 겹치는지**를 봅니다. 뜻이 같은지는 모릅니다.
그래서 **뜻이 같아도 표현이 다르면 점수가 낮게** 나옵니다.

아래에서 직접 확인해봅시다.
""")

code("""
question = "대한민국의 수도는 어디인가요?"
reference = "대한민국의 수도는 서울입니다."

cases = [
    ("정답과 완전히 동일",   "대한민국의 수도는 서울입니다."),
    ("뜻이 같고 표현만 다름", "서울이 대한민국의 수도입니다."),
    ("맞지만 더 짧음",       "서울입니다."),
    ("맞지만 더 자세함",     "대한민국의 수도는 서울특별시이며, 인구는 약 940만 명입니다."),
    ("★ 틀렸는데 표현이 비슷", "대한민국의 수도는 부산입니다."),
]

print(f"{'경우':<24}{'BLEU':>8}{'ROUGE-L':>10}")
print("-" * 44)
for name, pred in cases:
    b = bleu.compute(predictions=[pred], references=[[reference]])["score"]
    r = rouge.compute(predictions=[pred], references=[reference])["rougeL"]
    print(f"{name:<24}{b:>8.1f}{r:>10.3f}")
""")

md("""
### 결과를 읽어보세요

두 가지가 보일 겁니다.

**1) 뜻이 같아도 표현이 다르면 점수가 떨어집니다.**
"서울이 대한민국의 수도입니다" 는 정답과 뜻이 완전히 같은데 점수가 낮습니다.
어순만 바뀌었을 뿐인데도요.

**2) 틀린 답이 높은 점수를 받습니다.**
`대한민국의 수도는 부산입니다` 는 **사실이 틀렸는데** 정답과 글자가 거의 같아서
점수가 높게 나옵니다. 한 단어만 다르니까요.

**이것이 자동 지표의 근본적 한계입니다.** 표면 일치는 잴 수 있어도
**의미와 사실 여부는 못 잽니다.**

그렇다고 쓸모없는 것은 아닙니다. **빠르고 싸고 재현 가능**합니다.
번역이나 요약처럼 정답 형태가 어느 정도 정해진 작업에는 여전히 유용합니다.
문제는 **열린 생성**에서 이걸 유일한 기준으로 삼을 때입니다.
""")

# ---------------------------------------------------------------------
md("""
## 5. LLM-as-judge

그래서 요즘은 **모델에게 채점을 맡깁니다.**

사람이 채점하는 것이 가장 정확하지만 느리고 비쌉니다.
LLM-as-judge 는 그 중간입니다 — 의미를 어느 정도 이해하면서 자동화됩니다.

**핵심은 채점 기준을 명확히 주는 것입니다.** "좋은 답변인가?" 라고만 물으면
기준이 매번 달라집니다. 무엇을 어떻게 볼지 지정해야 합니다.

> 앞 실습에서 띄운 vLLM 서버를 씁니다.
> `source /opt/vllm-env/bin/activate && vllm serve ... --port 8000`
""")

code("""
from openai import OpenAI

BASE = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
client = OpenAI(api_key="EMPTY", base_url=BASE, timeout=60.0)

# 서버가 없으면 이 아래 셀들이 전부 깨진다. 먼저 확인하고 안내한다.
try:
    client.models.list()
    JUDGE_OK = True
    print(f"vLLM 서버 연결됨 — {MODEL}")
except Exception as e:
    JUDGE_OK = False
    print(f"vLLM 서버에 붙지 못했습니다: {type(e).__name__}")
    print()
    print("  별도 터미널에서 서버를 띄우세요:")
    print("    source /opt/vllm-env/bin/activate")
    print("    nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \\\\")
    print("        --gpu-memory-utilization 0.80 --max-model-len 16384 > /tmp/vllm.log 2>&1 &")
    print()
    print("  서버 없이도 앞의 BLEU/ROUGE 부분은 이미 확인했습니다.")
    print("  아래 LLM-as-judge 셀들은 건너뜁니다.")

JUDGE_PROMPT = \"\"\"당신은 한국어 QA 시스템의 답변을 채점합니다.

[질문]
{question}

[모범 답안]
{reference}

[채점 대상 답변]
{prediction}

다음 세 항목을 각각 1~5점으로 채점하세요.

- correctness: 사실이 맞는가. 모범 답안과 내용이 일치하는가
- relevance: 질문에 실제로 답하고 있는가
- fluency: 한국어가 자연스러운가

표현이 모범 답안과 달라도 **뜻이 맞으면 correctness 는 높게** 주세요.
반대로 표현이 비슷해도 **사실이 틀렸으면 낮게** 주세요.

reason 은 **한 문장으로 짧게** 쓰세요.
JSON 으로만 답하세요.\"\"\"

SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "relevance":   {"type": "integer", "minimum": 1, "maximum": 5},
        "fluency":     {"type": "integer", "minimum": 1, "maximum": 5},
        # maxLength 를 걸지 않으면 모델이 끝없이 쓴다. 실제로 한 요청이 90초를 넘겼다.
        "reason":      {"type": "string", "maxLength": 150},
    },
    "required": ["correctness", "relevance", "fluency", "reason"],
}


def judge(question, reference, prediction):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, reference=reference, prediction=prediction)}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "score", "schema": SCHEMA}},
        temperature=0.0,   # 채점은 재현 가능해야 한다
        max_tokens=200,    # ★ 안전장치. 스키마만 믿으면 안 된다
    )
    return json.loads(r.choices[0].message.content)
""")

code("""
if not JUDGE_OK:
    print("서버가 없어 건너뜁니다.")
else:
    results = []
    for ex, p in zip(eval_set, predictions):
        s = judge(ex["question"], ex["reference"], p)
        results.append(s)
        print(f"Q: {ex['question']}")
        print(f"   응답: {p}")
        print(f"   정확성 {s['correctness']}  관련성 {s['relevance']}  자연스러움 {s['fluency']}")
        print(f"   사유: {s['reason']}")
        print()

    print("── 평균 ──")
    for k in ["correctness", "relevance", "fluency"]:
        print(f"  {k:<14}{mean(r[k] for r in results):.2f}")
""")

md("""
### 자동 지표가 놓쳤던 경우를 다시 채점해봅시다
""")

code("""
if not JUDGE_OK:
    print("서버가 없어 건너뜁니다.")
else:
    print(f"{'경우':<24}{'BLEU':>7}{'ROUGE-L':>9}{'정확성':>8}")
    print("-" * 50)
    for name, pred in cases:
        b = bleu.compute(predictions=[pred], references=[[reference]])["score"]
        r = rouge.compute(predictions=[pred], references=[reference])["rougeL"]
        s = judge(question, reference, pred)
        print(f"{name:<24}{b:>7.1f}{r:>9.3f}{s['correctness']:>8}")
""")

md("""
**틀린 답(부산)의 점수를 비교해보세요.**

BLEU·ROUGE 는 높게 줬지만 LLM judge 는 낮게 줬을 겁니다.
반대로 "뜻이 같고 표현만 다름" 은 자동 지표에서 낮았지만 judge 는 높게 줍니다.

이것이 LLM-as-judge 를 쓰는 이유입니다.

### 다만 만능은 아닙니다

- **비용과 시간**이 듭니다. 자동 지표는 즉시 계산되지만 이건 API 호출입니다
- **채점 모델이 틀릴 수 있습니다.** 특히 전문 분야에서
- **재현성**을 위해 `temperature=0` 을 써야 합니다
- 채점 모델이 **자기가 만든 답을 후하게 주는 편향**이 알려져 있습니다.
  평가 대상과 채점 모델을 다르게 하는 편이 안전합니다
""")

# ---------------------------------------------------------------------
md("""
## 6. 정리 — 무엇을 언제 쓰나

| 상황 | 권장 |
|---|---|
| 번역·요약처럼 정답 형태가 정해짐 | BLEU · ROUGE |
| 분류·추출처럼 정답이 하나 | 정확도 · F1 |
| 열린 생성 (챗봇, QA) | **LLM-as-judge** + 자동 지표 보조 |
| 최종 출시 판단 | **사람 평가** — 표본이라도 |

**실무에서는 섞어 씁니다.**
개발 중에는 자동 지표로 빠르게 회귀를 잡고,
릴리즈 전에 LLM judge 로 넓게 보고,
최종적으로 사람이 표본을 확인합니다.

## 마무리

- 평가는 **지표를 고르는 것이 아니라 태스크를 정의하는 것**에서 시작합니다
- **BLEU·ROUGE 는 표면 일치만 잽니다.** 틀린 답에 높은 점수를 줄 수 있습니다
- **LLM-as-judge** 는 의미를 보지만 비용과 편향이 있습니다
- 지표 하나로 판단하지 않습니다. **성공 기준이 여러 개면 지표도 여러 개**입니다

앞에서 SFT·DPO·GRPO 로 모델을 바꿔봤습니다.
이제 그 변화가 **실제로 나아진 것인지 확인할 수단**이 생겼습니다.
""")


# =====================================================================
def build() -> dict:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == CODE:
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), ensure_ascii=False, indent=1), encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == CODE)
    print(f"생성: {OUT}")
    print(f"  셀 {len(CELLS)}개 (코드 {n_code} · 마크다운 {len(CELLS) - n_code})")
    print()
    print("커리큘럼 4단원 ⑪ '태스크 정의와 평가' 대응 자료다.")
    print("기존에는 서술형 슬라이드 10장만 있고 실습이 0건이었다.")


if __name__ == "__main__":
    main()
