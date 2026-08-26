"""⑧-3 "프롬프트 자동 최적화" 실습 노트북을 만든다.

    uv run python tools/build_promptopt_notebook.py

왜 새로 만드는가
----------------
과정에 **"학습 전에 최선을 다하는 방법"** 이 없었다. 파인튜닝은 비싸고 오래 걸린다.
실무에서는 프롬프트를 먼저 끝까지 밀어붙이고, 그래도 부족할 때 학습한다.
지금 자료는 그 순서를 가르치지 않고 곧장 SFT 로 간다.

3막 구성이다.

    1막 번역   영문 상품 설명 → 한국어.  **데이터를 만드는 단계**
    2막 평가   LLM-as-judge 로 품질 채점 → 저품질 걸러내기
    3막 최적화 DSPy GEPA 로 프롬프트 자동 개선

설계에서 가장 중요한 결정
-------------------------
**번역 품질을 judge 로 채점해서 그 점수를 최적화하지 않는다.**

judge 와 task 가 같은 4B 모델이면 판정이 흔들린다. 노이즈가 큰 metric 위에서
옵티마이저는 **노이즈를 맞춘다** — 점수는 오르는데 번역은 그대로인, 강의에서 가장
곤란한 결과가 나온다. 게다가 번역 품질은 대부분 모델의 언어 능력에서 나와
프롬프트 개선 여지가 작다.

그래서 역할을 나눴다. judge 는 **필터링·평가 기법**으로 가르치고,
최적화 대상은 **구조화 추출**로 삼는다. metric 을 프로그램 검증으로 짜면 노이즈가 0이다.
그리고 "왜 judge 를 목적함수로 쓰면 안 되는가" 자체를 2막→3막 전환의 강의 포인트로 쓴다.

근거: CLAUDE.md §13 (DSPy 조사 결과)
"""

from __future__ import annotations

import json

from layout import work_path
from nbcommon import DATA_DIR_CODE, DATA_DIR_MD

OUT = work_path("HPC_프롬프트최적화실습.ipynb")   # 일자 배치는 tools/layout.py 가 정한다

MD, CODE = "markdown", "code"
CELLS: list[tuple[str, str]] = []


def md(t: str) -> None:
    CELLS.append((MD, t.strip("\n")))


def code(t: str) -> None:
    CELLS.append((CODE, t.strip("\n")))


# =====================================================================
md("""
# 프롬프트 자동 최적화

파인튜닝은 비쌉니다. GPU 시간만이 아니라 **데이터를 모으고, 평가 체계를 만들고,
한 번 하면 유지보수가 따라붙습니다.**

그래서 실무 순서는 이렇습니다.

```
프롬프트로 해결되는가?  →  예: 끝. 학습하지 않는다.
                          아니오: 그때 학습한다.
```

이 노트북은 **왼쪽 가지를 끝까지 밀어붙이는** 방법을 다룹니다.
그것도 손으로 고쳐가며 찍어 맞추는 것이 아니라 **자동으로** 합니다.

## 무엇을 하나

| 막 | 하는 일 |
|---|---|
| **1막** | 영문 상품 설명을 **한국어로 번역**해 데이터셋을 만든다 |
| **2막** | **LLM-as-judge** 로 번역 품질을 채점하고 나쁜 것을 걸러낸다 |
| **3막** | **DSPy GEPA** 로 프롬프트를 자동으로 개선한다 |

1막에서 만든 한국어 데이터가 3막의 재료가 되고,
3막에서 최적화한 프롬프트로 만든 데이터가 **내일 SFT 의 학습 데이터**가 됩니다.
""")

md("""
## 준비

이 노트북은 **vLLM 서버가 떠 있어야** 돌아갑니다. 학습은 하지 않습니다 —
GPU 는 추론에만 씁니다.

```bash
source /opt/vllm-env/bin/activate
nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \\
    --gpu-memory-utilization 0.80 --max-model-len 16384 > /tmp/vllm.log 2>&1 &
```
""")

code("""
%pip install -q -U datasets openai
%pip install -q -U "dspy[optuna]"
""")

md(DATA_DIR_MD)
code(DATA_DIR_CODE)

code("""
import json
from openai import OpenAI

BASE = "http://localhost:8000/v1"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
client = OpenAI(api_key="EMPTY", base_url=BASE, timeout=120.0)

# 서버가 없으면 이 아래가 전부 깨진다. 먼저 확인하고 안내한다.
try:
    client.models.list()
    SERVER_OK = True
    print(f"vLLM 서버 연결됨 — {MODEL}")
except Exception as e:
    SERVER_OK = False
    print(f"vLLM 서버에 붙지 못했습니다: {type(e).__name__}")
    print()
    print("  별도 터미널에서 서버를 띄우고 커널을 재시작하세요:")
    print("    source /opt/vllm-env/bin/activate")
    print("    nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \\\\")
    print("        --gpu-memory-utilization 0.80 --max-model-len 16384 > /tmp/vllm.log 2>&1 &")
""")

# =====================================================================
# 1막 — 번역
# =====================================================================
md("""
---

# 1막. 한국어 데이터 만들기

한국어 LLM 을 만들려는데 **한국어 데이터가 없는** 상황은 흔합니다.
영어 데이터는 많으니, 그걸 번역해서 쓰는 것이 현실적인 출발점입니다.

여기서는 영문 상품 설명을 한국어로 옮깁니다.
""")

code("""
from datasets import load_dataset

N_TRANSLATE = 20          # 실측 기준 이 분량이면 번역이 2~3분이다
MAX_CHARS   = 800         # 번역 시간은 길이에 비례한다. 원문을 잘라 쓴다

raw = load_dataset("Taekyoon/test_amazon", split=f"train[:{N_TRANSLATE}]")
print(raw)
print()
print("── 원문 예시 ──")
print(raw[0]["text"][:400])
""")

md("""
### 원본을 먼저 눈으로 보세요

위 출력에 이상한 것이 보입니다.

```
- Brand: F, a, t,  , S, h, a, r, k
```

`FatShark` 라는 브랜드명이 **글자 단위로 쪼개져** 있습니다. 데이터를 만들 때
문자열을 문자 리스트로 잘못 다룬 흔적입니다. 실무 데이터에서 흔한 종류의 결함입니다.

**이걸 못 보고 넘어가면** 번역기가 쪼개진 상태를 그대로 옮기고, 그 데이터로 학습한
모델은 "브랜드명은 글자를 쉼표로 나눠 쓴다"고 배웁니다. 아무도 의도하지 않은 결과입니다.

얼마나 있는지 세어봅시다.
""")

code(r"""
import re

# 'F, a, t' 처럼 한 글자 + 쉼표가 세 번 이상 이어지는 패턴
split_name = re.compile(r"(?:[A-Za-z],\s+){2,}[A-Za-z]")

hits = [i for i, r in enumerate(raw) if split_name.search(r["text"])]
print(f"글자가 쉼표로 쪼개진 흔적: {len(hits)}/{len(raw)}건")
for i in hits[:3]:
    m = split_name.search(raw[i]["text"])
    print(f"  [{i}] {m.group()!r}")
""")

md("""
### 번역 프롬프트에서 놓치기 쉬운 것

그냥 "번역하세요" 라고 하면 **브랜드명과 모델명까지 한글로 옮깁니다.**
`Anker PowerCore` 가 "앵커 파워코어" 가 되면 검색도 안 되고 상품 식별도 안 됩니다.

고유명사는 **원문 그대로 두라고 명시**해야 합니다.
그리고 출력이 JSON 이어야 뒤에서 프로그램으로 다룰 수 있습니다.
""")

code("""
TRANSLATE_PROMPT = '''다음 영문 상품 설명을 한국어로 번역하세요.

지켜야 할 것:
1. 브랜드명, 모델명, 제품명 같은 고유명사는 **원문 그대로** 둡니다.
   (예: Anker PowerCore 10000 -> Anker PowerCore 10000)
2. 단위와 숫자는 그대로 둡니다. (10000mAh, 13-inch)
3. 자연스러운 한국어 문장으로 옮깁니다. 직역투를 피하세요.
4. 원문에 없는 내용을 만들어내지 마세요.
5. 'F, a, t, S, h, a, r, k' 처럼 **글자가 쉼표로 쪼개진 고유명사**가 있으면
   'FatShark' 로 붙여서 복원하세요. 원본 데이터의 결함입니다.

JSON 으로만 답하세요.

[영문 원문]
{text}'''

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        # maxLength 를 걸지 않으면 모델이 끝없이 쓴다. 실제로 한 요청이 90초를 넘긴 적이 있다.
        "translation": {"type": "string", "maxLength": 3000},
    },
    "required": ["translation"],
}


def translate(text):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user",
                   "content": TRANSLATE_PROMPT.format(text=text[:MAX_CHARS])}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "translation", "schema": TRANSLATE_SCHEMA}},
        temperature=0.2,
        max_tokens=1500,   # ★ 안전장치. 스키마의 maxLength 만 믿으면 안 된다
    )
    return json.loads(r.choices[0].message.content)["translation"]
""")

code("""
import time

if not SERVER_OK:
    print("서버가 없어 건너뜁니다.")
    ko_data = []
else:
    ko_data, skipped = [], 0
    t0 = time.time()
    for i, row in enumerate(raw):
        try:
            ko = translate(row["text"])
            ko_data.append({"source": row["text"][:MAX_CHARS], "translation": ko})
        except Exception as e:
            skipped += 1
            print(f"  [{i}] 제외 — {type(e).__name__}")
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(raw)}  ({time.time()-t0:.0f}초)")

    print()
    print(f"번역 완료 {len(ko_data)}건 · 제외 {skipped}건 · {time.time()-t0:.0f}초")
""")

code("""
# ★ 완주했다고 잘 된 것이 아니다. 실제 결과를 눈으로 본다.
for d in ko_data[:2]:
    print("=" * 72)
    print("[영문]  ", d["source"][:220].replace("\\n", " "))
    print()
    print("[한국어]", d["translation"][:220].replace("\\n", " "))
    print()

# 고유명사가 살아남았는지 기계적으로도 확인한다
import re

def ascii_words(s):
    return set(re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", s))

kept = [len(ascii_words(d["source"]) & ascii_words(d["translation"])) for d in ko_data]
if kept:
    print(f"원문의 영문 토큰이 번역문에 남은 개수 — 중앙값 {sorted(kept)[len(kept)//2]}개")
    print("  0에 가까우면 브랜드명까지 전부 한글로 옮긴 것이다. 프롬프트를 고쳐야 한다.")
""")

# =====================================================================
# 2막 — judge
# =====================================================================
md("""
---

# 2막. 품질을 어떻게 재나

번역을 20건 만들었습니다. **이걸 그대로 학습에 쓰면 될까요?**

몇 건은 분명 엉망일 겁니다. 그런데 20건은 사람이 다 읽을 수 있어도
2만 건은 못 읽습니다. 자동으로 걸러야 합니다.

BLEU 같은 자동 지표는 **정답 번역문이 있어야** 씁니다. 우리에겐 없습니다.
그래서 **LLM 에게 채점을 시킵니다.** 이것이 LLM-as-judge 입니다.
""")

code("""
JUDGE_PROMPT = '''당신은 영한 번역 품질을 채점합니다.

[영문 원문]
{source}

[한국어 번역]
{translation}

다음 세 항목을 각각 1~5점으로 채점하세요.

- accuracy: 원문의 내용이 빠짐없이, 왜곡 없이 옮겨졌는가
- fluency: 한국어가 자연스러운가. 직역투는 아닌가
- terms: 브랜드명·모델명·단위 같은 고유명사가 원문 그대로 보존됐는가

reason 은 **한 문장으로 짧게** 쓰세요.
JSON 으로만 답하세요.'''

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
        "fluency":  {"type": "integer", "minimum": 1, "maximum": 5},
        "terms":    {"type": "integer", "minimum": 1, "maximum": 5},
        "reason":   {"type": "string", "maxLength": 150},
    },
    "required": ["accuracy", "fluency", "terms", "reason"],
}


def judge(source, translation, temperature=0.0):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            source=source[:MAX_CHARS], translation=translation)}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "score", "schema": JUDGE_SCHEMA}},
        temperature=temperature,
        max_tokens=200,    # ★ 안전장치
    )
    return json.loads(r.choices[0].message.content)
""")

code("""
if not ko_data:
    print("번역 결과가 없어 건너뜁니다.")
    scored = []
else:
    scored = []
    t0 = time.time()
    for i, d in enumerate(ko_data):
        try:
            s = judge(d["source"], d["translation"])
            scored.append({**d, **s,
                           "mean": (s["accuracy"] + s["fluency"] + s["terms"]) / 3})
        except Exception as e:
            print(f"  [{i}] 채점 제외 — {type(e).__name__}")
    print(f"채점 {len(scored)}건 · {time.time()-t0:.0f}초")
""")

code("""
if scored:
    from collections import Counter

    print("── 항목별 평균 ──")
    for k in ["accuracy", "fluency", "terms"]:
        print(f"  {k:<10} {sum(s[k] for s in scored)/len(scored):.2f}")

    print()
    print("── 종합 점수 분포 ──")
    hist = Counter(round(s["mean"]) for s in scored)
    for v in sorted(hist):
        print(f"  {v}점 {'#' * hist[v]} ({hist[v]}건)")

    print()
    print("── 가장 낮게 받은 것 ──")
    worst = min(scored, key=lambda s: s["mean"])
    print(f"  {worst['mean']:.1f}점 — {worst['reason']}")
    print(f"  {worst['translation'][:200]}")
""")

md("""
### 잠깐 — 이 점수를 믿어도 되나

채점한 모델과 번역한 모델이 **같은 4B 모델**입니다. 자기가 만든 것을 자기가 채점합니다.

점수가 얼마나 흔들리는지 직접 재봅시다. **같은 번역을 세 번 채점**해서
매번 같은 점수가 나오는지 보는 겁니다.
""")

code("""
if scored:
    sample = scored[0]

    # temperature 0.0 으로도 흔들리는지를 먼저 본다
    runs = [judge(sample["source"], sample["translation"])["accuracy"] for _ in range(3)]
    print(f"같은 번역을 3번 채점 (temperature=0.0): accuracy = {runs}")

    runs_t = [judge(sample["source"], sample["translation"], temperature=0.7)["accuracy"]
              for _ in range(3)]
    print(f"temperature=0.7 로 3번:                accuracy = {runs_t}")
    print()
    if len(set(runs)) > 1 or len(set(runs_t)) > 1:
        print("→ 같은 입력인데 점수가 달라집니다. judge 에는 노이즈가 있습니다.")
    else:
        print("→ 이번엔 일치했습니다. 다른 샘플에서는 갈릴 수 있습니다.")
""")

md("""
**이 점을 기억해 두세요.** 3막에서 다시 나옵니다.

judge 점수는 **거르는 데 쓰기에는 충분**합니다. 1점과 5점은 확실히 구분되니까요.
하지만 **최적화의 목적함수로 쓰기에는 위험**합니다. 4.1 과 4.3 의 차이가
진짜 품질 차이인지 노이즈인지 알 수 없기 때문입니다.
""")

code("""
THRESHOLD = 3.5

if scored:
    good = [s for s in scored if s["mean"] >= THRESHOLD]
    print(f"{len(scored)}건 → {len(good)}건  "
          f"(기준 {THRESHOLD}점 미만 {len(scored)-len(good)}건 제외)")

    out = DATA_DIR / "amazon_ko.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in good:
            # ensure_ascii=False 가 없으면 한글이 escape 되어 눈으로 확인할 수 없다
            f.write(json.dumps({"source": s["source"],
                                "translation": s["translation"],
                                "score": s["mean"]}, ensure_ascii=False) + "\\n")
    print(f"저장: {out}  ({out.stat().st_size/1024:.0f} KB)")
""")

# =====================================================================
# 3막 — GEPA
# =====================================================================
md("""
---

# 3막. 프롬프트를 자동으로 고친다

여기까지는 **사람이 프롬프트를 썼습니다.** 1막의 번역 프롬프트를 떠올려 보세요.
"고유명사는 그대로 두세요" 를 넣은 것은, 안 넣었을 때 문제가 생기는 걸
**알고 있었기 때문**입니다.

모르는 문제는 어떻게 할까요. 그리고 프롬프트를 20번 고쳐가며 점수를 비교하는 일을
사람이 계속할 수 있을까요.

**자동화합니다.** 프롬프트를 파라미터로 보고, 점수를 목적함수로 삼아 탐색합니다.
""")

md("""
### 무엇을 최적화 대상으로 삼을까

번역을 대상으로 삼고 싶어집니다. 방금 judge 로 채점까지 했으니까요.
**그런데 그러면 안 됩니다.**

방금 봤듯이 judge 점수에는 노이즈가 있습니다. **노이즈가 있는 점수를 목적함수로 삼으면
옵티마이저는 노이즈를 맞춥니다.** 점수는 오르는데 번역은 그대로인 결과가 나옵니다.
실무에서 사람들이 실제로 저지르는 실수입니다.

그래서 대상을 바꿉니다. **구조화 추출** — 방금 만든 한국어 상품 설명에서
정해진 형식으로 정보를 뽑는 작업입니다. 이건 **프로그램으로 채점할 수 있습니다.**

| | 번역 | 구조화 추출 |
|---|---|---|
| 채점 방법 | LLM judge | **프로그램 검증** |
| 노이즈 | 있음 | **0** |
| 개선 여지 | 작음 (모델 언어능력이 좌우) | **큼** (형식 준수는 지시에 달림) |

> 규칙: **최적화의 목적함수는 최대한 노이즈가 없어야 합니다.**
> 프로그램으로 검증되는 것이 있으면 그것을 씁니다.
""")

code("""
import dspy

# 접두사는 openai/ 다. hosted_vllm/ 가 아니다.
lm = dspy.LM(
    f"openai/{MODEL}",
    api_base=BASE, api_key="EMPTY",
    model_type="chat",
    temperature=0.0,
    max_tokens=1024,   # ★ 구조화 출력에는 반드시 상한을 건다
    cache=False,       # ★ 기본값 True. 재실행이 LM 을 안 치고 캐시를 돌려준다
)
# 4B 급은 ChatAdapter 의 [[ ## field ## ]] 마커를 자주 어긴다. 처음부터 JSON 으로 고정한다.
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())

print("dspy", getattr(dspy, "__version__", "?"), "· LM 설정 완료")
print(lm("한국의 수도는? 한 단어로만.")[0][:80])
""")

md("""
### 무엇을 고정하고 무엇을 최적화할까

**출력 형식은 고정하고, 지시문만 최적화합니다.**

`category`, `features`, `target_users` — 어떤 항목을 낼지는 우리가 정합니다.
이건 요구사항이지 최적화 대상이 아닙니다. 옵티마이저가 이걸 알아맞히게 하면
"무엇을 원하는지도 안 알려주고 맞혀보라"는 셈이 됩니다.

최적화하는 것은 **지시문** — `"정보를 추출한다."` 한 줄입니다.
같은 형식을 요구해도 *어떻게 채우라고 말하느냐*에 따라 결과가 크게 달라집니다.

시작 문장을 일부러 부실하게 둡니다. 잘 쓴 프롬프트에서 출발하면 개선 여지가 없어
전후 차이가 안 보이고, **사람들이 처음 쓰는 프롬프트가 실제로 이렇습니다.**
""")

code("""
class ExtractProduct(dspy.Signature):
    '''정보를 추출한다.'''          # ← 시작 프롬프트. GEPA 가 이 문장을 고쳐 나간다

    text: str = dspy.InputField()
    # 출력 **형식**은 여기서 못박는다. 최적화 대상이 아니다.
    category: str = dspy.OutputField()
    features: list[str] = dspy.OutputField()
    target_users: list[str] = dspy.OutputField()

program = dspy.Predict(ExtractProduct)

START_INSTRUCTIONS = ExtractProduct.instructions   # 나중에 전후 비교용
print("시작 프롬프트:")
print(" ", START_INSTRUCTIONS)
""")

md("""
### metric — 판정에 LLM 을 끼우지 않는다

무엇을 만족해야 "잘한 것" 인지 코드로 적습니다.
**전부 프로그램이 판정하므로 노이즈가 0입니다.**

| 보는 것 | 왜 |
|---|---|
| `features` 3개 이상 | 하나만 뽑고 마는 것을 막는다 |
| `target_users` 2개 이상 | 같은 이유 |
| 항목이 50자 이하 | 문장을 통째로 복사하는 것을 막는다 |
| 항목의 단어가 **원문에 있는가** | 지어낸 내용을 막는다 |
| 한국어인가 | 입력이 한국어인데 영어로 답하는 경우가 있다 |
| `category` 가 25자 이하 | 분류명이지 설명이 아니다 |

전부 **입력과 출력만 보고 코드로 판정**됩니다. 채점하는 LLM 이 없습니다.

그리고 GEPA 는 점수만 받지 않습니다. **`feedback` 문자열**을 함께 받아서
"무엇이 틀렸는지" 를 reflection 모델에게 알려줍니다. 이게 GEPA 의 핵심 채널이라
숫자만 주면 성능이 크게 떨어집니다.
""")

code("""
# 앞 셀에서 이미 import 했지만, 3막만 따로 돌려보는 경우를 위해 다시 적는다.
import json
import re

MIN_FEATURES = 3
MIN_USERS    = 2
HANGUL = re.compile(r"[가-힣]")


def _tokens(x):
    return re.findall(r"[A-Za-z0-9]{2,}|[가-힣]{2,}", str(x))


def extract_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    '''프로그램 검증 metric. 판정에 LLM 이 끼지 않으므로 노이즈가 0이다.

    GEPA 는 5인자 시그니처를 쓰고 Prediction(score=, feedback=) 을 받는다.
    feedback 문자열이 reflection 모델에 전달되는 **핵심 채널**이라 구체적으로 쓴다.
    '''
    src   = gold.text
    cat   = str(getattr(pred, "category", "") or "").strip()
    feats = getattr(pred, "features", None) or []
    users = getattr(pred, "target_users", None) or []
    feats = [str(x).strip() for x in feats] if isinstance(feats, list) else []
    users = [str(x).strip() for x in users] if isinstance(users, list) else []

    problems = []

    if not cat:
        problems.append("category 가 비었다")
    elif len(cat) > 25:
        problems.append(f"category 가 {len(cat)}자로 길다. 한두 단어의 분류명이어야 한다")

    if len(feats) < MIN_FEATURES:
        problems.append(f"features 가 {len(feats)}개뿐이다. {MIN_FEATURES}개 이상 뽑아야 한다")
    if len(users) < MIN_USERS:
        problems.append(f"target_users 가 {len(users)}개뿐이다. {MIN_USERS}개 이상이어야 한다")

    long_feats = [f for f in feats if len(f) > 50]
    if long_feats:
        problems.append(f"features 항목이 너무 길다(예: {long_feats[0][:30]}...). "
                        "문장이 아니라 짧은 구로 써야 한다")

    # 원문에 없는 내용을 지어냈는지 본다. 토큰이 하나도 원문에 없으면 근거가 없는 것이다.
    ungrounded = [f for f in feats if not any(t in src for t in _tokens(f))]
    if feats and len(ungrounded) > len(feats) // 2:
        problems.append(f"features 상당수가 원문에 없는 내용이다(예: {ungrounded[0][:30]}). "
                        "원문에 실제로 있는 것만 뽑아야 한다")

    body = " ".join([cat] + feats + users)
    if body and len(HANGUL.findall(body)) / max(len(body), 1) < 0.15:
        problems.append("값이 한국어가 아니다. 한국어로 작성해야 한다")

    score = max(0.0, 1.0 - 0.2 * len(problems))
    return dspy.Prediction(score=score, feedback="; ".join(problems) or "정상")
""")

code("""
# 1막에서 만든 한국어 데이터를 그대로 쓴다.
# gold 라벨이 없어도 된다 — metric 이 프로그램 검증이라 정답이 필요 없다.
pool = ([dspy.Example(text=s["translation"]).with_inputs("text") for s in scored]
        if scored else [])

# ★ 고정 인덱스로 자르면 안 된다. 2막에서 몇 건이 걸러질지 모르기 때문이다.
#   실제로 pool 이 19건일 때 pool[20:30] 이 빈 리스트가 되어 valset 이 0건이었고,
#   전후 비교가 통째로 건너뛰어졌다. 비율로 나눈다.
n_val = max(3, round(len(pool) * 0.3))
trainset, valset = pool[:-n_val], pool[-n_val:]

print(f"pool {len(pool)}건 → trainset {len(trainset)}건 · valset {len(valset)}건")
if len(trainset) < 5 or len(valset) < 3:
    print("★ 1막·2막 결과가 부족합니다. N_TRANSLATE 를 늘리거나 THRESHOLD 를 낮추세요.")
""")

code("""
def evaluate(prog, examples):
    '''valset 평균 점수. 최적화 전후 비교용.'''
    if not examples:
        return 0.0
    total = 0.0
    for ex in examples:
        try:
            total += extract_metric(ex, prog(text=ex.text)).score
        except Exception:
            pass          # 호출 자체가 실패하면 0점으로 친다
    return total / len(examples)


before_score = 0.0
if valset:
    before_score = evaluate(program, valset)
    print(f"최적화 전 val 점수: {before_score:.3f}")
    print()
    print("── 지금 뭘 내놓는지 ──")
    r0 = program(text=valset[0].text)
    print("  category    :", r0.category)
    print("  features    :", r0.features)
    print("  target_users:", r0.target_users)
    print()
    print("  채점 결과:", extract_metric(valset[0], r0).feedback)
""")

md("""
### GEPA 를 돌린다

**호출 횟수에 하드캡을 겁니다.** `auto="light"` 도 metric 호출이 420회라
7~14분이 걸립니다. 강의 시간에는 맞지 않습니다.

> `auto` / `max_full_evals` / `max_metric_calls` 는 **정확히 하나만** 지정해야 합니다.
> 둘 이상 주면 생성자에서 에러가 납니다.
""")

code("""
MAX_CALLS = 120        # 실측 60회 52.5초 → 120회 약 1.7분 (verify/07)

if len(trainset) >= 5:
    t0 = time.time()
    # ★ reflection 은 프롬프트를 **통째로 새로 쓰는** 일이라 task 보다 출력이 훨씬 길다.
    #   task LM(max_tokens=1024)을 그대로 넘겼더니 제안이 중간에서 잘렸다:
    #     "LM response was truncated due to exceeding max_tokens=512"
    #   잘린 제안은 통째로 버려지므로 최적화가 헛돈다. 별도 LM 으로 분리한다.
    reflection_lm = dspy.LM(
        f"openai/{MODEL}", api_base=BASE, api_key="EMPTY",
        model_type="chat",
        temperature=1.0,   # 공식 권장. 다양한 제안이 나와야 한다
        max_tokens=4096,   # ★ 여기가 핵심
        cache=False,
    )

    optimizer = dspy.GEPA(
        metric=extract_metric,
        max_metric_calls=MAX_CALLS,
        reflection_lm=reflection_lm,
        num_threads=8,
        track_stats=True,
    )
    optimized = optimizer.compile(program, trainset=trainset, valset=valset)
    print(f"\\n최적화 완료 — {time.time()-t0:.0f}초")
else:
    optimized = program
    print("데이터가 부족해 건너뜁니다.")
""")

code("""
if not valset:
    print("★ valset 이 비어 비교할 수 없습니다. 위 분할 셀을 확인하세요.")
else:
    after_score = evaluate(optimized, valset)

    # ★ 결론을 먼저 찍는다. 프롬프트 원문이 길어서 뒤에 두면 잘려 안 보인다.
    print("=" * 72)
    print(f"val 점수  {before_score:.3f}  →  {after_score:.3f}   "
          f"({after_score-before_score:+.3f})")
    print(f"학습 0회 · LLM 호출 {MAX_CALLS}회로 얻은 차이입니다.")
    print("=" * 72)
    print()
    print("── 최적화 후 출력 ──")
    r1 = optimized(text=valset[0].text)
    print("  category    :", r1.category)
    print("  features    :", r1.features)
    print("  target_users:", r1.target_users)
    print()
    print("  채점 결과:", extract_metric(valset[0], r1).feedback)
    print()
    print("=" * 72)
    print("프롬프트가 어떻게 바뀌었나")
    print("=" * 72)
    print("[전]")
    print(" ", START_INSTRUCTIONS)
    print()
    print("[후]")
    for name, p in optimized.named_predictors():
        print(" ", p.signature.instructions)
""")

md("""
### 점수가 안 올랐다면

그럴 수 있습니다. 이 실습에는 알려진 한계가 있습니다.

**task 모델과 reflection 모델이 같은 4B 입니다.** DSPy 공식 권장은
"작은 모델을 최적화할 때는 **더 큰 모델을 reflection_lm 으로** 쓰라" 입니다.
프롬프트를 고쳐 쓰는 쪽이 고쳐질 쪽보다 똑똑해야 개선이 나오는데,
지금은 같은 모델이 자기 프롬프트를 고치고 있습니다.

실제로 처음 만들 때 이것 때문에 한 번 실패했습니다. reflection LM 의 `max_tokens` 를
task 와 같은 값으로 뒀더니 **제안이 중간에서 잘려** 통째로 버려졌습니다
(`LM response was truncated`). 프롬프트를 새로 쓰는 일은 출력이 훨씬 길다는 것을
놓친 겁니다. 지금은 별도 LM 으로 분리해 `max_tokens=4096` 을 줍니다.

GPU 한 장에 4B 하나만 띄울 수 있어서 이렇게 했습니다.
실무에서는 **task 는 작은 모델, reflection 은 큰 모델** 로 나눕니다.
reflection 호출은 수십 번뿐이라 비싼 모델을 써도 부담이 적습니다.
""")

md("""
---

## 마무리

**학습을 한 번도 하지 않고** 여기까지 왔습니다.

- **번역**으로 없던 한국어 데이터를 만들었습니다
- **LLM-as-judge** 로 품질을 재고 나쁜 것을 걸렀습니다 —
  그리고 그 점수에 **노이즈가 있다**는 것도 직접 확인했습니다
- **프롬프트를 자동으로 개선**했습니다. 목적함수는 노이즈가 없는 것으로 골랐습니다

여기서 얻은 두 가지가 다음으로 이어집니다.

| 만든 것 | 어디로 |
|---|---|
| `amazon_ko.jsonl` | 내일 **SFT 학습 데이터** |
| "프롬프트만으로 얻은 점수" | 학습 후 성능과 비교할 **기준선** |

내일 같은 작업을 **학습으로** 해봅니다. 그리고 물어봅니다 —
**학습은 프롬프트 위에 얼마를 더 얹어주는가?**
그 차이가 데이터를 모으고 GPU 를 돌릴 값어치가 있는지가 판단 기준입니다.
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
            "accelerator": "GPU",
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
    print("커리큘럼 3단원 ⑧ 에 '학습 전에 최선을 다하는 방법' 을 넣는 자료다.")
    print("기존 자료에는 이 항목이 아예 없었다.")


if __name__ == "__main__":
    main()
