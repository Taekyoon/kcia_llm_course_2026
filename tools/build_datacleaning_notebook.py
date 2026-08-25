"""⑧ "오픈소스를 활용한 Pre-training 데이터 처리" 실습 노트북을 만든다.

    uv run python tools/build_datacleaning_notebook.py

왜 새로 만드는가
----------------
커리큘럼 3단원(LLM Pre-training, 7H)의 세부항목 중 하나인
**"오픈소스를 활용한 Pre-training 데이터 처리"** 에 대응하는 자료가
슬라이드 264장·노트북 9종을 통틀어 **0장**이었다 (review/01_대조표.md ⑧ 미충족).

과정명이 "LLM **데이터 처리** 및 파인튜닝 방법" 인데 데이터 처리가 없었다.
3일차의 "vLLM 을 활용한 데이터처리 실습"(36장)은 이름만 비슷하고 실제로는
Post-training 용 합성 데이터 생성이라 이 항목을 대체하지 못한다.

설계 방침
---------
- 데이터는 `wikimedia/wikipedia` `20231101.ko` 를 쓴다. 3일차 RAG 실습이 이미
  같은 데이터를 쓰므로 과정 안에서 연결된다.
- MinHash 를 **직접 구현**한다. 라이브러리를 부르면 한 줄로 끝나지만
  "근사 중복을 어떻게 찾는가" 가 보이지 않는다. 원리를 본 뒤 실무 도구를 소개한다.
- 필터를 적용할 때마다 **무엇이 걸러졌는지 실제로 보여준다.** 숫자만 보면
  필터가 제대로 도는지 알 수 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "notebook" / "3일차" / "HPC_데이터처리실습.ipynb"

MD, CODE = "markdown", "code"
CELLS: list[tuple[str, str]] = []


def md(t: str) -> None:
    CELLS.append((MD, t.strip("\n")))


def code(t: str) -> None:
    CELLS.append((CODE, t.strip("\n")))


# =====================================================================
md("""
# Pre-training 데이터 처리

LLM 을 학습시키기 전에 **데이터를 정제하는** 과정을 다룹니다.

모델 구조와 학습 코드는 공개돼 있습니다. 같은 구조로 학습해도 성능이 갈리는
가장 큰 이유는 **데이터**입니다. 그래서 실무에서 시간이 가장 많이 들어가는 곳도 여기입니다.

## 무엇을 하게 되나

1. **원본 코퍼스의 상태를 확인**합니다 — 실제로 얼마나 지저분한지
2. **정확 중복**을 제거합니다 — 완전히 같은 문서
3. **근사 중복**을 제거합니다 — 조금씩 다른 문서. MinHash 를 직접 구현합니다
4. **품질 필터**를 적용합니다 — 너무 짧거나, 반복이 심하거나, 한국어가 아닌 것
5. **실무 도구**를 살펴봅니다 — datatrove, dolma

## 왜 중요한가

- **중복**은 모델이 특정 문장을 통째로 외우게 만듭니다. 학습 비용도 그만큼 낭비됩니다.
- **저품질 문서**는 모델이 이상한 패턴을 배우게 합니다.
- 학습에 쓰는 GPU 시간은 비쌉니다. **쓰레기를 학습시킬 여유가 없습니다.**
""")

code("""
%pip install -q -U datasets
""")

code("""
import hashlib
import random
import re
import unicodedata
from collections import Counter, defaultdict

from datasets import load_dataset

random.seed(42)
""")

# ---------------------------------------------------------------------
md("""
## 1. 원본 코퍼스 확인

한국어 위키백과를 씁니다. 3일차 RAG 실습에서 쓰는 것과 같은 데이터입니다.

실습 시간을 고려해 일부만 가져옵니다. 실제 Pre-training 은 이보다
**수천 배 큰** 코퍼스를 다룹니다. 처리 원리는 같습니다.
""")

code("""
N_DOCS = 20000

raw = load_dataset("wikimedia/wikipedia", "20231101.ko", split=f"train[:{N_DOCS}]")
docs = [{"id": r["id"], "title": r["title"], "text": r["text"]} for r in raw]

print(f"문서 {len(docs):,}개")
print(f"총 글자 수 {sum(len(d['text']) for d in docs):,}")
""")

code("""
# 길이 분포부터 본다. 평균만 보면 실태가 안 보인다.
lens = sorted(len(d["text"]) for d in docs)
def pct(p):
    return lens[int(len(lens) * p / 100)]

print(f"  최소   {lens[0]:>8,}자")
print(f"  25%    {pct(25):>8,}자")
print(f"  중앙값 {pct(50):>8,}자")
print(f"  75%    {pct(75):>8,}자")
print(f"  최대   {lens[-1]:>8,}자")
print()
print(f"  100자 미만: {sum(1 for l in lens if l < 100):,}개")
print(f"  50자  미만: {sum(1 for l in lens if l < 50):,}개")
""")

code("""
# 가장 짧은 문서들을 실제로 본다. 숫자보다 이게 빠르다.
print("── 짧은 문서 5개 ──")
for d in sorted(docs, key=lambda x: len(x["text"]))[:5]:
    print(f"[{d['title']}] ({len(d['text'])}자)")
    print(f"  {d['text'][:120]!r}")
    print()
""")

# ---------------------------------------------------------------------
md("""
## 2. 정확 중복 제거

가장 쉬운 것부터 합니다. **완전히 같은 문서**를 걷어냅니다.

문서 전체를 해시로 바꿔서 비교합니다. 문자열끼리 직접 비교하면
문서 수의 제곱만큼 비교해야 하지만, 해시를 쓰면 한 번씩만 훑으면 됩니다.

먼저 **정규화**를 합니다. 눈에는 같아 보여도 유니코드 표현이 다르면
다른 문자열로 취급되기 때문입니다. 한국어는 특히 조합형/완성형 문제가 있습니다.
""")

code("""
def normalize(text: str) -> str:
    # NFKC: 유니코드 정규화. 한글 자모 조합형을 완성형으로 통일한다.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\\s+", " ", text)   # 연속 공백을 하나로
    return text.strip()


def doc_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


seen, exact_dedup, dup_examples = set(), [], []
for d in docs:
    h = doc_hash(d["text"])
    if h in seen:
        dup_examples.append(d)
        continue
    seen.add(h)
    exact_dedup.append(d)

print(f"{len(docs):,} → {len(exact_dedup):,}  (정확 중복 {len(docs)-len(exact_dedup):,}개 제거)")
if dup_examples:
    print("\\n── 제거된 예시 ──")
    for d in dup_examples[:3]:
        print(f"  [{d['title']}] {d['text'][:80]!r}")
""")

# ---------------------------------------------------------------------
md("""
## 3. 근사 중복 제거 — MinHash

정확 중복은 사실 별로 없습니다. 진짜 문제는 **거의 같은 문서**입니다.
글자 하나만 달라도 해시는 완전히 달라지니까요.

웹 코퍼스에는 이런 것이 많습니다. 같은 기사의 재게시, 템플릿이 같은 페이지,
문단 하나만 다른 문서. 이걸 잡아야 합니다.

### 어떻게 하나

두 문서가 얼마나 겹치는지는 **자카드 유사도**로 잽니다.

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

문서를 **shingle**(연속된 n글자 조각) 집합으로 만든 뒤 겹침을 보는 겁니다.
그런데 문서 수가 많으면 모든 쌍을 비교할 수 없습니다. 2만 개면 2억 번입니다.

**MinHash** 는 이 문제를 이렇게 풉니다.

1. shingle 집합을 여러 해시 함수로 돌려서 **각 함수의 최솟값**만 남깁니다 → 시그니처
2. 두 집합의 자카드 유사도는 **시그니처가 일치하는 비율**과 통계적으로 같습니다
3. 긴 문서도 고정 길이(예: 64개) 숫자로 압축됩니다

여기에 **LSH(밴딩)** 를 더하면 비교 자체를 줄일 수 있습니다.
시그니처를 여러 밴드로 쪼개서, **한 밴드라도 완전히 같은 문서끼리만** 후보로 봅니다.
""")

code("""
SHINGLE = 5     # 5글자씩 끊는다. 한국어는 글자당 정보량이 커서 영어보다 짧게 잡는다
NUM_HASH = 64   # 시그니처 길이
BANDS = 16      # 밴드 수 (밴드당 64/16 = 4개)

MOD = (1 << 61) - 1   # 큰 소수
HASHES = [(random.randrange(1, MOD), random.randrange(0, MOD)) for _ in range(NUM_HASH)]


def shingles(text: str) -> set[int]:
    t = normalize(text)
    if len(t) < SHINGLE:
        return set()
    # 조각을 그대로 들고 있으면 메모리를 많이 쓴다. 해시로 바꿔 저장한다.
    return {hash(t[i:i + SHINGLE]) & 0xFFFFFFFF for i in range(len(t) - SHINGLE + 1)}


def minhash(sh: set[int]) -> tuple[int, ...]:
    if not sh:
        return tuple([0] * NUM_HASH)
    # 각 해시 함수마다 최솟값 하나씩
    return tuple(min((a * s + b) % MOD for s in sh) for a, b in HASHES)
""")

code("""
# 원리 확인: 비슷한 문서와 다른 문서를 넣어보고 추정치가 맞는지 본다.
A = "고양이는 포유류에 속하는 동물이다. 집에서 기르는 경우가 많다."
B = "고양이는 포유류에 속하는 동물이다. 집에서 기르는 일이 많다."   # 거의 같음
C = "맥스웰 방정식은 전자기 현상을 기술하는 네 개의 편미분 방정식이다."  # 다름

def jaccard(x, y):
    return len(x & y) / len(x | y) if (x | y) else 0.0

def sig_sim(x, y):
    return sum(1 for i, j in zip(x, y) if i == j) / NUM_HASH

for name, other in [("B(거의 같음)", B), ("C(다름)", C)]:
    sa, so = shingles(A), shingles(other)
    print(f"A vs {name}")
    print(f"  실제 자카드      {jaccard(sa, so):.3f}")
    print(f"  MinHash 추정치   {sig_sim(minhash(sa), minhash(so)):.3f}")
    print()
""")

code("""
# LSH 밴딩: 한 밴드라도 완전히 같으면 후보로 본다.
ROWS = NUM_HASH // BANDS

sigs = [minhash(shingles(d["text"])) for d in exact_dedup]

buckets = defaultdict(list)
for idx, sig in enumerate(sigs):
    for b in range(BANDS):
        band = sig[b * ROWS:(b + 1) * ROWS]
        buckets[(b, band)].append(idx)

candidates = set()
for members in buckets.values():
    if len(members) > 1:
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                candidates.add((members[i], members[j]))

print(f"전체 쌍 비교라면 {len(sigs)*(len(sigs)-1)//2:,}번")
print(f"LSH 후보만 보면   {len(candidates):,}번")
print(f"  → {(1 - len(candidates)/max(1, len(sigs)*(len(sigs)-1)//2))*100:.4f}% 를 건너뛴다")
""")

code("""
THRESHOLD = 0.8   # 이 이상이면 중복으로 본다

drop, near_dup_pairs = set(), []
for i, j in candidates:
    s = sig_sim(sigs[i], sigs[j])
    if s >= THRESHOLD:
        near_dup_pairs.append((i, j, s))
        drop.add(max(i, j))       # 뒤에 나온 것을 버린다

near_dedup = [d for k, d in enumerate(exact_dedup) if k not in drop]
print(f"{len(exact_dedup):,} → {len(near_dedup):,}  (근사 중복 {len(drop):,}개 제거)")

print("\\n── 근사 중복으로 판정된 쌍 ──")
for i, j, s in sorted(near_dup_pairs, key=lambda x: -x[2])[:3]:
    print(f"  유사도 {s:.3f}")
    print(f"    [{exact_dedup[i]['title']}] {exact_dedup[i]['text'][:90]!r}")
    print(f"    [{exact_dedup[j]['title']}] {exact_dedup[j]['text'][:90]!r}")
    print()
""")

# ---------------------------------------------------------------------
md("""
## 4. 품질 필터

중복을 걷어냈으니 이제 **질이 낮은 문서**를 거릅니다.

정답이 있는 작업이 아닙니다. 코퍼스마다, 목적마다 기준이 달라집니다.
아래는 실무에서 흔히 쓰는 휴리스틱들입니다.

| 필터 | 왜 |
|---|---|
| 최소 길이 | 문장 몇 개짜리로는 문맥을 배울 수 없다 |
| 한국어 비율 | 한국어 모델인데 영어·숫자만 가득한 문서 |
| 반복도 | 같은 줄이 반복되는 목록·표는 학습에 해롭다 |
| 특수문자 비율 | 마크업 잔해, 깨진 인코딩 |
| 문장 부호 | 문장 구조가 없는 나열식 문서 |

**중요한 건 무엇이 걸러지는지 직접 보는 것입니다.** 숫자만 보면
필터가 너무 세거나 엉뚱한 것을 자르고 있어도 알 수 없습니다.
""")

code("""
HANGUL = re.compile(r"[가-힣]")
SPECIAL = re.compile(r"[^가-힣a-zA-Z0-9\\s.,!?~%()\\-]")


def quality_report(text: str) -> dict:
    t = normalize(text)
    n = max(1, len(t))
    lines = [l for l in text.split("\\n") if l.strip()]
    line_counts = Counter(lines)

    return {
        "length": len(t),
        "hangul_ratio": len(HANGUL.findall(t)) / n,
        "special_ratio": len(SPECIAL.findall(t)) / n,
        # 가장 많이 반복된 줄이 전체의 몇 %인가
        "dup_line_ratio": (line_counts.most_common(1)[0][1] / len(lines)) if lines else 1.0,
        "sentence_end": len(re.findall(r"[.!?]", t)),
    }


RULES = {
    "너무 짧음":        lambda r: r["length"] < 300,
    "한국어 부족":      lambda r: r["hangul_ratio"] < 0.3,
    "특수문자 과다":    lambda r: r["special_ratio"] > 0.15,
    "같은 줄 반복":     lambda r: r["dup_line_ratio"] > 0.3,
    "문장 구조 없음":   lambda r: r["sentence_end"] < 3,
}
""")

code("""
kept, rejected = [], defaultdict(list)
for d in near_dedup:
    rep = quality_report(d["text"])
    reasons = [name for name, fn in RULES.items() if fn(rep)]
    if reasons:
        rejected[reasons[0]].append((d, rep))
    else:
        kept.append(d)

print(f"{len(near_dedup):,} → {len(kept):,}\\n")
print("── 사유별 제거 건수 ──")
for name in RULES:
    print(f"  {name:<14} {len(rejected[name]):>6,}개")
""")

code("""
# ★ 실제로 무엇이 걸러졌는지 본다. 이 확인을 건너뛰면 안 된다.
for name in RULES:
    if not rejected[name]:
        continue
    d, rep = rejected[name][0]
    print("=" * 70)
    print(f"[{name}] {d['title']}")
    print(f"  길이={rep['length']}  한글={rep['hangul_ratio']:.2f}  "
          f"특수={rep['special_ratio']:.2f}  반복줄={rep['dup_line_ratio']:.2f}  "
          f"문장부호={rep['sentence_end']}")
    print(f"  {d['text'][:200]!r}")
    print()
""")

md("""
### 걸러진 것을 보고 판단하세요

위 출력을 보고 **정상 문서가 잘려나갔다면 기준이 너무 센 것**입니다.
반대로 명백히 쓸모없는 문서가 남아 있으면 기준을 조여야 합니다.

필터 기준은 한 번에 정해지지 않습니다. 돌려보고 눈으로 확인하고 조정하는
과정을 반복합니다. 그래서 데이터 처리에 시간이 오래 걸립니다.
""")

# ---------------------------------------------------------------------
md("""
## 5. 결과 정리
""")

code("""
stages = [
    ("원본",           docs),
    ("정확 중복 제거", exact_dedup),
    ("근사 중복 제거", near_dedup),
    ("품질 필터",      kept),
]

print(f"{'단계':<16}{'문서 수':>10}{'남은 비율':>10}{'총 글자':>14}")
print("-" * 52)
base = len(docs)
for name, dd in stages:
    chars = sum(len(x["text"]) for x in dd)
    print(f"{name:<16}{len(dd):>10,}{len(dd)/base*100:>9.1f}%{chars:>14,}")
""")

code("""
# 정제 전후 길이 분포 비교
def summary(dd, label):
    ls = sorted(len(x["text"]) for x in dd)
    if not ls:
        return
    print(f"{label:<8} 중앙값 {ls[len(ls)//2]:>7,}자   "
          f"최소 {ls[0]:>6,}   최대 {ls[-1]:>8,}")

summary(docs, "원본")
summary(kept, "정제후")
""")

# ---------------------------------------------------------------------
md("""
## 6. 실무에서 쓰는 오픈소스 도구

지금까지 직접 구현한 것은 **원리를 보기 위해서**입니다.
실제 코퍼스는 수 TB 규모라 분산 처리와 최적화가 필요합니다.

| 도구 | 만든 곳 | 특징 |
|---|---|---|
| **datatrove** | HuggingFace | 파이프라인 방식. 로컬·Slurm·S3 에서 같은 코드로 실행 |
| **dolma** | AI2 | OLMo 학습에 쓴 툴킷. 태거 기반 필터링 |
| **text-dedup** | 커뮤니티 | 중복 제거 특화. MinHash·SimHash·Suffix Array |

`datatrove` 로 쓰면 위에서 한 작업이 이렇게 줄어듭니다.

```python
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.filters import (
    GopherQualityFilter,       # 우리가 만든 휴리스틱들의 표준 구현
    LanguageFilter,
)
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.executor import LocalPipelineExecutor

LocalPipelineExecutor(
    pipeline=[
        HuggingFaceDatasetReader("wikimedia/wikipedia", dataset_options={"name": "20231101.ko"}),
        LanguageFilter(languages=["ko"]),
        GopherQualityFilter(),
        MinhashDedupSignature(output_folder="sigs/"),
    ],
    tasks=8,
).run()
```

`GopherQualityFilter` 는 DeepMind 의 Gopher 논문에서 정리한 휴리스틱 모음입니다.
우리가 위에서 손으로 만든 규칙들과 성격이 같습니다.

> 도구를 쓰더라도 **어떤 기준으로 무엇이 걸러지는지는 직접 확인해야 합니다.**
> 기본값이 내 코퍼스에 맞으리라는 보장이 없습니다.
""")

# ---------------------------------------------------------------------
md("""
## 마무리

Pre-training 데이터가 어떻게 만들어지는지 봤습니다.

- **중복**은 정확 중복과 근사 중복으로 나뉘고, 후자가 훨씬 많고 잡기 어렵습니다
- **MinHash + LSH** 로 비교 횟수를 크게 줄일 수 있습니다
- **품질 필터**는 정답이 없습니다. 걸러진 결과를 보면서 조정합니다
- 실무에서는 `datatrove` 같은 도구를 쓰지만 **원리와 확인 과정은 같습니다**

여기서 정제한 코퍼스가 다음 단계인 **Continuous Pre-training** 의 입력이 됩니다.
좋은 모델은 좋은 데이터에서 나오고, 그 데이터는 이런 과정을 거쳐 만들어집니다.
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
    print("커리큘럼 3단원 ⑧ '오픈소스를 활용한 Pre-training 데이터 처리' 대응 자료다.")
    print("기존 자료에는 이 항목이 0장이었다 (review/01_대조표.md).")


if __name__ == "__main__":
    main()
