"""Amazon 요약 노트북의 프롬프트가 한국어로 바뀌었는지 확인한다.

    uv run python tools/check_amazon_korean.py

PowerShell 로 확인하면 기본 인코딩(UTF-16) 때문에 한글이 깨져 오탐이 난다.
"""

from __future__ import annotations

import json

from layout import work_path

NB = work_path("HPC_Amazon요약실습.ipynb")   # 일자 배치는 tools/layout.py 가 정한다

ENGLISH_LEFTOVERS = [
    "in English only", "Objective:", "Key Considerations",
    "Example of Desired Output", "Please output as a json format",
]
KOREAN_MARKERS = [
    "모든 출력은 한국어로", "목표:", "할 일:", "주의할 점:", "출력 예시:",
]

nb = json.loads(NB.read_text(encoding="utf-8"))
src = "".join("".join(c.get("source", "")) for c in nb["cells"])

print("=" * 60)
print("영어 잔재 (0 이어야 정상)")
print("=" * 60)
bad = 0
for p in ENGLISH_LEFTOVERS:
    n = src.count(p)
    print(f"  {p:<32} {n}회")
    bad += n

print()
print("=" * 60)
print("한국어 프롬프트")
print("=" * 60)
for p in KOREAN_MARKERS:
    print(f"  {p:<32} {src.count(p)}회")

print()
print("=" * 60)
print("프롬프트 실물 확인")
print("=" * 60)
for i, cell in enumerate(nb["cells"], start=1):
    s = "".join(cell.get("source", ""))
    if "_PROMPT = " in s:
        head = s.split("\n")[0]
        body = "\n".join(s.split("\n")[1:5])
        print(f"\n[셀 {i}] {head}")
        print("  " + body.replace("\n", "\n  "))

print()
if bad:
    raise SystemExit(f"\n★ 영어 잔재 {bad}건. 확인 필요")
print("→ 영어 잔재 없음")
