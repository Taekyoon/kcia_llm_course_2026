"""노트북의 **설명(산문)** 상태를 점검한다.

    uv run python tools/check_prose.py
    uv run python tools/check_prose.py --gaps    # 설명 공백 위치까지

`validate_notebooks.py` 는 문법만 본다. 산문은 아무도 안 본다.
그런데 이 프로젝트에서 실제로 문제였던 것은 문법이 아니라 **설명이 없는 것**이었다 —
25년 원본 8종의 마크다운 평균이 622자였고, 신규 4종은 5,060자였다.

여기서 보는 것 넷:
  1. 노트북별 마크다운 분량 (목표 대비)
  2. 용어 표준 위반 (CLAUDE.md §7)
  3. 영어 헤딩 (고유 기술용어는 예외)
  4. **설명 공백** — 마크다운 없이 코드 셀이 이어지는 구간

4번이 핵심이다. 어디가 비었는지 사람이 세지 않아도 된다.
"""

from __future__ import annotations

import argparse
import json
import re

from layout import LAYOUT, work_path

TARGET_MIN = 3_500          # 이 아래면 설명이 부족하다
GAP_LIMIT = 5               # 코드 셀이 이만큼 이어지면 공백으로 본다

# CLAUDE.md §7 용어 표준
BAD_TERMS = {
    "트렌스포머": "트랜스포머",
    "Vllm": "vLLM",
    "VLLM": "vLLM",
    "RLPH": "RLHF",
    "높히": "높이",
    "Persistance": "Persistence",
}

# 영어 헤딩이어도 되는 것 — 고유 기술용어
TERM_OK = re.compile(
    r"Causal Self-Attention|LLM-as-judge|BM25|MinHash|LoRA|GRPO|DPO|ORPO|SFT|"
    r"RAG|GPT|BERT|NER|Pre-training|Top-K|Top-P|Beam|Greedy|DSPy|GEPA",
    re.I,
)
HANGUL = re.compile(r"[가-힣]")


def cells_of(path):
    return json.loads(path.read_text(encoding="utf-8"))["cells"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", action="store_true", help="설명 공백 위치를 모두 표시")
    args = ap.parse_args()

    print("=" * 74)
    print(f"{'노트북':<30}{'셀':>4}{'MD':>4}{'글자':>8}{'공백':>6}  판정")
    print("=" * 74)

    problems: list[str] = []
    total_gaps = 0

    for day, names in LAYOUT.items():
        for name in names:
            path = work_path(name)
            if not path.exists():
                continue
            cells = cells_of(path)
            mds = [c for c in cells if c["cell_type"] == "markdown"]
            chars = sum(len("".join(c["source"])) for c in mds)

            # --- 설명 공백: 마크다운 없이 이어지는 코드 셀 구간 ---
            gaps, run, start = [], 0, 0
            for i, c in enumerate(cells, 1):
                if c["cell_type"] == "code":
                    if run == 0:
                        start = i
                    run += 1
                else:
                    if run >= GAP_LIMIT:
                        gaps.append((start, i - 1, run))
                    run = 0
            if run >= GAP_LIMIT:
                gaps.append((start, len(cells), run))
            total_gaps += len(gaps)

            verdict = "충분" if chars >= TARGET_MIN else "★ 부족"
            print(f"{name[:29]:<30}{len(cells):>4}{len(mds):>4}{chars:>8,}"
                  f"{len(gaps):>6}  {verdict}")
            if chars < TARGET_MIN:
                problems.append(f"{name} — 마크다운 {chars:,}자 (목표 {TARGET_MIN:,}+)")
            if args.gaps:
                for a, b, n in gaps:
                    print(f"      공백: 셀 {a}~{b} ({n}개 연속)")

            # --- 용어 · 영어 헤딩 ---
            src = "".join("".join(c["source"]) for c in cells)
            for bad, good in BAD_TERMS.items():
                if bad in src:
                    problems.append(f"{name} — 용어 '{bad}' → '{good}' ({src.count(bad)}회)")
            for i, c in enumerate(cells, 1):
                if c["cell_type"] != "markdown":
                    continue
                for line in "".join(c["source"]).splitlines():
                    if not line.startswith("#"):
                        continue
                    if HANGUL.search(line) or TERM_OK.search(line):
                        continue
                    problems.append(f"{name} 셀 {i} — 영어 헤딩: {line.strip()[:50]}")

    print("=" * 74)
    print(f"설명 공백 총 {total_gaps}곳 (코드 셀 {GAP_LIMIT}개 이상 연속)")
    if not args.gaps and total_gaps:
        print("  --gaps 로 위치를 볼 수 있습니다.")
    print()
    if problems:
        print(f"★ 지적 {len(problems)}건")
        for p in problems:
            print(f"  · {p}")
    else:
        print("지적 없음.")


if __name__ == "__main__":
    main()
