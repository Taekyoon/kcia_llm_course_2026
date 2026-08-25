"""실행된 노트북(.executed.ipynb)에서 셀 출력을 꺼내 본다.

    python tools/show_nb_outputs.py /tmp/nb_exec/HPC_MiniGPT실습.executed.ipynb
    python tools/show_nb_outputs.py <파일> --grep 학습
    python tools/show_nb_outputs.py <파일> --cell 12

왜 필요한가
-----------
nbconvert 는 "완주했다" 만 알려준다. 그런데 완주가 정상 동작을 뜻하지 않는다.
  - 데이터 필터가 전부 걸러내서 빈 데이터로 학습이 끝났을 수도 있고
  - try/except 가 예외를 삼키고 'error' 를 반환하며 진행했을 수도 있다
실제로 무엇이 출력됐는지 눈으로 봐야 판단이 된다.

표준 라이브러리만 쓴다. VESSL 에서 uv 없이 `python` 으로 바로 돌아간다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MAX = 1200  # 셀당 출력 표시 상한


def text_of(out: dict) -> str:
    t = out.get("output_type")
    if t == "stream":
        s = out.get("text", "")
        return "".join(s) if isinstance(s, list) else s
    if t in ("execute_result", "display_data"):
        d = out.get("data", {})
        s = d.get("text/plain", "")
        return "".join(s) if isinstance(s, list) else s
    if t == "error":
        tb = out.get("traceback", [])
        return "\n".join(tb)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--grep", help="이 문자열이 들어간 셀만")
    ap.add_argument("--cell", type=int, help="이 셀만")
    ap.add_argument("--full", action="store_true", help="출력 자르지 않기")
    args = ap.parse_args()

    nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    cells = nb["cells"]

    shown = 0
    empty = 0
    for i, c in enumerate(cells, start=1):
        if c.get("cell_type") != "code":
            continue
        if args.cell and i != args.cell:
            continue

        src = c.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        outs = c.get("outputs", [])
        body = "\n".join(t for t in (text_of(o) for o in outs) if t).rstrip()

        if not body:
            empty += 1
            continue
        if args.grep and args.grep not in body and args.grep not in src:
            continue

        head = next((l for l in src.splitlines() if l.strip()
                     and not l.strip().startswith("#")), "")[:66]
        print("=" * 74)
        print(f"셀 {i}  |  {head}")
        print("=" * 74)
        print(body if args.full or len(body) <= MAX
              else body[:MAX] + f"\n... (총 {len(body)}자, --full 로 전체 보기)")
        print()
        shown += 1

    n_code = sum(1 for c in cells if c.get("cell_type") == "code")
    print("-" * 74)
    print(f"코드 셀 {n_code}개 · 출력 있는 셀 {shown}개 표시 · 출력 없는 셀 {empty}개")

    # 실패 흔적 요약
    n_err = sum(1 for c in cells for o in c.get("outputs", [])
                if o.get("output_type") == "error")
    if n_err:
        print(f"★ error 출력 {n_err}건 — 예외가 발생했습니다")
    joined = json.dumps(nb, ensure_ascii=False)
    for pat in ("'error'", '"error"', "Traceback"):
        n = joined.count(pat)
        if n:
            print(f"★ {pat!r} 문자열 {n}회 — 삼켜진 예외일 수 있습니다")


if __name__ == "__main__":
    main()
