"""노트북에서 소스만 추출해 마크다운으로 변환한다.

notebook/ 의 파일들은 확장자가 없지만 내용은 .ipynb JSON이다.
outputs / execution_count 를 제거하지 않으면 파일 크기의 대부분(base64 이미지, pip 로그)이
그대로 딸려와 컨텍스트를 낭비한다. (CLAUDE.md §6 참조)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def to_markdown(nb_path: Path) -> str:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = nb.get("cells", [])

    out: list[str] = [
        f"# {nb_path.name}",
        "",
        f"> 원본: `{nb_path.relative_to(nb_path.parents[2]) if len(nb_path.parents) > 2 else nb_path.name}`"
        f"  ·  셀 {len(cells)}개",
        "> 추출: `tools/extract_ipynb.py` (outputs 제거됨)",
        "",
    ]

    for idx, cell in enumerate(cells, start=1):
        kind = cell.get("cell_type", "?")
        src = _source(cell).rstrip()
        if not src:
            out.append(f"### 셀 {idx} · {kind} · _(비어 있음)_")
            out.append("")
            continue

        out.append(f"### 셀 {idx} · {kind}")
        out.append("")
        if kind == "code":
            out.append("```python")
            out.append(src)
            out.append("```")
        else:
            out.append(src)
        out.append("")

    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="노트북 소스 추출 (outputs 제거)")
    ap.add_argument("notebook", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    args = ap.parse_args()

    md = to_markdown(args.notebook)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"wrote {args.out} ({len(md)} chars)")
    else:
        print(md)


if __name__ == "__main__":
    main()
