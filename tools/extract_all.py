"""원본 자료 전부를 extracted/ 로 변환한다.

    uv run python tools/extract_all.py

원본은 절대 수정하지 않는다. 실행 후 무결성 기준선(CLAUDE.md §1)과 대조할 수 있도록
각 원본의 크기를 함께 출력한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import extract_hwp  # noqa: E402
import extract_ipynb  # noqa: E402
import extract_pptx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "extracted"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    written: list[tuple[str, int, int]] = []

    for hwp in sorted(ROOT.glob("*.hwp")):
        dst = OUT / "curriculum.md"
        dst.write_text(extract_hwp.to_markdown(hwp), encoding="utf-8")
        written.append((f"{hwp.name} → {dst.name}", hwp.stat().st_size, dst.stat().st_size))

    for pptx in sorted((ROOT / "ppt").glob("*.pptx")):
        day = pptx.stem.split()[0]  # "1일차 강의자료" → "1일차"
        dst = OUT / f"ppt_{day}.md"
        dst.write_text(extract_pptx.to_markdown(pptx), encoding="utf-8")
        written.append((f"{pptx.name} → {dst.name}", pptx.stat().st_size, dst.stat().st_size))

    for nb in sorted((ROOT / "notebook").rglob("*")):
        if not nb.is_file():
            continue
        day = nb.parent.name  # "1일차"
        name = nb.name.replace(" ", "")
        dst = OUT / f"nb_{day}_{name}.md"
        dst.write_text(extract_ipynb.to_markdown(nb), encoding="utf-8")
        written.append((f"{day}/{nb.name} → {dst.name}", nb.stat().st_size, dst.stat().st_size))

    print(f"{'변환':<62} {'원본':>10} {'추출':>10}  축소")
    print("-" * 100)
    for label, src_size, dst_size in written:
        ratio = dst_size / src_size if src_size else 0
        print(f"{label:<62} {src_size:>10,} {dst_size:>10,}  {ratio:5.1%}")
    print(f"\n{len(written)}개 파일을 {OUT} 에 생성했습니다.")


if __name__ == "__main__":
    main()
