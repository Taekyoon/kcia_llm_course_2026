"""슬라이드↔노트북 싱크 검사 — 코드박스의 코드/프롬프트가 노트북에 실제로 있나.

    uv run --project tools python tools/check_slide_sync.py <덱.pptx> <노트북.ipynb> [노트북...]

배경: Amazon 슬라이드가 구판 영어 프롬프트인데 노트북은 한국어였다. 라운드트립·잔재
스캔은 "영어 텍스트가 보존되나"만 봐서 이 결함을 통과시켰다(2026-09-03 회고). 이 검사는
**의미 등가**를 본다 — 코드박스의 각 줄이 노트북 어딘가에 있는지.

원칙(사용자 지시 2026-09-03): **놓치느니 과하게 잡는다.** 여기서 걸린 것이 전부 결함은
아니다(발췌·압축·주석 차이로 인한 false alarm 포함). 사람이 훑어 실제 결함을 고른다.
0건이면 확실히 안전, 다수면 그중 '언어가 통째로 다른' 류를 먼저 본다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from pptx import Presentation

CODE_FONTS = {"Courier New", "Consolas"}
TRIVIAL = {")", "]", "}", "):", "]:", "})", "),", "],", "},", "'''", '"""',
           "]}", "})", "else:", "try:", "return", "pass", "()", "[", "{", "("}


def norm(line: str) -> str:
    line = line.replace("\xa0", " ")          # 구 슬라이드는 nbsp 로 들여쓴다
    i = line.find(" #")                        # 인라인 주석 제거 (슬라이드가 덧단 주석 차이 무시)
    if i > 0:
        line = line[:i]
    return " ".join(line.split())              # 공백 정규화


def notebook_index(nb_paths: list[str]) -> tuple[str, set[str]]:
    parts = []
    for p in nb_paths:
        nb = json.loads(Path(p).read_text(encoding="utf-8"))
        for c in nb.get("cells", []):
            if c.get("cell_type") == "code":
                parts.append("".join(c.get("source", [])))
    blob = "\n".join(parts).replace("\xa0", " ")
    lines = {norm(ln) for ln in blob.splitlines()}
    return blob, lines


def is_codebox(shape) -> bool:
    if not shape.has_text_frame:
        return False
    return any(r.font.name in CODE_FONTS
               for para in shape.text_frame.paragraphs for r in para.runs)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("사용: check_slide_sync.py <덱.pptx> <노트북.ipynb> [...]")
    deck, nb_paths = sys.argv[1], sys.argv[2:]
    blob, nb_lines = notebook_index(nb_paths)
    prs = Presentation(deck)

    flagged: list[tuple[int, str]] = []
    total_lines = 0
    for i, s in enumerate(prs.slides):
        for sh in s.shapes:
            if not is_codebox(sh):
                continue
            for para in sh.text_frame.paragraphs:
                raw = "".join(r.text for r in para.runs)
                n = norm(raw)
                total_lines += 1
                if len(n) < 6 or n in TRIVIAL or n.lstrip().startswith("#"):
                    continue
                if n in nb_lines or n in blob:      # 정확 줄 or 부분일치면 통과
                    continue
                flagged.append((i + 1, raw.rstrip()))

    print(f"검사 대상 : {Path(deck).name}")
    print(f"노트북    : {', '.join(Path(p).name for p in nb_paths)}")
    print(f"코드박스 라인 {total_lines}개 대조 → 노트북에서 못 찾음 {len(flagged)}건")
    print("(놓치느니 과하게 잡음 — 발췌·압축은 false alarm. '언어가 통째로 다른' 류를 먼저 볼 것)\n")
    cur = None
    for pno, line in flagged:
        if pno != cur:
            print(f"── p{pno} ──")
            cur = pno
        print(f"   {line}")
    if not flagged:
        print("전부 노트북에서 확인됨 — 싱크 이상 없음.")


if __name__ == "__main__":
    main()
