"""재조립된 3덱의 무결성 검증.

    uv run --project tools python tools/validate_slides.py            # 기본 4종
    uv run --project tools python tools/validate_slides.py --twice    # 결정론까지

검사:
  1. 재오픈 + 장수가 배치표와 일치
  2. rId 린트 — 슬라이드 XML 의 r:embed/r:link/r:id 가 전부 그 파트 rels 에 실재
  3. 텍스트 라운드트립 — 원본 텍스트에 규칙을 문자열로 적용한 결과 == 출력 텍스트
     (치환 누락·과잉과 텍스트 손실을 동시에 잡는다)
  4. 원본 불가침 — ppt/*.pptx 크기가 CLAUDE.md §1 기준선과 동일
  5. (--twice) 두 번 빌드해 zip 멤버별 sha256 비교 — zip 자체는 mtime 때문에 다르다
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx import Presentation
from pptx.oxml.ns import qn

from migrate_slides import GLOBAL_RULES, RULES, build, expected_counts
from slide_layout import DECKS, PLACEMENT, SlideRef, out_path

BASELINE_SIZES = {"1일차": 15450466, "2일차": 4297796, "3일차": 4588720}

R_ATTRS = [qn("r:embed"), qn("r:link"), qn("r:id")]


def slide_texts(slide) -> list[str]:
    return sorted(sh.text_frame.text for sh in slide.shapes if sh.has_text_frame)


def apply_rules_str(deck: str, page: int, texts: list[str]) -> list[str]:
    out = []
    for t in texts:
        for r in RULES:
            if r.deck == deck and r.page == page:
                t = t.replace(r.old, r.new)
        for g in GLOBAL_RULES:
            t = t.replace(g.old, g.new)
        out.append(t)
    return sorted(out)


def main() -> None:
    fails = []
    print("=" * 70)

    # 4. 원본 불가침 (빌드 전후 모두 지켜져야 하므로 먼저·나중 두 번 본다)
    def guard_originals(tag):
        for d, p in DECKS.items():
            sz = p.stat().st_size
            if sz != BASELINE_SIZES[d]:
                fails.append(f"원본 훼손({tag}): {d} 크기 {sz} ≠ 기준선 {BASELINE_SIZES[d]}")

    guard_originals("빌드 전")
    outputs = build(verbose=False)
    guard_originals("빌드 후")
    print("[4] 원본 불가침 — 기준선과 일치")

    srcs = {d: Presentation(str(p)) for d, p in DECKS.items()}
    want_counts = expected_counts()

    for day, out in outputs.items():
        prs = Presentation(str(out))

        # 1. 장수
        if len(prs.slides) != want_counts[day]:
            fails.append(f"{day}: 장수 {len(prs.slides)} ≠ 배치표 {want_counts[day]}")

        # 2. rId 린트
        for i, sl in enumerate(prs.slides, 1):
            rids_in_xml = {el.get(a) for el in sl.part._element.iter()
                           for a in R_ATTRS if el.get(a)}
            rels_keys = set(sl.part.rels.keys()) if hasattr(sl.part.rels, "keys") else set()
            missing = rids_in_xml - rels_keys
            if missing:
                fails.append(f"{day} 새 p{i}: XML 이 참조하는 rId 가 rels 에 없음 {missing}")

        # 3. 텍스트 라운드트립 (SlideRef 만 — 신규 장은 제목 존재만 본다)
        idx = 0
        for it in PLACEMENT[day]:
            if isinstance(it, SlideRef):
                got = slide_texts(prs.slides[idx])
                want = apply_rules_str(it.deck, it.page,
                                       slide_texts(srcs[it.deck].slides[it.page - 1]))
                if got != want:
                    diff = [(a, b) for a, b in zip(want, got) if a != b][:2]
                    fails.append(f"{day} 새 p{idx+1} ← {it.deck} p{it.page}: 텍스트 불일치 {diff}")
                idx += 1
            else:
                from migrate_slides import NEW_SLIDES
                for spec in NEW_SLIDES[it.key]:
                    joined = " ".join(slide_texts(prs.slides[idx]))
                    want_txt = (spec["toc"][0].split(chr(9))[-1]   # 자동번호 도너는 텍스트 번호를 뗀다
                                if "toc" in spec else spec["title"])
                    if want_txt not in joined:
                        fails.append(f"{day} 새 p{idx+1}: 신규 장 내용 누락 — {want_txt!r}")
                    idx += 1
        print(f"[1-3] {day}: {len(prs.slides)}장 — 장수·rId·라운드트립 확인")

    # 5. 결정론
    if "--twice" in sys.argv:
        with tempfile.TemporaryDirectory() as td:
            second = build(dst_dir=Path(td), verbose=False)
            for day in outputs:
                h1 = _member_hashes(outputs[day])
                h2 = _member_hashes(second[day])
                if h1 != h2:
                    diff = {k for k in h1.keys() ^ h2.keys()} | \
                           {k for k in h1.keys() & h2.keys() if h1[k] != h2[k]}
                    fails.append(f"{day}: 비결정 멤버 {sorted(diff)[:5]}")
        print("[5] 결정론 — 두 빌드의 zip 멤버 해시 일치" if not fails else "[5] 결정론 검사 수행")

    print("=" * 70)
    if fails:
        print(f"실패 {len(fails)}건:")
        for f in fails:
            print("  ✗", f)
        raise SystemExit(1)
    print("전부 통과.")


def _member_hashes(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        return {n: hashlib.sha256(z.read(n)).hexdigest() for n in sorted(z.namelist())}


if __name__ == "__main__":
    main()
