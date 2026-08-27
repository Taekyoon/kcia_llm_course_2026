"""PPTX 크로스-파일 슬라이드 복제 프리미티브.

설계 근거 (2026-08-27 원본 zip 실사):
- 3덱의 theme·slideMaster·slideLayout 11종이 자동 날짜 필드 1곳 빼고 바이트 동일
  → 레이아웃 참조를 대상 덱의 **같은 파일명** 레이아웃으로 매핑해도 시각 차이 없음
- 264장 전부 slideLayout7("Blank")만 참조. 차트·OLE·전환효과 0건
- 슬라이드 rels 는 image / slideLayout / 외부 hyperlink 1건 / notesSlide 1건(빈 것)뿐

핵심 원칙 — **rId 를 슬라이드 XML 에서 고치지 않는다.**
원본 슬라이드 XML 을 그대로 새 파트로 만들고, 새 파트의 rels 를 **원본과 같은 rId** 로
구성한다. rId 는 파트별 rels 에 지역적이라 이것으로 충분하다.

허용 목록 밖의 관계 타입을 만나면 **즉시 중단**한다 (migrate_notebooks 의 fail-loud).
"""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path

from pptx import Presentation
from pptx.opc.constants import CONTENT_TYPE as CT
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.opc.package import OpcPackage
from pptx.opc.packuri import PackURI
from pptx.oxml.ns import qn
from pptx.parts.slide import SlidePart
from lxml import etree

# 이미지 확장자 → content type (원본 실사: png·jpeg·svg 만 존재)
_IMAGE_CT = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "svg": "image/svg+xml",
}

# 자동 날짜 필드 캐시 — 마스터 동일성 비교 전에 정규화한다
_DATE_FIELD = re.compile(rb"<a:t>\d{1,2}/\d{1,2}/\d{4}</a:t>")


# ---------------------------------------------------------------------------
# 마스터 동일성 가드
# ---------------------------------------------------------------------------
def _normalized_part_hashes(pptx_path: Path) -> dict[str, str]:
    import zipfile

    out = {}
    with zipfile.ZipFile(pptx_path) as z:
        for name in z.namelist():
            if name.startswith(("ppt/slideMasters/", "ppt/slideLayouts/", "ppt/theme/")) \
                    and name.endswith(".xml"):
                blob = _DATE_FIELD.sub(b"<a:t>DATE</a:t>", z.read(name))
                out[name] = hashlib.sha256(blob).hexdigest()
    return out


def guard_masters_identical(deck_paths: dict[str, Path]) -> None:
    """3덱의 마스터·레이아웃·테마가 (날짜 정규화 후) 동일한지 매 빌드 확인한다.

    다르면 크로스-덱 레이아웃 매핑이 시각 차이를 만들 수 있으므로 중단한다.
    """
    ref_day, *rest = deck_paths
    ref = _normalized_part_hashes(deck_paths[ref_day])
    for day in rest:
        other = _normalized_part_hashes(deck_paths[day])
        if other != ref:
            diff = {k for k in ref.keys() ^ other.keys()} | \
                   {k for k in ref.keys() & other.keys() if ref[k] != other[k]}
            raise SystemExit(
                f"[중단] {ref_day} 와 {day} 의 마스터/레이아웃/테마가 다릅니다: {sorted(diff)}\n"
                f"  원본이 교체된 것 같습니다. 크로스-덱 복제 가정이 깨졌습니다."
            )


# ---------------------------------------------------------------------------
# 대상 덱(셸) 준비
# ---------------------------------------------------------------------------
def drop_all_slides(prs) -> int:
    """셸 덱의 기존 슬라이드를 전부 뗀다. 고아 파트는 저장 시 자동 소거된다."""
    sldIdLst = prs.slides._sldIdLst
    n = 0
    for sldId in list(sldIdLst):
        prs.part.drop_rel(sldId.get(qn("r:id")))
        sldIdLst.remove(sldId)
        n += 1
    return n


def layout_by_basename(prs) -> dict[str, object]:
    """대상 덱의 레이아웃 파트를 파일명(slideLayout7.xml)으로 찾는 표."""
    table = {}
    for master in prs.slide_masters:
        for layout in master.slide_layouts:
            table[Path(str(layout.part.partname)).name] = layout.part
    return table


# ---------------------------------------------------------------------------
# 슬라이드 1장 복제
# ---------------------------------------------------------------------------
class MediaCache:
    """(sha1, 확장자) → 대상 패키지에 이미 넣은 이미지 파트. 덱 간 로고 공유 dedup."""

    def __init__(self, package: OpcPackage):
        self.package = package
        self._by_hash: dict[tuple[str, str], object] = {}
        self._n = 0

    def get_or_add(self, blob: bytes, ext: str):
        key = (hashlib.sha1(blob).hexdigest(), ext)
        if key in self._by_hash:
            return self._by_hash[key]
        ct = _IMAGE_CT.get(ext)
        if ct is None:
            raise SystemExit(f"[중단] 지원 목록 밖 이미지 확장자: {ext}")
        self._n += 1
        # next_partname 은 마스터가 참조하는 기존 media 와 충돌한 적이 있다(실측 —
        # zip 에 Duplicate name 경고). 원본 인덱스가 닿지 않는 500번대에서 직접 배번한다.
        partname = PackURI(f"/ppt/media/image{500 + self._n}.{ext}")
        part = _make_part(partname, ct, blob, self.package)
        self._by_hash[key] = part
        return part


def _make_part(partname, content_type, blob, package):
    """python-pptx 1.0.x Part 생성. (버전에 따라 인자 순서가 달라 방어적으로.)"""
    from pptx.opc.package import Part

    try:
        return Part(partname, content_type, package, blob)
    except Exception:
        return Part(partname, content_type, blob, package)  # 구버전 시그니처


def _add_rel_with_rid(rels, reltype, target, rId, external=False):
    """새 파트의 rels 에 **지정한 rId 로** 관계를 추가한다.

    python-pptx 1.0.2 의 `_add_relationship` 은 rId 를 자동 배번만 하므로,
    같은 내부 표현(`_Relationship`)을 직접 만들어 지정 rId 로 넣는다 (실측 확인).
    시그니처가 바뀌면 여기서만 고친다.
    """
    from pptx.opc.package import _Relationship

    if rId in rels._rels:
        raise SystemExit(f"[중단] rId 충돌: {rId} 가 이미 있습니다 ({rels._base_uri})")
    rels._rels[rId] = _Relationship(
        rels._base_uri, rId, reltype,
        target_mode="External" if external else "Internal",
        target=target,
    )


def clone_slide(src_prs, src_zip, page: int, dst_prs, media: MediaCache,
                mutate_element=None):
    """src 덱의 page(1-기준) 슬라이드를 dst 덱의 새 파트로 복제해 돌려준다.

    mutate_element(element) 훅으로 복제된 XML 트리에 텍스트 치환을 적용할 수 있다
    (파트 등록 전 · 원본 트리는 불변).
    """
    src_slide = src_prs.slides[page - 1]
    element = copy.deepcopy(src_slide._element)
    if mutate_element is not None:
        mutate_element(element)

    package = dst_prs.part.package
    partname = PackURI(package.next_partname("/ppt/slides/slide%d.xml"))
    blob = etree.tostring(element, xml_declaration=True, encoding="UTF-8",
                          standalone=True)
    new_part = _load_slide_part(partname, blob, package)

    layouts = layout_by_basename(dst_prs)
    for rel in _iter_rels(src_slide.part):
        reltype = rel.reltype
        if reltype == RT.SLIDE_LAYOUT:
            base = Path(str(rel._target.partname)).name
            if base not in layouts:
                raise SystemExit(f"[중단] 대상 덱에 레이아웃 {base} 이 없습니다")
            _add_rel_with_rid(new_part.rels, reltype, layouts[base], rel.rId)
        elif reltype == RT.IMAGE:
            src_partname = str(rel._target.partname)
            img_blob = src_zip.read(src_partname.lstrip("/"))
            ext = src_partname.rsplit(".", 1)[-1].lower()
            _add_rel_with_rid(new_part.rels, reltype, media.get_or_add(img_blob, ext),
                              rel.rId)
        elif reltype == RT.HYPERLINK:
            _add_rel_with_rid(new_part.rels, reltype, rel.target_ref, rel.rId,
                              external=True)
        elif reltype == RT.NOTES_SLIDE:
            continue  # 원본 전체에 1건, 내용은 슬라이드 번호뿐 — 드롭 (실사 확인)
        else:
            raise SystemExit(
                f"[중단] 허용 목록 밖 관계: {reltype}\n"
                f"  원본이 바뀌었거나 가정이 깨졌습니다. pptxcommon 의 정책 표를 보강하세요."
            )
    return new_part


def _load_slide_part(partname, blob, package):
    """SlidePart 를 blob 에서 만든다. 1.0.x 의 load 클래스메서드를 쓴다."""
    try:
        return SlidePart.load(partname, CT.PML_SLIDE, package, blob)
    except TypeError:
        return SlidePart.load(partname, CT.PML_SLIDE, blob, package)  # 구버전 순서


def _iter_rels(part):
    rels = part.rels
    values = rels.values() if hasattr(rels, "values") else rels
    return list(values)


def append_slide_part(dst_prs, slide_part, slide_id: int) -> None:
    """복제한 파트를 프레젠테이션 sldIdLst 끝에 단다."""
    rId = dst_prs.part.relate_to(slide_part, RT.SLIDE)
    sldIdLst = dst_prs.slides._sldIdLst
    sldId = sldIdLst.makeelement(qn("p:sldId"), {"id": str(slide_id)})
    sldId.set(qn("r:id"), rId)
    sldIdLst.append(sldId)


# ---------------------------------------------------------------------------
# 문단 결합 텍스트 치환 (run 파편화 79~82% 대응)
# ---------------------------------------------------------------------------
def paragraph_text(p_el) -> str:
    return "".join(t.text or "" for t in p_el.findall(".//" + qn("a:t")))


def replace_in_tree(element, old: str, new: str) -> int:
    """복제된 슬라이드 트리에서 old→new. 문단 결합 문자열 기준으로 찾고,
    매치와 겹치는 run 들에 재배분한다. 치환 횟수를 돌려준다."""
    n = 0
    for p_el in element.findall(".//" + qn("a:p")):
        joined = paragraph_text(p_el)
        if old not in joined:
            continue
        ts = p_el.findall(".//" + qn("a:t"))
        # run 오프셋 표
        spans, pos = [], 0
        for t in ts:
            txt = t.text or ""
            spans.append((pos, pos + len(txt)))
            pos += len(txt)
        replaced = joined.replace(old, new)
        n += joined.count(old)
        # 재배분: 첫 run 에 전부 넣고 나머지는 비운다 — 서식은 첫 run 을 따른다.
        # (모델명·오탈자 교체 수준에서는 충분. 문단 전체가 하나의 서식 덩어리가 된다는
        #  한계가 있으므로, 부분 서식이 중요한 문단은 규칙을 더 좁게 잡을 것.)
        if ts:
            ts[0].text = replaced
            for t in ts[1:]:
                t.text = ""
    return n
