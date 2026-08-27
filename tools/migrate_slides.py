"""슬라이드 재조립기 — 원본 3덱에서 새 3덱을 **매번 다시** 만든다.

    uv run --project tools python tools/migrate_slides.py

`migrate_notebooks.py` 와 같은 규율:
- 원본(`ppt/`)은 읽기만 한다. 산출물은 `work/ppt/` 에 통째로 다시 쓴다.
- 치환 규칙에는 근거(why)를 붙인다. 앵커를 못 찾으면 조용히 넘어가지 않고 멈춘다.
- 배치는 `slide_layout.PLACEMENT` 가 단독으로 정한다.

신규 슬라이드는 **도너 복제** 방식이다: 그림 없는 텍스트 전용 장(1일차 p19)을
복제해 헤더·본문 텍스트만 갈아 끼운다. placeholder 방식은 이 덱들과 맞지 않는다 —
전 장이 Blank 레이아웃 + 슬라이드 내 장식 도형 구조라, 도너 쪽이 장식을 보존한다.
"""

from __future__ import annotations

import copy
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.slide import Slide
from pptx.util import Pt

import pptxcommon as pc
from slide_layout import DECKS, DECK_PAGES, PLACEMENT, WORK_PPT, NewSlide, SlideRef, out_path

# 신규 슬라이드의 도너 — 그림 없는 텍스트 전용 장 (2026-08-27 도형 실사로 선정)
DONOR = ("1일차", 19)
BOILER = "쉬운 HPC 활용교육"


# ---------------------------------------------------------------------------
# 치환 규칙
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlideRule:
    """원본 (덱, 페이지) 한 장 안의 문자열 치환. 임포트 시점에 적용되므로
    재배치로 번호가 밀릴 일이 없다. 그 장에서 정확히 count 회 치환돼야 한다."""
    deck: str
    page: int
    old: str
    new: str
    why: str
    count: int = 1


@dataclass(frozen=True)
class GlobalSlideRule:
    """재사용되는 원본 전 장 대상. 총 치환 횟수가 expect 와 다르면 중단."""
    old: str
    new: str
    why: str
    expect: int


# 개별 장 규칙 — 코드 슬라이드의 현행 노트북 동기화는 Phase E 후반에 채운다.
RULES: list[SlideRule] = []

# 전역 규칙. expect 는 재사용 257장의 XML 결합 텍스트 실측값 (2026-08-27).
# 섹션 규칙은 이름으로 구분되어 서로의 출력을 다시 잡지 않는다 (충돌 분석 완료).
GLOBAL_RULES: list[GlobalSlideRule] = [
    GlobalSlideRule('3.\t트렌스포머와 GPT', '2.\t트랜스포머와 ChatGPT',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 14),
    GlobalSlideRule('2.\t자연어처리 머신러닝 소개', '3.\t자연어처리 머신러닝 소개',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('2. 자연어처리 머신러닝 소개', '3. 자연어처리 머신러닝 소개',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 26),
    GlobalSlideRule('4. 미니 GPT 만들기', '3. 미니 GPT 만들기',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 28),
    GlobalSlideRule('1. 도메인 최적화 프리트레인에 대해서', '4. 도메인 최적화 프리트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측) + 표기 정정', 1),
    GlobalSlideRule('1. 도메인 최적화 프리트레이닝에 대해서', '4. 도메인 최적화 프리트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 13),
    GlobalSlideRule('2. VLLM을 활용한 데이터처리 실습', '5. vLLM을 활용한 데이터 생성 실습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측). Amazon 은 정제가 아니라 생성 축이라 명칭도 맞춘다', 35),
    GlobalSlideRule('3. Llama Index를 활용한 RAG 실습', '1. Llama Index를 활용한 RAG 실습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 18),
    GlobalSlideRule('4. 지속적인 LLM 챗봇 개발을 위해서', '2. 태스크 정의와 평가',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측). 커리큘럼 세부항목명과 일치', 10),
    GlobalSlideRule('1. 퓨샷 러닝에 대해서', '3. 퓨샷 러닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('1. 퓨샷러닝에 대해서', '3. 퓨샷러닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 14),
    GlobalSlideRule('2. 포스트 트레이닝에 대해서', '4. 포스트 트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 11),
    GlobalSlideRule('3.\t 인스트럭션 모델 학습', '5.\t 인스트럭션 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('3. 인스트럭션 모델학습', '5. 인스트럭션 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 22),
    GlobalSlideRule('4.\t 선호기반 모델 학습', '6.\t 선호기반 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('4. 선호기반 모델학습', '6. 선호기반 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 17),
    GlobalSlideRule('5.\t 리즈닝 모델 학습', '7.\t 리즈닝 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('5. 리즈닝 모델학습', '7. 리즈닝 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 18),
    GlobalSlideRule('Vllm', 'vLLM',
                    '용어 표준 (CLAUDE.md §7)', 15),
    GlobalSlideRule('evaluation_strategy', 'eval_strategy',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측) — transformers 5 에서 제거, 슬라이드 코드가 깨진다 (C-3)', 2),
    GlobalSlideRule('trainer.tokenizer', 'trainer.processing_class',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측)', 3),
    GlobalSlideRule('max_seq_length', 'max_length',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측) — SFTConfig 인자명 변경', 1),
    GlobalSlideRule('RLPH', 'RLHF',
                    '용어 표준 (CLAUDE.md §7)', 1),
    GlobalSlideRule('높히', '높이',
                    '용어 표준 (CLAUDE.md §7)', 1),
]

# ---------------------------------------------------------------------------
# 신규 슬라이드 레지스트리 — key → 장 목록
#   각 장: dict(header=우상단 헤더, title=소제목, bullets=[(level, text), ...])
#   Phase F 에서 원고로 대체한다. 지금은 파이프라인 검증용 스텁.
# ---------------------------------------------------------------------------
def _stub(key: str, memo: str, n: int) -> list[dict]:
    return [{
        "header": "(신규) " + memo.split("(")[0].strip(),
        "title": f"{memo.split('(')[0].strip()} — 초안 {i + 1}/{n}",
        "bullets": [(0, "원고는 Phase F 에서 작성됩니다."),
                    (1, f"배치 키: {key}")],
    } for i in range(n)]


NEW_SLIDES: dict[str, list[dict]] = {
    "toc1": [{
        "header": "목 차",
        "title": "1일차",
        "bullets": [(0, "1. AI 소프트웨어 개발 개론"),
                    (0, "2. 트랜스포머와 ChatGPT"),
                    (1, "머신러닝에서 딥러닝, 그리고 어텐션"),
                    (1, "트랜스포머 모델에서 BERT까지 · GPT와 ChatGPT"),
                    (1, "실습: 감성 분류 · 개체명 인식(NER)")],
    }],
    "toc2": [{
        "header": "목 차",
        "title": "2일차 — LLM Pre-training",
        "bullets": [(0, "Pre-training 등장 배경과 목적"),
                    (0, "오픈소스를 활용한 Pre-training 데이터 처리"),
                    (0, "미니 GPT 학습 실습"),
                    (0, "Continuous Pre-training과 한국어 GPT"),
                    (0, "데이터 생성 파이프라인과 프롬프트 자동 최적화")],
    }],
    "toc3": [{
        "header": "목 차",
        "title": "3일차 — LLM Post-training",
        "bullets": [(0, "RAG 시스템 만들기 (심화 선행)"),
                    (0, "태스크 정의와 평가"),
                    (0, "Few-shot 프롬프트 엔지니어링"),
                    (0, "모델 파인튜닝과 LoRA"),
                    (0, "Preference Learning과 GRPO"),
                    (0, "마무리: 방법 선택 가이드 · RAG 개선 확인")],
    }],
    "ml2dl": _stub("ml2dl", "④ ML→DL 발전사", 4),
    "pretrain_intro": _stub("pretrain_intro", "⑦ Pre-training 등장 배경과 목적", 5),
    "datacleaning": _stub("datacleaning", "⑧ Pre-training 데이터 처리", 10),
    "promptopt": _stub("promptopt", "프롬프트 자동 최적화", 4),
    "lora": _stub("lora", "⑬ LoRA 이론", 4),
    "method_guide": _stub("method_guide", "방법 선택 가이드", 7),
    "ragcheck": _stub("ragcheck", "RAG 개선 확인 — 3일의 마무리", 2),
}


# ---------------------------------------------------------------------------
# 텍스트 채우기 (도너 복제 후)
# ---------------------------------------------------------------------------
def _set_text(tf, lines: list[tuple[int, str]], base_size: Pt | None) -> None:
    """텍스트 프레임을 비우고 (level, text) 목록으로 다시 채운다.
    서식은 도너 첫 run 의 크기만 물려받는다 — 텍스트 초안 수준(사용자 확정)."""
    first_run = None
    for p_el in tf.paragraphs:
        for r in p_el.runs:
            first_run = r
            break
        if first_run:
            break
    font_name = first_run.font.name if first_run else None
    tf.clear()
    for i, (level, text) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = min(level, 4)
        for r in p.runs:
            if font_name:
                r.font.name = font_name
            if base_size:
                r.font.size = base_size


def fill_new_slide(part, spec: dict) -> None:
    slide = Slide(part._element, part)
    boxes = [sh for sh in slide.shapes
             if sh.has_text_frame and BOILER not in sh.text_frame.text]
    boxes = [b for b in boxes if b.text_frame.text.strip()]
    if len(boxes) < 3:
        raise SystemExit(f"[중단] 도너 장의 텍스트 상자가 예상과 다릅니다: {len(boxes)}개")
    boxes.sort(key=lambda b: (b.top or 0))
    header, body, tagline = boxes[0], max(boxes[1:], key=lambda b: (b.width or 0) * (b.height or 0)), None
    tagline = next((b for b in boxes[1:] if b is not body), None)

    _set_text(header.text_frame, [(0, spec["header"])], None)
    lines = [(0, spec["title"])] + [(lv + 1, tx) for lv, tx in spec["bullets"]]
    _set_text(body.text_frame, lines, None)
    if tagline is not None:
        _set_text(tagline.text_frame, [(0, "")], None)


# ---------------------------------------------------------------------------
# 빌드
# ---------------------------------------------------------------------------
def expected_counts() -> dict[str, int]:
    out = {}
    for day, items in PLACEMENT.items():
        n = 0
        for it in items:
            n += len(NEW_SLIDES[it.key]) if isinstance(it, NewSlide) else 1
        out[day] = n
    return out


def build(dst_dir: Path | None = None, verbose: bool = True) -> dict[str, Path]:
    dst_dir = dst_dir or WORK_PPT
    pc.guard_masters_identical(DECKS)

    srcs = {d: Presentation(str(p)) for d, p in DECKS.items()}
    zips = {d: zipfile.ZipFile(p) for d, p in DECKS.items()}
    for d, prs in srcs.items():
        if len(prs.slides) != DECK_PAGES[d]:
            raise SystemExit(f"[중단] {d} 원본 장수 {len(prs.slides)} ≠ 기대 {DECK_PAGES[d]}")

    rules_by_page: dict[tuple[str, int], list[SlideRule]] = {}
    for r in RULES:
        rules_by_page.setdefault((r.deck, r.page), []).append(r)
    global_hits = {id(g): 0 for g in GLOBAL_RULES}

    outputs = {}
    for day, items in PLACEMENT.items():
        shell = Presentation(str(DECKS[day]))
        pc.drop_all_slides(shell)
        media = pc.MediaCache(shell.part.package)
        sid = 256
        n_ref = n_new = 0

        for it in items:
            if isinstance(it, SlideRef):
                def mutate(el, _ref=it):
                    for rule in rules_by_page.get((_ref.deck, _ref.page), []):
                        n = pc.replace_in_tree(el, rule.old, rule.new)
                        if n != rule.count:
                            raise SystemExit(
                                f"[중단] 규칙 불일치 — {_ref.deck} p{_ref.page}: "
                                f"{rule.old!r} 를 {n}회 치환 (기대 {rule.count})\n  사유: {rule.why}")
                    for g in GLOBAL_RULES:
                        global_hits[id(g)] += pc.replace_in_tree(el, g.old, g.new)
                part = pc.clone_slide(srcs[it.deck], zips[it.deck], it.page, shell,
                                      media, mutate_element=mutate)
                pc.append_slide_part(shell, part, sid)
                n_ref += 1
            else:
                for spec in NEW_SLIDES[it.key]:
                    part = pc.clone_slide(srcs[DONOR[0]], zips[DONOR[0]], DONOR[1],
                                          shell, media)
                    fill_new_slide(part, spec)
                    pc.append_slide_part(shell, part, sid)
                    sid += 1
                    n_new += 1
                sid -= 1
            sid += 1

        dst_dir.mkdir(parents=True, exist_ok=True)
        out = dst_dir / out_path(day).name
        shell.save(str(out))
        outputs[day] = out
        if verbose:
            print(f"{day}: 재사용 {n_ref} + 신규 {n_new} = {n_ref + n_new}장 → {out}")

    for g in GLOBAL_RULES:
        if global_hits[id(g)] != g.expect:
            raise SystemExit(
                f"[중단] 전역 규칙 {g.old!r}: {global_hits[id(g)]}회 치환 (기대 {g.expect})\n"
                f"  사유: {g.why}")
    if verbose and GLOBAL_RULES:
        print(f"전역 규칙 {len(GLOBAL_RULES)}건 — 전부 기대 횟수와 일치")
    return outputs


if __name__ == "__main__":
    build()
    print("\n검증: uv run --project tools python tools/validate_slides.py")
