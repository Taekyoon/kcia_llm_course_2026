"""슬라이드 재조립의 **배치 SSOT** — 새 3덱의 장 순서를 여기에만 적는다.

`tools/layout.py`(노트북 배치)와 대칭이다. 근거:
- 재배치 결정: `review/06_구성재설계.md` §5·§7 (2026-08-25 확정)
- 3일차 순서만 예외 — RAG 수미상관 재편(2026-08-27 사용자 확정)이 §5 를 대체한다:
  RAG → 평가 → 퓨샷 → SFT → DPO → GRPO → RAG개선
- 시간 배분: `review/09_시간배분표.md`
- 원본 구간 경계: extracted/ppt_*.md 실사 (2026-08-27)

원본 3덱은 불가침(CLAUDE.md §1). 출력은 `work/ppt/N일차 강의자료.pptx`.

표기:
- `ref("1일차", 63)`            원본 1일차 p63 한 장
- `refs("3일차", 19, 53)`       원본 3일차 p19~p53 (양끝 포함)
- `NewSlide("toc1", ...)`       신규 텍스트 초안 장. 본문은 migrate_slides.py 의
                                NEW_SLIDES 레지스트리가 채운다 — 여기는 자리만.

중복 제거(review/02 C-6)는 배치에서 뺀다: 3일차 p26·p27(p25 의 3중복),
3일차 p62(p61 중복), 2일차 p78(p77 중복).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_PPT = ROOT / "ppt"                 # 원본 — 읽기 전용
WORK_PPT = ROOT / "work" / "ppt"       # 산출물 — 매번 새로 씀

DECKS = {
    "1일차": SRC_PPT / "1일차 강의자료.pptx",
    "2일차": SRC_PPT / "2일차 강의자료.pptx",
    "3일차": SRC_PPT / "3일차 강의자료.pptx",
}

# 원본 총 장수. 빌드 시작 시 실측과 대조해 원본 훼손·교체를 감지한다.
DECK_PAGES = {"1일차": 91, "2일차": 90, "3일차": 83}


@dataclass(frozen=True)
class SlideRef:
    """원본 덱의 슬라이드 한 장. page 는 1-기준 (extract_pptx 의 pN 과 동일)."""
    deck: str
    page: int


@dataclass(frozen=True)
class NewSlide:
    """신규 텍스트 초안 장. 본문(제목·불릿)은 migrate_slides.NEW_SLIDES[key]."""
    key: str
    memo: str        # 이 자리에 무엇이 들어가는지 (배치표만 봐도 알도록)


def ref(deck: str, page: int) -> SlideRef:
    if not 1 <= page <= DECK_PAGES[deck]:
        raise SystemExit(f"[중단] {deck} p{page} 는 존재하지 않는다 (1~{DECK_PAGES[deck]})")
    return SlideRef(deck, page)


def refs(deck: str, first: int, last: int, *, skip: tuple[int, ...] = ()) -> list[SlideRef]:
    """p{first}~p{last} 를 순서대로. skip 은 중복 제거 등으로 빼는 페이지."""
    return [ref(deck, p) for p in range(first, last + 1) if p not in skip]


# ---------------------------------------------------------------------------
# 새 3덱 구성
# ---------------------------------------------------------------------------
# 신규 장수 계획(review/09 · 계획 Phase F): ④4 ⑦5 ⑧10 ⑬4 프롬프트최적화4
# 방법선택가이드7 RAG개선2 목차3 = 39장. 여기서는 구간당 NewSlide 마커 1개로 두고,
# 실제 장수는 NEW_SLIDES 레지스트리가 여러 장을 돌려줄 수 있게 한다 (1 마커 = 1구간).

PLACEMENT: dict[str, list] = {
    # ── 새 1일차: 1단원(3H) + 2단원(4H) ──────────────────────────
    "1일차": [
        ref("1일차", 1),                       # 표지
        NewSlide("toc1", "새 목차 — 1·2단원 구성"),
        ref("1일차", 3),                       # 간지: 1. AI 소프트웨어 개발 개론
        *refs("1일차", 4, 20),                 # 1단원 본문 (17)
        ref("1일차", 48),                      # 간지 재활용 → "2. 트랜스포머와 ChatGPT" 로 치환
        NewSlide("ml2dl", "④ ML→DL 발전사 (신규 ~4장. 착수 전 p49-53 그림 중복 확인)"),
        *refs("1일차", 49, 55),                # 트랜스포머 이론 (7)
        *refs("1일차", 56, 58),                # BERT (3)
        *refs("1일차", 59, 61),                # GPT·ChatGPT (3)
        ref("1일차", 21),                      # 간지: 자연어처리 머신러닝 소개 (⑤ 실습 도입)
        *refs("1일차", 22, 26),                # 챗봇 NLP 사례 (5)
        *refs("1일차", 27, 36),                # 분류 실습 (10) — mmBERT 로 치환
        *refs("1일차", 37, 47),                # NER 실습 (11) — mmBERT 로 치환
        ref("1일차", 91),                      # 감사합니다
    ],

    # ── 새 2일차: 3단원 Pre-training (7H) ────────────────────────
    #    노트북 순서와 동일: 데이터처리 → MiniGPT(+CPT) → Amazon → 프롬프트최적화
    "2일차": [
        ref("2일차", 1),                       # 표지 (원래 2일차 표지 — "-2일차-" 그대로)
        NewSlide("toc2", "새 목차 — 3단원 구성"),
        NewSlide("pretrain_intro", "⑦ Pre-training 등장 배경과 목적 (신규 ~5장)"),
        NewSlide("datacleaning", "⑧ 데이터 처리 이론 (신규 ~10장 — 노트북 마크다운 압축)"),
        ref("1일차", 62),                      # 간지: 미니 GPT 만들기
        *refs("1일차", 63, 90),                # 미니GPT (28) — PyTorch+HF 로 전면 치환
        ref("3일차", 3),                       # 간지: 도메인 최적화 프리트레인 (CPT)
        *refs("3일차", 4, 16),                 # CPT (13)
        ref("3일차", 17),                      # 간지: vLLM 데이터처리 실습 (Amazon)
        *refs("3일차", 18, 53, skip=(26, 27)), # Amazon 개요+7단계 (34) — p25 3중복 정리
        NewSlide("promptopt", "프롬프트 자동 최적화 안내 (신규 ~4장 — 3막 구조)"),
        ref("2일차", 90),                      # 감사합니다
    ],

    # ── 새 3일차: 4단원 Post-training (7H) — RAG 수미상관 ─────────
    "3일차": [
        ref("3일차", 1),                       # 표지
        NewSlide("toc3", "새 목차 — 4단원 구성 (RAG 수미상관)"),
        ref("3일차", 54),                      # 간지: Llama Index RAG 실습
        ref("3일차", 55),                      # RAG 개요
        *refs("3일차", 56, 72, skip=(62,)),    # RAG (16) — p61 중복 정리
        ref("3일차", 73),                      # 간지: 지속적인 LLM 챗봇 개발 (평가)
        *refs("3일차", 74, 82),                # 평가: 정의+평가 (9)
        ref("2일차", 3),                       # 간지: 퓨샷 러닝
        *refs("2일차", 4, 17),                 # 퓨샷 (14)
        ref("2일차", 18),                      # 간지: 포스트 트레이닝
        *refs("2일차", 19, 28),                # 포스트 트레이닝 소개 (10)
        ref("2일차", 29),                      # 간지: 인스트럭션 모델 학습
        NewSlide("lora", "⑬ LoRA 이론 (신규 ~4장 — SFT 학습 대기 중 진행)"),
        *refs("2일차", 30, 51),                # SFT (22)
        ref("2일차", 52),                      # 간지: 선호기반 모델 학습
        *refs("2일차", 53, 69),                # DPO (17)
        ref("2일차", 70),                      # 간지: 리즈닝 모델 학습
        *refs("2일차", 71, 89, skip=(78,)),    # GRPO (18) — p77 중복 정리
        NewSlide("method_guide", "방법 선택 가이드 (신규 ~7장 — review/07 원고)"),
        NewSlide("ragcheck", "RAG 개선 확인 — 수미상관 마무리 (신규 ~2장)"),
        ref("3일차", 83),                      # 감사합니다
    ],
}


def out_path(day: str) -> Path:
    return WORK_PPT / f"{day} 강의자료.pptx"


def summarize() -> None:
    total_ref, total_new = 0, 0
    for day, items in PLACEMENT.items():
        n_ref = sum(1 for i in items if isinstance(i, SlideRef))
        n_new = sum(1 for i in items if isinstance(i, NewSlide))
        moved = sum(1 for i in items if isinstance(i, SlideRef) and i.deck != day)
        total_ref += n_ref
        total_new += n_new
        print(f"{day}: 원본 재사용 {n_ref}장 (타 일자에서 {moved}장) + 신규 구간 {n_new}곳")
        for i in items:
            if isinstance(i, NewSlide):
                print(f"    [신규 {i.key}] {i.memo}")
    used = {(i.deck, i.page) for items in PLACEMENT.values()
            for i in items if isinstance(i, SlideRef)}
    dropped = [(d, p) for d, n in DECK_PAGES.items()
               for p in range(1, n + 1) if (d, p) not in used]
    print(f"\n원본 264장 중 재사용 {total_ref}장 · 미사용 {len(dropped)}장 · 신규 구간 {total_new}곳")
    print("미사용 (목차·중복·타 표지 등):")
    for d, p in dropped:
        print(f"  {d} p{p}")


if __name__ == "__main__":
    summarize()
