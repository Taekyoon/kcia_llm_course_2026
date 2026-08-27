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
        # 원본 p63-90(28장)은 영어 simplebooks + tf.data + keras-nlp 전면이라
        # 치환이 불가능 — 현행 PyTorch 노트북 기준 신규 18장으로 교체 (E-3)
        NewSlide("minigpt", "미니GPT — 현행 노트북(PyTorch+HF·한국어 동화·CPT) 기준 신규 ~18장"),
        ref("3일차", 3),                       # 간지: 도메인 최적화 프리트레인 (CPT)
        *refs("3일차", 4, 16),                 # CPT (13)
        ref("3일차", 17),                      # 간지: vLLM 데이터처리 실습 (Amazon)
        *refs("3일차", 18, 53, skip=(26, 27)), # Amazon 개요+7단계 (34) — p25 3중복 정리
        NewSlide("promptopt", "프롬프트 자동 최적화 안내 (신규 ~4장 — 3막 구조)"),
        ref("2일차", 90),                      # 감사합니다
    ],

    # ── 새 3일차: 4단원 Post-training (7H) — RAG 수미상관 ─────────
    #    E-3 재구성(2026-08-27): 평가 구간은 노트북과 겹침이 p80 한 장뿐이라
    #    사실상 재작성(신규 6장), SFT 의 KoAlpaca 예시 4장(p34·35·41·42)은 드롭,
    #    각 실습 구간에 노트북 핵심(이어받기·Group Relative 등) 신규 장을 삽입.
    "3일차": [
        ref("3일차", 1),                       # 표지
        NewSlide("toc3", "새 목차 — 4단원 구성 (RAG 수미상관)"),
        # RAG
        ref("3일차", 54),                      # 간지
        ref("3일차", 55),                      # 개요
        *refs("3일차", 56, 58),                # 환경·데이터
        NewSlide("rag_bm25", "BM25 vs 임베딩 · 청킹 (노트북 핵심 개념)"),
        *refs("3일차", 59, 65, skip=(62,)),    # 리트리버·검색·Q&A (p61 중복 정리)
        NewSlide("rag_prompt", "프롬프트 교체의 함정 · 검색/생성 실패 진단"),
        *refs("3일차", 66, 72),                # Q&A · 서브질문
        # 평가 — 재작성 (원본 p74-79·81·82 는 옛 챗봇 기획론이라 제외)
        ref("3일차", 73),                      # 간지 → "2. 태스크 정의와 평가"
        NewSlide("evalsec_a", "태스크 정의 · BLEU/ROUGE (신규 2장)"),
        ref("3일차", 80),                      # 수치화 개요 (유일한 겹침 장)
        NewSlide("evalsec_b", "자동 지표의 한계 · LLM-as-judge · 정리 (신규 4장)"),
        # 퓨샷
        ref("2일차", 3),
        *refs("2일차", 4, 17),
        NewSlide("fewshot_extra", "형식 강제 3단계 · 흔들림 · 결론 비교 · 스키마 추출 (신규 4장)"),
        # 포스트 트레이닝
        ref("2일차", 18),
        *refs("2일차", 19, 28),
        # SFT (+⑬ LoRA) — KoAlpaca 예시 p34·35·41·42 드롭
        ref("2일차", 29),
        NewSlide("lora", "⑬ LoRA 이론 (신규 ~4장 — SFT 학습 대기 중 진행)"),
        *refs("2일차", 30, 31),
        NewSlide("sft_data", "어제 만든 877건 — 데이터 흐름 (신규 1장)"),
        *refs("2일차", 32, 33),
        *refs("2일차", 36, 37),
        NewSlide("sft_tmpl", "Base 모델에는 대화 형식이 없다 (신규 1장)"),
        *refs("2일차", 38, 40),
        NewSlide("sft_len", "토큰 길이 확인 — 조용한 실패 (신규 1장)"),
        *refs("2일차", 43, 51),
        # DPO
        ref("2일차", 52),
        *refs("2일차", 53, 63),
        NewSlide("dpo_resume", "SFT 이어받기와 참조 모델 (신규 1장)"),
        *refs("2일차", 64, 65),
        NewSlide("dpo_logs", "rewards 로그 읽는 법 · ORPO (신규 1장)"),
        *refs("2일차", 66, 69),
        # GRPO
        ref("2일차", 70),
        *refs("2일차", 71, 77),                # p78 중복 정리
        *refs("2일차", 79, 82),
        NewSlide("grpo_reward", "채점 함수의 품질이 곧 학습의 품질 (신규 1장)"),
        *refs("2일차", 83, 84),
        NewSlide("grpo_group", "Group Relative — num_generations 의 뜻 (신규 1장)"),
        *refs("2일차", 85, 89),
        # 마무리
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
