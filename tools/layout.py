"""실습 노트북의 **일자 배치**를 한 곳에서 정의한다.

왜 따로 두는가
--------------
노트북을 만드는 주체가 넷이다 — `migrate_notebooks.py` (원본 8종 변환) 와
`build_*_notebook.py` 3종 (신규 작성). 각자 자기 출력 경로를 들고 있으면
일자를 재배치할 때 네 군데를 고쳐야 하고, 하나를 빠뜨리면 **옛 위치와 새 위치에
같은 노트북이 둘 남는다.** 수강생은 어느 쪽이 최신인지 알 수 없다.

그래서 배치는 여기에만 적고, 나머지는 전부 여기를 참조한다.

배치 근거
---------
HWP 계획서의 단원별 시수(3/4/7/7H)에 일자를 맞춘 결과다
(review/06_구성재설계.md §7 에서 확정).

    1일차 = 1단원(AI SW 개론) + 2단원(트랜스포머와 ChatGPT)
    2일차 = 3단원(LLM Pre-training)
    3일차 = 4단원(LLM Post-training)

폴더명이 강의 일자와 어긋나 있으면 "1일차 폴더인데 2일차에 해요" 를 매번 안내해야 하고,
노트북 사이에 파일을 주고받을 때 경로가 강의 순서를 거스르는 것처럼 보인다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "notebook"            # 원본 — 읽기 전용 (CLAUDE.md §1)
WORK = ROOT / "work" / "notebook"  # 산출물 — 매번 새로 씀

# 일자 안의 순서가 곧 **강의 진행 순서**다. 파일 의존도 이 순서를 따른다.
#   2일차: (데이터처리 → MiniGPT) → (Amazon → 프롬프트최적화)
#          내용상 독립인 두 갈래라 갈래별로 묶는다. 정제 산출물을 MiniGPT 가 받고,
#          Amazon·프롬프트최적화 산출물은 3일차 SFT 가 받는다.
LAYOUT: dict[str, list[str]] = {
    "1일차": [                       # 2단원 ⑤ — 인코더 모델 실습
        "HPC_Classification실습.ipynb",
        "HPC_NER실습.ipynb",
    ],
    "2일차": [                       # 3단원 ⑧⑨⑩ — 갈래별로 묶는다
        # 갈래 A. 사전학습 축: 정제한 데이터를 바로 이어학습에 쓴다
        "HPC_데이터처리실습.ipynb",      # ⑧ 정제
        "HPC_MiniGPT실습.ipynb",       # ⑨ 사전학습 + ⑩ CPT (위 산출물을 받는다)
        # 갈래 B. 생성 데이터 축: 여기서 만든 데이터가 3일차 SFT 로 간다
        #         vLLM 서버는 이 구간에서만 필요하다 (학습과 GPU 를 다투지 않게)
        "HPC_Amazon요약실습.ipynb",     # ⑧ 구조화 추출
        "HPC_프롬프트최적화실습.ipynb",   # ⑧ 번역 → judge → 자동 최적화
    ],
    "3일차": [                       # 4단원 ⑪⑫⑬⑭ — RAG 수미상관 구조
        # 아침: 시스템을 먼저 만든다. 서버(vLLM 4B)는 앞 세 실습이 연달아 쓴다
        "HPC_BM25_RAG실습.ipynb",       # 심화 → 선두로. 하루의 축이 되는 시스템
        "HPC_평가실습.ipynb",           # ⑪ 태스크 정의와 평가
        "HPC_퓨샷실습.ipynb",           # ⑫ Few-shot
        # 오후: 모델 개선 (서버 내리고 학습)
        "HPC_SFT실습.ipynb",           # ⑬ 파인튜닝과 LoRA
        "HPC_DPO실습.ipynb",           # ⑭ Preference Learning
        "HPC_GRPO실습.ipynb",          # ⑭ GRPO
        # 마무리: 아침의 인덱스에 학습 전/후 0.6B 를 꽂아 전후 비교 (서버 불필요)
        "HPC_RAG개선실습.ipynb",        # 신규 — 수미상관의 닫는 괄호
    ],
}

# 이름 → 일자. 역인덱스.
DAY_OF: dict[str, str] = {n: d for d, names in LAYOUT.items() for n in names}


def disk_name(name: str) -> str:
    """디스크에 저장되는 파일 이름 — **순번 접두사**가 붙는다.

    일자 안의 실습 순서가 파일 목록에서 바로 보이도록 `0_`, `1_` ... 을 붙인다
    (Jupyter 파일 목록은 이름순 정렬이라 접두사가 곧 진행 순서가 된다).
    코드에서는 계속 논리 이름("HPC_SFT실습.ipynb")으로 부르고,
    접두사는 여기서만 계산한다 — 순서를 바꿔도 LAYOUT 만 고치면 된다.
    """
    day = DAY_OF.get(name)
    if day is None:
        raise SystemExit(
            f"[중단] LAYOUT 에 없는 노트북입니다: {name!r}\n"
            f"  tools/layout.py 의 LAYOUT 에 먼저 추가하세요."
        )
    return f"{LAYOUT[day].index(name)}_{name}"


def logical_name(fname: str) -> str:
    """디스크 이름에서 순번 접두사를 걷어내 논리 이름으로 되돌린다."""
    head, sep, tail = fname.partition("_")
    if sep and head.isdigit():
        return tail
    return fname


def work_path(name: str) -> Path:
    """노트북 이름으로 산출물 경로를 얻는다. 배치에 없으면 중단한다."""
    return WORK / DAY_OF[name] / disk_name(name) if name in DAY_OF else Path(disk_name(name))


def find_source(name: str) -> Path:
    """원본 노트북을 찾는다. **세 일자 폴더를 전부 뒤진다.**

    원본(`notebook/`)의 일자 폴더는 25년 배치 그대로이고 불가침이다.
    산출물만 재배치했으므로, 출력 일자로 원본을 찾으면 어긋난다.
    이름이 중복되지 않으므로 전 폴더 탐색이 안전하다.

    이름 대조 시 공백과 확장자를 걷어낸다. 원본은 "HPC_DPO 실습.ipynb" 처럼
    공백이 있고 산출물은 없앴다. (원본은 원래 확장자가 없었으나 Jupyter 에서
    열리지 않아 .ipynb 를 붙였다 — 내용은 그대로다. 해시로 확인했다.)
    """
    want = name.removesuffix(".ipynb")
    hits = [
        p
        for day in LAYOUT
        for p in (SRC / day).iterdir()
        if (SRC / day).is_dir()
        and p.is_file()
        and p.name.replace(" ", "").removesuffix(".ipynb") == want
    ]
    if not hits:
        raise SystemExit(f"[중단] 원본을 찾지 못했습니다: {SRC} 어디에도 {want!r} 없음")
    if len(hits) > 1:
        raise SystemExit(
            f"[중단] 원본이 여러 곳에 있습니다: {[str(p) for p in hits]}\n"
            f"  이름이 중복되면 어느 것을 쓸지 정할 수 없습니다."
        )
    return hits[0]


def prune_stale(verbose: bool = True) -> list[Path]:
    """LAYOUT 이 정한 자리가 아닌 곳에 있는 노트북 사본을 지운다.

    일자를 재배치하면 옛 위치에 사본이 남는다. 그대로 두면 같은 노트북이 둘이 되어
    수강생이 어느 쪽을 열지 알 수 없다. `work/` 는 언제든 재생성 가능한 산출물이므로
    지우는 것이 안전하다 (원본 `notebook/` 은 건드리지 않는다).
    """
    removed = []
    if not WORK.exists():
        return removed
    for p in sorted(WORK.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in p.parts:
            continue
        base = logical_name(p.name)
        if base not in DAY_OF:
            # LAYOUT 에 없는 노트북. 사람이 넣어둔 것일 수 있으니 지우지 않고 알린다.
            if verbose:
                print(f"  [경고] LAYOUT 에 없는 노트북: {p.relative_to(WORK)} (그대로 둠)")
            continue
        want = work_path(base)
        if p != want:
            # 옛 위치이거나, 접두사가 없거나 순번이 바뀐 옛 이름이다.
            p.unlink()
            removed.append(p)
            if verbose:
                print(f"  [정리] 옛 사본 삭제: {p.relative_to(WORK)} → {want.relative_to(WORK)}")

    # 비어버린 일자 폴더는 정리한다 (LAYOUT 에 있는 일자는 남긴다)
    for d in sorted(WORK.iterdir()) if WORK.exists() else []:
        if d.is_dir() and d.name not in LAYOUT and not any(d.iterdir()):
            d.rmdir()
            if verbose:
                print(f"  [정리] 빈 폴더 삭제: {d.name}/")
    return removed


if __name__ == "__main__":
    print(f"원본  : {SRC}")
    print(f"산출물: {WORK}")
    print()
    for day, names in LAYOUT.items():
        print(f"{day}/")
        for n in names:
            exists = "✓" if work_path(n).exists() else "·"
            print(f"  {exists} {disk_name(n)}")
    print()
    print(f"총 {sum(len(v) for v in LAYOUT.values())}종")
