"""Amazon 요약 파이프라인을 대량으로 돌려 **강사 배포용 학습 데이터**를 만든다 (D-4).

    # vLLM 이 떠 있는 상태에서, repo 루트에서
    python tools/build_amazon_pregen.py --n 300
    python tools/build_amazon_pregen.py --n 300 --resume     # 중단됐으면 이어받기

왜 노트북이 아니라 스크립트인가
-------------------------------
수강생은 실습에서 **10건**만 돌린다. 상품 1개에 LLM 호출이 16번쯤 들어가서
300건이면 5,000번 가까이 된다. 강의 시간에 할 수 없는 분량이다.

그렇다고 노트북 안에 "300건 버전" 셀을 두면 수강생이 실수로 실행한다.
그래서 **강사가 수업 전에 한 번 돌리는 독립 스크립트**로 분리했다.

왜 정의를 노트북에서 가져오는가
-------------------------------
프롬프트·스키마·헬퍼를 여기에 복사해두면 **노트북과 조용히 어긋난다.**
노트북 프롬프트를 고쳤는데 배포 데이터는 옛 프롬프트로 만들어진 상태가 되고,
수강생이 만든 10건과 배포본 300건의 성격이 달라진다.

그래서 노트북을 **단일 진실 소스**로 삼고, 필요한 셀만 골라 실행해서
정의를 가져온다. `data.map(...)` 같은 실행 구문은 무해한 스텁으로 삼킨다.

산출물
------
    assets/amazon_ko_sft.jsonl.gz     커밋한다. 3일차 SFT 가 읽는다
    data/.pregen_raw.jsonl            중간 저장. gitignore 대상
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import ModuleType

from layout import ROOT, work_path

NB_PATH = work_path("HPC_Amazon요약실습.ipynb")
OUT_PATH = ROOT / "assets" / "amazon_ko_sft.jsonl.gz"
RAW_PATH = ROOT / "data" / ".pregen_raw.jsonl"   # 중간 저장 (gitignore)

# 노트북에서 가져올 정의들. 이 문자열이 든 코드 셀만 실행한다.
WANT = (
    "_PROMPT = ",
    "(BaseModel)",          # 클래스 정의만. `from pydantic import BaseModel` 은 제외
    "def gen_text_",
    "def organize_features",
    "def summarize_subsection",
    "def summarize_by_users",
    "def restore_split_names",
)
# 실행하면 안 되는 것 — 설치, 데이터 로드, 파일 저장
SKIP = ("%pip", "!pip", "load_dataset(", "to_csv", "out_path.open",
        "from datasets import")   # import 는 우리가 직접 넣는다


class _NoOpData:
    """`data = data.map(...)` 를 삼키는 스텁.

    셀 50~53 은 함수 정의와 `data.map` 호출이 한 셀에 섞여 있다.
    정의만 필요하므로 map 은 아무 일도 하지 않고 자기 자신을 돌려준다.
    """

    def map(self, *_a, **_k):
        return self

    def select_columns(self, *_a, **_k):
        return self


def load_notebook_defs(client, model_name: str) -> dict:
    """노트북에서 프롬프트·스키마·헬퍼를 실행해 네임스페이스로 돌려준다."""
    from typing import Annotated, List

    from pydantic import BaseModel, Field

    # ★ 평범한 dict 에 exec 하면 pydantic 이 List[FeatureType] 을 해석하지 못한다.
    #     PydanticUserError: `FeatureTypeList` is not fully defined
    #   pydantic 은 sys.modules[cls.__module__] 에서 타입을 찾으므로
    #   **진짜 모듈**을 만들어 그 __dict__ 에 실행해야 한다.
    mod = ModuleType("_amazon_nb_defs")
    sys.modules[mod.__name__] = mod

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns: dict = mod.__dict__
    ns.update({
        "json": json,
        "BaseModel": BaseModel,
        "Field": Field,
        "List": List,
        "Annotated": Annotated,
        "client": client,
        "model_name": model_name,
        "data": _NoOpData(),
        "output_data": [],      # 셀 60 의 변환 루프가 빈 채로 돌게 한다
    })

    used = []
    for i, cell in enumerate(nb["cells"], 1):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(bad in src for bad in SKIP):
            continue
        if not any(w in src for w in WANT):
            continue
        try:
            # 셀 안의 print 를 삼킨다. 정의만 필요한데 변환 셀이 빈 채로 돌면서
            # "학습 예시 0건" 을 찍어 실제 결과인 것처럼 보였다.
            with redirect_stdout(io.StringIO()):
                exec(src, ns)
        except Exception as e:
            raise SystemExit(
                f"[중단] 노트북 셀 {i} 를 실행할 수 없습니다: {type(e).__name__}: {e}\n"
                f"  노트북 구조가 바뀌었을 수 있습니다. 셀 내용:\n"
                f"  {src[:300]}"
            )
        used.append(i)

    need = ["gen_text_step_1", "gen_text_step_2", "gen_text_step_3", "gen_text_step_4",
            "gen_text_summary", "gen_text_featured_summary",
            "organize_features", "summarize_subsection", "summarize_by_users",
            "restore_split_names", "INSTRUCTION", "TEMPLATES", "MAX_SRC"]
    missing = [n for n in need if n not in ns]
    if missing:
        raise SystemExit(
            f"[중단] 노트북에서 다음을 찾지 못했습니다: {', '.join(missing)}\n"
            f"  실행한 셀: {used}\n"
            f"  노트북이 바뀌었다면 WANT 선택자를 고치세요."
        )
    print(f"  노트북에서 정의를 가져왔습니다 — 셀 {used}")
    return ns


def run_one(ns: dict, text: str) -> dict:
    """상품 하나에 7단계를 돌려 노트북과 같은 형태의 행을 만든다.

    조립 순서는 노트북 셀 50~53 과 동일하다.
    """
    row = {"text": text}

    # Step 1~4
    row["feature_type_list_str"] = ns["gen_text_step_1"](text)
    row["subsection_list_str"] = ns["gen_text_step_2"](row["feature_type_list_str"])
    row["extracted_feature_list_str"] = ns["gen_text_step_3"](
        "Provided Features:" + chr(10) * 2 + row["feature_type_list_str"]
        + chr(10) * 2 + text
    )
    row["user_category_list_str"] = ns["gen_text_step_4"](text)

    # Step 5 — 속성값을 그룹으로 묶는다
    row["feature_map"] = json.dumps({
        r["feature_type"]: r["value"]
        for r in json.loads(row["extracted_feature_list_str"])["feature_list"]
    })
    row.update(ns["organize_features"](row))

    # Step 6·7 — 그룹별 요약, 구매자 유형별 요약
    row.update(ns["summarize_subsection"](row))
    row.update(ns["summarize_by_users"](row))

    return {k: row[k] for k in ("text", "summary_by_users", "summary_by_subsection")}


def to_records(ns: dict, rows: list[dict]) -> list[dict]:
    """노트북의 변환 로직을 그대로 써서 instruction/output 을 만든다."""
    restore = ns["restore_split_names"]
    unwrap = ns["unwrap"]
    INSTRUCTION, TEMPLATES, MAX_SRC = ns["INSTRUCTION"], ns["TEMPLATES"], ns["MAX_SRC"]

    records, dropped = [], 0
    for row in rows:
        src = restore(row["text"])[:MAX_SRC]
        for col, tmpl in TEMPLATES:
            try:
                outer = json.loads(row[col])
            except (json.JSONDecodeError, TypeError):
                dropped += 1
                continue
            if not isinstance(outer, dict):
                dropped += 1
                continue
            for k, v in outer.items():
                try:
                    summary = unwrap(v)
                except (json.JSONDecodeError, TypeError):
                    summary = None
                if not summary:
                    dropped += 1
                    continue
                records.append({
                    "instruction": INSTRUCTION.format(task=tmpl.format(k=k), src=src),
                    "output": summary,
                })
    return records, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Amazon 사전생성 (강사용, 1회 실행)")
    ap.add_argument("--n", type=int, default=300, help="생성할 상품 수 (기본 300)")
    ap.add_argument("--workers", type=int, default=4,
                    help="동시 처리 상품 수. vLLM 이 배치로 처리하므로 4~8 이 적당")
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--resume", action="store_true",
                    help="중간 저장에서 이어받는다")
    args = ap.parse_args()

    from datasets import load_dataset
    from openai import OpenAI

    print("=" * 70)
    print(" Amazon 사전생성 — 강사가 수업 전에 한 번만 돌립니다")
    print("=" * 70)

    client = OpenAI(api_key="EMPTY", base_url=args.base, timeout=180.0)
    try:
        client.models.list()
    except Exception as e:
        raise SystemExit(
            f"[중단] vLLM 서버에 붙지 못했습니다: {type(e).__name__}\n"
            f"  source /opt/vllm-env/bin/activate\n"
            f"  nohup vllm serve {args.model} --port 8000 \\\n"
            f"      --gpu-memory-utilization 0.80 --max-model-len 16384 > /tmp/vllm.log 2>&1 &"
        )
    print(f"  vLLM 연결됨 — {args.model}")

    ns = load_notebook_defs(client, args.model)

    raw = load_dataset("Taekyoon/test_amazon", split=f"train[:{args.n}]")
    texts = [r["text"] for r in raw]
    print(f"  대상 {len(texts)}건")

    # ── 중간 저장 이어받기 ────────────────────────────────────────
    done: dict[int, dict] = {}
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and RAW_PATH.exists():
        for line in RAW_PATH.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["i"]] = rec["row"]
        print(f"  이어받기 — 이미 끝난 {len(done)}건은 건너뜁니다")
    elif RAW_PATH.exists():
        RAW_PATH.unlink()

    todo = [i for i in range(len(texts)) if i not in done]
    print(f"  이번에 처리할 것 {len(todo)}건 · 동시 {args.workers}개")
    print()

    t0 = time.time()
    failed = 0
    raw_f = RAW_PATH.open("a", encoding="utf-8")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, ns, texts[i]): i for i in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            try:
                row = fut.result()
                done[i] = row
                print(json.dumps({"i": i, "row": row}, ensure_ascii=False), file=raw_f)
                raw_f.flush()          # 중간에 죽어도 여기까지는 남는다
            except Exception as e:
                failed += 1
                print(f"  [{i}] 제외 — {type(e).__name__}: {str(e)[:80]}")
            if n % 10 == 0 or n == len(todo):
                el = time.time() - t0
                rate = n / el if el else 0
                left = (len(todo) - n) / rate if rate else 0
                print(f"  {n}/{len(todo)}  ({el/60:.1f}분 경과 · 남은 예상 {left/60:.0f}분)")
    raw_f.close()

    print()
    print(f"생성 완료 {len(done)}건 · 제외 {failed}건 · {(time.time()-t0)/60:.1f}분")

    # ── 학습 데이터로 변환 ────────────────────────────────────────
    rows = [done[i] for i in sorted(done)]
    records, dropped = to_records(ns, rows)
    print(f"학습 예시 {len(records)}건 · 변환 제외 {dropped}건")
    if not records:
        raise SystemExit("[중단] 학습 예시가 하나도 만들어지지 않았습니다.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        for r in records:
            print(json.dumps(r, ensure_ascii=False), file=f)

    size = OUT_PATH.stat().st_size / 1024
    print()
    print(f"저장: {OUT_PATH}  ({size:,.0f} KB)")
    print()
    print("  이 파일을 커밋하세요. 3일차 SFT 실습이 읽습니다.")
    print("  중간 저장은 지워도 됩니다:")
    print(f"    rm {RAW_PATH}")


if __name__ == "__main__":
    if not NB_PATH.exists():
        raise SystemExit(f"[중단] 노트북이 없습니다: {NB_PATH}\n"
                         f"  uv run python tools/migrate_notebooks.py 를 먼저 실행하세요.")
    main()
