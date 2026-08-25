"""마이그레이션 전역 치환 대상의 실제 출현 횟수를 센다.

migrate_notebooks.py 의 GlobalRule.expect 값을 추정으로 적으면 안 되므로,
실제 횟수를 먼저 세어 확인하는 용도. 자료가 바뀌면 다시 돌려 값을 갱신한다.
"""

from __future__ import annotations

import json
from pathlib import Path

WORK = Path(__file__).resolve().parent.parent / "work" / "notebook"

EXAONE = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"

TARGETS: dict[str, list[str]] = {
    "2일차/HPC_퓨샷실습.ipynb": [
        EXAONE,
        'extra_body={"guided_choice": ["긍정", "부정"]},',
        'extra_body={"guided_json": json_schema},',
    ],
    "3일차/HPC_Amazon요약실습.ipynb": [
        EXAONE,
        "completion = client.beta.chat.completions.parse(",
        'extra_body={"guided_json": feature_type_schema},',
        'extra_body={"guided_json": subsectoin_schema},',
        'extra_body={"guided_json": extracted_feature_schema},',
        'extra_body={"guided_json": consumer_category_schema},',
        'extra_body={"guided_json": summary_schema},',
    ],
    "3일차/HPC_BM25_RAG실습.ipynb": [
        EXAONE,
    ],
}


def main() -> None:
    for rel, pats in TARGETS.items():
        nb = json.loads((WORK / rel).read_text(encoding="utf-8"))
        joined = "".join("".join(c.get("source", "")) for c in nb["cells"])
        print(f"\n=== {rel} ===")
        for p in pats:
            n = joined.count(p)
            mark = " " if n else "  ← 없음!"
            print(f"  {n:>3}회{mark}  {p[:72]}")


if __name__ == "__main__":
    main()
