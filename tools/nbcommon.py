"""여러 노트북이 **글자 하나까지 똑같이** 써야 하는 셀 소스를 모아둔다.

왜 모아두는가
-------------
2일차 노트북들은 앞 단계가 만든 파일을 뒤 단계가 읽는다.

    데이터처리 → ko_wiki_clean.jsonl      → MiniGPT 이어학습(CPT)
    Amazon    → amazon_ko_sft.mine.jsonl → 3일차 SFT

그런데 Jupyter 도 nbconvert 도 **커널 cwd 를 노트북이 있는 디렉터리**로 잡는다.
각 노트북이 자기 식대로 경로를 계산하면 쓰는 쪽과 읽는 쪽이 다른 곳을 본다.
그것도 조용히 — 파일이 없으면 "아직 안 만들었나 보다" 로 보여서 원인을 찾기 어렵다.

그래서 경로를 정하는 코드는 여기 한 벌만 두고 전부 이것을 붙여 쓴다.
"""

from __future__ import annotations

# 데이터 경로 확정. 파일을 주고받는 모든 노트북이 이 셀을 그대로 갖는다.
DATA_DIR_CODE = '''
import os
from pathlib import Path


def _repo_root():
    """cwd 에서 위로 올라가며 repo 루트를 찾는다. setup_vessl.sh 를 표지로 쓴다."""
    p = Path.cwd().resolve()
    return next((c for c in [p, *p.parents] if (c / "setup_vessl.sh").exists()), None)


_root = _repo_root()
if "HPC_DATA" in os.environ:
    DATA_DIR = Path(os.environ["HPC_DATA"])          # setup_vessl.sh 가 절대경로로 지정
elif _root is not None:
    DATA_DIR = _root / "data"
else:
    # 조용히 현재 폴더로 폴백하지 않는다.
    # 엉뚱한 곳에 써두면 다음 노트북이 "파일이 없다" 고만 말해서 원인을 못 찾는다.
    raise RuntimeError(
        "데이터 경로를 정할 수 없습니다.\\n"
        "  repo 를 통째로 clone 했는지 확인하거나, 터미널에서\\n"
        "    export HPC_DATA=/절대/경로\\n"
        "  를 지정한 뒤 커널을 재시작하세요."
    )

# assets/ 는 강사가 미리 만들어 배포하는 것(커밋됨), data/ 는 실행하면 생기는 것.
ASSETS_DIR = (_root / "assets") if _root else DATA_DIR
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"DATA_DIR   = {DATA_DIR}")
print(f"ASSETS_DIR = {ASSETS_DIR}")
'''.strip("\n")

DATA_DIR_MD = """
### 파일을 어디에 둘 것인가

이 노트북이 만든 결과를 **다음 노트북이 읽습니다.** 그런데 Jupyter 는 노트북이 있는
폴더를 작업 디렉터리로 잡기 때문에, `"data/파일.jsonl"` 처럼 상대경로를 쓰면
노트북마다 서로 다른 곳을 보게 됩니다.

그래서 경로를 한 곳으로 고정합니다. `setup_vessl.sh` 가 `HPC_DATA` 를 지정해 두었고,
없으면 repo 루트를 찾아 `data/` 를 씁니다. **둘 다 안 되면 그 자리에서 멈춥니다** —
엉뚱한 곳에 저장해두고 나중에 "파일이 없다"고 헤매는 것보다 낫습니다.
""".strip("\n")


def jsonl_save_code(var: str, filename: str, fields: str) -> str:
    """list[dict] 를 JSONL 로 저장하는 셀 소스를 만든다.

    ensure_ascii=False 를 반드시 준다. 기본값 True 로 쓰면 한글이 전부
    \\uXXXX 로 escape 되어 파일 크기가 3배가 되고 눈으로 확인할 수 없다.
    """
    return f'''
import json

out_path = DATA_DIR / "{filename}"
with out_path.open("w", encoding="utf-8") as f:
    for d in {var}:
        # ensure_ascii=False 가 없으면 한글이 \\\\uXXXX 로 escape 되어
        # 파일이 3배로 커지고 눈으로 확인할 수 없게 된다.
        f.write(json.dumps({{{fields}}}, ensure_ascii=False) + "\\n")

size_mb = out_path.stat().st_size / 1024 / 1024
print(f"저장: {{out_path}}")
print(f"  {{len({var}):,}}건 · {{size_mb:.1f}} MB")
print()
print("첫 줄:")
print(f"  {{out_path.open(encoding='utf-8').readline()[:200]}}")
'''.strip("\n")
