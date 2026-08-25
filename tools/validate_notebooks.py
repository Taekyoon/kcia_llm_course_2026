"""work/notebook/ 의 노트북을 실행 전에 정적 검사한다.

    uv run python tools/validate_notebooks.py

VESSL 에서 GPU 를 붙잡고 nbconvert 로 돌리기 전에 로컬에서 걸러낼 수 있는 것들:
  1. JSON 구조가 노트북 규격에 맞는가
  2. 코드 셀이 파이썬으로 파싱되는가 (문법 오류)
  3. 셀 순서상 정의되기 전에 쓰이는 이름이 있는가 (얕은 검사)

3번은 완벽하지 않다. 동적으로 만들어지는 이름을 못 잡고, 조건부 정의도 못 본다.
그래도 "셀을 자르다가 정의를 날린" 류의 실수는 잡힌다.

실행 검증(verify/06_노트북실행.sh)을 대체하지 않는다. 그 앞단에서 값싸게 거르는 용도다.
"""

from __future__ import annotations

import ast
import builtins
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work" / "notebook"

# 노트북에서 흔히 쓰는데 정적으로는 안 보이는 이름들
NB_BUILTINS = {"get_ipython", "display", "In", "Out", "_", "__"}


def strip_magics(src: str) -> str:
    """%pip, !ls 같은 IPython 매직/셸 호출을 걷어낸다. ast 가 못 읽는다.

    백슬래시로 이어지는 여러 줄 매직도 함께 삼킨다. 그러지 않으면 이어지는 줄이
    들여쓰기된 코드로 남아 unexpected indent 오탐이 난다.
    """
    out: list[str] = []
    lines = src.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.lstrip()
        if s.startswith(("%", "!")) or s.startswith("?") or s.endswith("?"):
            out.append("pass  # (magic)")
            while lines[i].rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                out.append("")
        else:
            out.append(line)
        i += 1
    return "\n".join(out)


def collect_bound(tree: ast.AST) -> set[str]:
    """이 셀에서 새로 정의되는 이름을 모은다."""
    names: set[str] = set()
    for node in ast.walk(tree):
        # 함수/람다의 인자는 별도 if 로 처리한다.
        # elif 체인에 넣으면 FunctionDef 가 위쪽 분기에 먼저 걸려
        # 인자를 영영 수집하지 못한다 (실제로 그 버그를 냈다).
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = node.args
            for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs):
                names.add(arg.arg)
            if a.vararg:
                names.add(a.vararg.arg)
            if a.kwarg:
                names.add(a.kwarg.arg)

        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            tgt = node.target
            for n in ast.walk(tgt):
                if isinstance(n, ast.Name):
                    names.add(n.id)
        elif isinstance(node, ast.With) or isinstance(node, ast.AsyncWith):
            for item in node.items:
                if item.optional_vars:
                    for n in ast.walk(item.optional_vars):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
        elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
            names.update(node.names)
    return names


def collect_used(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def check(path: Path) -> list[str]:
    problems: list[str] = []
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return [f"JSON 파싱 실패: {e}"]

    if "cells" not in nb or not isinstance(nb["cells"], list):
        return ["'cells' 키가 없거나 리스트가 아닙니다"]

    known = set(dir(builtins)) | NB_BUILTINS
    for idx, cell in enumerate(nb["cells"], start=1):
        kind = cell.get("cell_type")
        if kind not in ("code", "markdown", "raw"):
            problems.append(f"셀 {idx}: 알 수 없는 cell_type={kind!r}")
            continue
        if kind != "code":
            continue

        src = cell.get("source", "")
        src = "".join(src) if isinstance(src, list) else src
        if not src.strip():
            continue

        cleaned = strip_magics(src)
        try:
            tree = ast.parse(cleaned)
        except SyntaxError as e:
            head = src.strip().splitlines()[0][:70]
            problems.append(
                f"셀 {idx}: 문법 오류 (line {e.lineno}) {e.msg}\n"
                f"            첫 줄: {head}"
            )
            continue

        used = collect_used(tree)
        bound = collect_bound(tree)
        missing = sorted(used - known - bound)
        if missing:
            problems.append(f"셀 {idx}: 정의 전 사용 의심 → {', '.join(missing[:8])}")
        known |= bound

    return problems


def main() -> None:
    files = sorted(WORK.rglob("*.ipynb"))
    if not files:
        raise SystemExit(f"노트북이 없습니다: {WORK}")

    print("=" * 72)
    print(f"노트북 정적 검사 — {len(files)}종")
    print("=" * 72)

    bad = 0
    for f in files:
        rel = f.relative_to(WORK)
        problems = check(f)
        n_cells = len(json.loads(f.read_text(encoding="utf-8"))["cells"])
        if problems:
            bad += 1
            print(f"\n❌ {rel}  ({n_cells}셀)")
            for p in problems:
                print(f"   · {p}")
        else:
            print(f"✅ {rel}  ({n_cells}셀)")

    print()
    print("=" * 72)
    print(f"이상 {bad}종 / 전체 {len(files)}종")
    print("※ '정의 전 사용 의심' 은 오탐이 있습니다. 동적 생성이나 조건부 정의는 못 봅니다.")
    print("   문법 오류는 확실한 문제입니다.")
    print("=" * 72)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
