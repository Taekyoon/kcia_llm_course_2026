"""Amazon 요약 노트북의 JSON 키 정합성 확인.

프롬프트를 한국어화할 때 **출력 예시의 JSON 키까지 한국어로 바꿔버리면**
다운스트림 코드가 조용히 깨진다. 7단계가 사슬로 엮여 있어서
Step2 출력 → json.loads() → Step6 입력 식으로 이어지기 때문이다.

이 스크립트는 두 집합을 뽑아 비교한다.
  (A) 프롬프트 안 출력 예시에 등장하는 JSON 키
  (B) 코드가 실제로 인덱싱하는 키  (obj["key"] / .get("key"))

(B) 에는 있는데 (A) 에는 없으면 → 모델이 그 키를 만들어줄 근거가 없다. 위험.
"""

from __future__ import annotations

import json
import re

from layout import work_path

NB = work_path("HPC_Amazon요약실습.ipynb")   # 일자 배치는 tools/layout.py 가 정한다

nb = json.loads(NB.read_text(encoding="utf-8"))
cells = ["".join(c.get("source", "")) for c in nb["cells"]]

# ── (A) 프롬프트 문자열 안의 JSON 키 ──────────────────────────
# 프롬프트는 삼중따옴표 안에 있다. 그 안에서 "key": 패턴을 찾는다.
prompt_keys: dict[str, list[int]] = {}
for i, s in enumerate(cells, start=1):
    for m in re.finditer(r'(\w+_PROMPT)\s*=\s*"""(.*?)"""', s, re.S):
        for k in re.findall(r'"([^"\n]{1,40})"\s*:', m.group(2)):
            prompt_keys.setdefault(k, []).append(i)

# ── (B) 코드가 인덱싱하는 키 ──────────────────────────────────
# 프롬프트 본문(삼중따옴표)은 지운 뒤에 찾아야 프롬프트 예시가 섞이지 않는다.
code_keys: dict[str, list[int]] = {}
for i, s in enumerate(cells, start=1):
    stripped = re.sub(r'"""(.*?)"""', '""', s, flags=re.S)
    for k in re.findall(r'\[\s*"([^"\n]{1,40})"\s*\]', stripped):
        code_keys.setdefault(k, []).append(i)
    for k in re.findall(r'\.get\(\s*"([^"\n]{1,40})"', stripped):
        code_keys.setdefault(k, []).append(i)


def show(title: str, d: dict[str, list[int]]) -> None:
    print("=" * 64)
    print(title)
    print("=" * 64)
    for k in sorted(d):
        cs = sorted(set(d[k]))
        han = "한글" if re.search(r"[가-힣]", k) else "    "
        print(f"  {han}  {k:<28} 셀 {cs}")
    print()


show("(A) 프롬프트 출력 예시의 JSON 키", prompt_keys)
show("(B) 코드가 인덱싱하는 키", code_keys)

# 코드는 인덱싱하는데 프롬프트가 안 알려주는 키
danger = {k: v for k, v in code_keys.items() if k not in prompt_keys}
# 판다스 컬럼/딕셔너리 등 JSON 과 무관한 것이 섞이므로 참고용으로만 출력한다.
print("=" * 64)
print("★ 코드는 쓰는데 프롬프트 예시에 없는 키 (JSON 관련이면 위험)")
print("=" * 64)
for k in sorted(danger):
    print(f"     {k:<28} 셀 {sorted(set(danger[k]))}")
print()

han_keys = [k for k in prompt_keys if re.search(r"[가-힣]", k)]
if han_keys:
    print("★ 프롬프트 JSON 키에 한글이 섞였습니다:", han_keys)
    print("  키는 영어로 두는 편이 안전합니다(코드가 영어 키로 인덱싱).")
else:
    print("→ 프롬프트 JSON 키는 전부 영어. 코드와 어긋날 위험 없음.")
