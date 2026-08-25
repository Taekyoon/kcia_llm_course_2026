"""데이터처리 실습의 MinHash 구현이 맞는지 로컬에서 검증한다.

    uv run --with numpy python tools/check_minhash.py

VESSL 로 보내기 전에 확인할 것:
  1. int64 오버플로가 나지 않는가  (a*s 가 2^63 을 넘으면 값이 망가진다)
  2. MinHash 추정치가 실제 자카드 유사도에 수렴하는가
  3. 문서 수천 개를 다룰 만한 속도인가

노트북 안에서 눈으로 보는 것만으로는 2번을 확인하기 어렵다.
여기서는 유사도를 통제한 합성 데이터로 오차를 직접 잰다.
"""

from __future__ import annotations

import time
import zlib

import numpy as np

SHINGLE = 5
NUM_HASH = 64
MAX_CHARS = 1500
MOD = (1 << 31) - 1

rng = np.random.default_rng(42)
A = rng.integers(1, MOD, size=NUM_HASH, dtype=np.int64)
B = rng.integers(0, MOD, size=NUM_HASH, dtype=np.int64)


def shingles(text: str) -> np.ndarray:
    t = text[:MAX_CHARS]
    if len(t) < SHINGLE:
        return np.empty(0, dtype=np.int64)
    hs = {zlib.crc32(t[i:i + SHINGLE].encode()) & 0x7FFFFFFF
          for i in range(len(t) - SHINGLE + 1)}
    return np.fromiter(hs, dtype=np.int64, count=len(hs))


def minhash(sh: np.ndarray) -> np.ndarray:
    if sh.size == 0:
        return np.zeros(NUM_HASH, dtype=np.int64)
    return ((A[:, None] * sh[None, :] + B[:, None]) % MOD).min(axis=1)


def jaccard(x: np.ndarray, y: np.ndarray) -> float:
    sx, sy = set(x.tolist()), set(y.tolist())
    return len(sx & sy) / len(sx | sy) if (sx | sy) else 0.0


def sig_sim(x: np.ndarray, y: np.ndarray) -> float:
    return float((x == y).mean())


# ---------------------------------------------------------------------
print("=" * 66)
print("1. int64 오버플로 검사")
print("=" * 66)
max_s = 0x7FFFFFFF
max_a = MOD - 1
prod = int(max_a) * int(max_s) + int(MOD)
print(f"  최대 a*s+b = {prod:,}")
print(f"  int64 상한 = {2**63 - 1:,}")
assert prod < 2**63 - 1, "★ 오버플로! MOD 를 줄여야 한다"
print("  → 안전")

# ---------------------------------------------------------------------
print()
print("=" * 66)
print("2. MinHash 추정치가 실제 자카드에 수렴하는가")
print("=" * 66)
print(f"  {'겹침 목표':>10}{'실제 자카드':>14}{'MinHash 추정':>14}{'오차':>10}")
print("  " + "-" * 48)

alphabet = "가나다라마바사아자차카타파하국어한글모델학습"
errors = []
for overlap in [1.0, 0.9, 0.7, 0.5, 0.2, 0.0]:
    # 앞부분을 공유하고 뒤를 다르게 만들어 겹침 정도를 조절한다
    n = 1200
    shared = "".join(rng.choice(list(alphabet), int(n * overlap)))
    a = shared + "".join(rng.choice(list(alphabet), n - len(shared)))
    b = shared + "".join(rng.choice(list(alphabet), n - len(shared)))

    sa, sb = shingles(a), shingles(b)
    true_j = jaccard(sa, sb)
    est = sig_sim(minhash(sa), minhash(sb))
    err = abs(true_j - est)
    errors.append(err)
    print(f"  {overlap:>10.1f}{true_j:>14.3f}{est:>14.3f}{err:>10.3f}")

mean_err = float(np.mean(errors))
print(f"\n  평균 절대오차 {mean_err:.3f}")
# 시그니처 64개면 표준편차가 대략 sqrt(J(1-J)/64) ~ 0.06 이다. 0.12 를 넘으면 이상하다.
assert mean_err < 0.12, f"★ 오차가 너무 크다 ({mean_err:.3f}). 구현을 다시 봐야 한다"
print("  → 수렴함 (시그니처 64개 기준 기대 오차 범위 안)")

# ---------------------------------------------------------------------
print()
print("=" * 66)
print("3. 속도 — 실습 규모(5,000문서)를 감당하는가")
print("=" * 66)
docs = ["".join(rng.choice(list(alphabet), 1500)) for _ in range(300)]

t0 = time.time()
sigs = [minhash(shingles(d)) for d in docs]
el = time.time() - t0
per = el / len(docs)
print(f"  300문서 {el:.2f}초  (문서당 {per*1000:.1f}ms)")
print(f"  5,000문서 환산 → 약 {per*5000:.0f}초")
if per * 5000 > 300:
    print("  ★ 5분을 넘는다. MAX_CHARS 나 문서 수를 줄여야 한다")
else:
    print("  → 실습에 쓸 만함")

print()
print("=" * 66)
print("전부 통과")
print("=" * 66)
