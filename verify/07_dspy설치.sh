#!/usr/bin/env bash
# =====================================================================
# [VESSL 점검 7] DSPy 설치 + vLLM 연결 + GEPA 소요시간 실측
#
#   bash verify/07_dspy설치.sh
#
# 왜 이걸 먼저 돌리는가
# ---------------------
# 프롬프트 자동 최적화 실습(2일차 ⑧-3)을 DSPy 로 만들기로 했다. 그런데
# **dspy 3.3.1 은 2026-08-21 릴리스**다. 커뮤니티 회귀 보고가 쌓일 시간이 없었다.
# 노트북을 다 쓰고 나서 설치가 안 된다는 걸 알면 되돌리는 비용이 크다.
#
# 이 repo 는 llama-index-question-gen-openai 하나 때문에 core 가 0.14 -> 0.12 로
# 끌려 내려가 하루를 날린 적이 있다. 같은 일을 반복하지 않는다.
#
# 확인하는 것 넷
#   1. 설치가 되는가 — 그리고 **HF 스택을 끌어내리지 않는가**
#   2. vLLM 에 붙는가 — 접두사 openai/ · JSONAdapter 동작
#   3. GEPA 가 도는가 — import 부터 compile 까지
#   4. **얼마나 걸리는가** — 강의에서 쓸 max_metric_calls 를 정하는 근거
#
# 전제
#   - bash setup_vessl.sh 완료
#   - vLLM 서버 기동:
#       source /opt/vllm-env/bin/activate
#       nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
#           --gpu-memory-utilization 0.80 --max-model-len 16384 \
#           > /tmp/vllm.log 2>&1 &
# =====================================================================
set -uo pipefail   # -e 는 쓰지 않는다. 하나 실패해도 나머지 정보를 끝까지 뽑는다.

CONSTRAINTS=/tmp/torch_pin.txt
SERVE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
BASE_URL="http://localhost:8000/v1"

echo "======================================================================"
echo " 0. 설치 전 스택 기록 (되돌림 감지용)"
echo "======================================================================"
python - <<'PY'
import json, pathlib
from importlib.metadata import version, PackageNotFoundError
PKGS = ["torch","transformers","datasets","trl","peft","accelerate",
        "huggingface_hub","openai","llama-index-core","pydantic"]
before = {}
for p in PKGS:
    try:    before[p] = version(p)
    except PackageNotFoundError: before[p] = None
    print(f"  {p:<22} {before[p] or '-'}")
pathlib.Path("/tmp/dspy_before.json").write_text(json.dumps(before))
PY

# constraints 를 다시 만든다. setup_vessl.sh 를 안 돌렸거나 /tmp 가 비었을 수 있다.
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
{
  echo "torch==${TORCH_VER}"
  echo "openai<3"
  # litellm 은 dspy 가 새로 끌고 온다. gepa 의 일부 extra 가 litellm<1.92 상한을
  # 갖는데 gepa[dspy] 에는 그 상한이 없다. 그래도 여기서 한 번 못박아 둔다.
  echo "litellm<2"
} > "$CONSTRAINTS"
echo
echo "  constraints:"
sed 's/^/    /' "$CONSTRAINTS"

echo
echo "======================================================================"
echo " 1. dspy 설치"
echo "======================================================================"
# optuna 는 MIPROv2 에만 필요하다. 우리는 GEPA 를 쓰므로 불필요하지만,
# 강의 중에 MIPROv2 를 만져볼 수 있으니 같이 넣는다.
# (MIPROv2 는 optuna 가 없으면 import 가 아니라 **compile() 실행 중**에 죽는다)
if pip install -c "$CONSTRAINTS" -U "dspy[optuna]"; then
  echo "  설치 명령 성공"
else
  echo "  ★ 설치 실패. 위 로그를 회신해 주세요."
  exit 1
fi

echo
echo "======================================================================"
echo " 2. HF 스택이 끌려 내려갔는지 확인 ★ 가장 중요"
echo "======================================================================"
python - <<'PY'
import json, sys
from importlib.metadata import version, PackageNotFoundError
before = json.loads(open("/tmp/dspy_before.json").read())
changed, lost = [], []
for p, was in before.items():
    try:    now = version(p)
    except PackageNotFoundError: now = None
    if now != was:
        changed.append((p, was, now))
        # 버전이 내려갔으면 심각하다
        def key(v): return tuple(int(x) for x in (v or "0").split(".")[:3] if x.isdigit())
        if was and now and key(now) < key(was):
            lost.append(p)
    mark = "" if now == was else f"   ← {was} 에서 바뀜"
    print(f"  {p:<22} {now or '-'}{mark}")

print()
if not changed:
    print("  → 기존 스택 그대로. 안전합니다.")
elif lost:
    print(f"  ★ 다운그레이드된 패키지: {', '.join(lost)}")
    print("    dspy 가 기존 스택을 끌어내렸습니다. 이 상태로 강의에 쓸 수 없습니다.")
    sys.exit(1)
else:
    print("  → 올라가기만 했습니다. 아래 pip check 와 노트북 실행으로 확인하세요.")
PY
HF_OK=$?

echo
echo "  torch 보존 확인:"
python -c "
import torch
print('    torch:', torch.__version__, '/ CUDA', torch.version.cuda, '/ 사용가능:', torch.cuda.is_available())
assert '${TORCH_VER}' in torch.__version__, '★ torch 가 교체됐습니다.'
print('    → 보존됨')
"

echo
echo "  pip check:"
if pip check > /tmp/pipcheck_dspy.txt 2>&1; then
  echo "    충돌 없음"
else
  grep -v "anaconda-cli-base" /tmp/pipcheck_dspy.txt | sed 's/^/    /' || true
fi

echo
echo "  설치된 dspy 계열:"
python - <<'PY'
from importlib.metadata import version
for p in ["dspy","gepa","litellm","optuna","openai","pydantic"]:
    try:    print(f"    {p:<12} {version(p)}")
    except Exception: print(f"    {p:<12} -")
PY

if [ "${HF_OK:-0}" -ne 0 ]; then
  echo
  echo " ★ 스택이 훼손되어 이후 단계를 건너뜁니다."
  exit 1
fi

echo
echo "======================================================================"
echo " 3. vLLM 서버 확인"
echo "======================================================================"
if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "  응답함"
else
  echo "  ★ vLLM 서버가 응답하지 않습니다. 아래로 띄운 뒤 다시 실행하세요."
  echo "      source /opt/vllm-env/bin/activate"
  echo "      nohup vllm serve ${SERVE_MODEL} --port 8000 \\"
  echo "          --gpu-memory-utilization 0.80 --max-model-len 16384 > /tmp/vllm.log 2>&1 &"
  exit 1
fi

echo
echo "======================================================================"
echo " 4. DSPy → vLLM 연결 + GEPA 실측"
echo "======================================================================"
MODEL="$SERVE_MODEL" BASE="$BASE_URL" python - <<'PY'
import json, os, sys, time

MODEL, BASE = os.environ["MODEL"], os.environ["BASE"]

try:
    import dspy
except Exception as e:
    print(f"  ★ import dspy 실패: {type(e).__name__}: {e}")
    sys.exit(1)
print(f"  dspy {dspy.__version__ if hasattr(dspy,'__version__') else '(버전 속성 없음)'}")

# ── 4-1. LM 연결 ─────────────────────────────────────────────────
# 접두사는 openai/ 다. hosted_vllm/ 가 아니다.
# max_tokens 를 반드시 건다 — 미지정 시 한 요청이 90초 넘게 생성한 전례가 있다.
# cache=False 로 둔다. 기본 True 라 재실행이 LM 을 안 치고 캐시를 돌려준다.
lm = dspy.LM(f"openai/{MODEL}", api_base=BASE, api_key="dummy",
             model_type="chat", temperature=0.0, max_tokens=512, cache=False)
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())   # 4B 는 ChatAdapter 마커를 자주 어긴다

t0 = time.time()
try:
    out = lm("한국의 수도는? 한 단어로만 답하세요.")
    print(f"  LM 응답 OK ({time.time()-t0:.1f}초): {str(out)[:80]}")
except Exception as e:
    print(f"  ★ LM 호출 실패: {type(e).__name__}: {e}")
    sys.exit(1)

# ── 4-2. 구조화 출력 (JSONAdapter) ───────────────────────────────
# vLLM 0.12 에서 guided_json 이 제거됐다. DSPy 는 표준 response_format 을 쓴다.
class Extract(dspy.Signature):
    """상품 설명에서 정보를 뽑는다."""
    text: str = dspy.InputField()
    name: str = dspy.OutputField(desc="상품명")
    color: str = dspy.OutputField(desc="색상. 없으면 '미상'")

prog = dspy.Predict(Extract)
try:
    r = prog(text="Blue Anker PowerCore 10000 portable charger with USB-C.")
    print(f"  구조화 출력 OK: name={r.name!r} color={r.color!r}")
except Exception as e:
    print(f"  ★ 구조화 출력 실패: {type(e).__name__}: {e}")
    sys.exit(1)

# ── 4-3. GEPA ────────────────────────────────────────────────────
try:
    from dspy import GEPA
except Exception as e:
    print(f"  ★ GEPA import 실패: {type(e).__name__}: {e}")
    sys.exit(1)
print("  GEPA import OK")

SAMPLES = [
    ("Blue Anker PowerCore 10000 portable charger with USB-C.", "Anker"),
    ("Red Logitech MX Master 3S wireless mouse for creators.",   "Logitech"),
    ("Black Sony WH-1000XM5 noise cancelling headphones.",       "Sony"),
    ("Silver Apple MacBook Air 13 inch M3 laptop computer.",     "Apple"),
    ("Green Stanley Quencher 40oz stainless steel tumbler.",     "Stanley"),
    ("White Samsung Galaxy Buds3 Pro wireless earbuds.",         "Samsung"),
]
train = [dspy.Example(text=t, brand=b).with_inputs("text") for t, b in SAMPLES[:4]]
val   = [dspy.Example(text=t, brand=b).with_inputs("text") for t, b in SAMPLES[4:]]

class Brand(dspy.Signature):
    """정보를 뽑는다."""          # ← 일부러 부실하게. 개선 여지를 남긴다
    text: str = dspy.InputField()
    brand: str = dspy.OutputField()

# metric 은 프로그램 검증이다. 판정에 LLM 이 끼지 않아 노이즈가 0이다.
# GEPA 는 5인자 시그니처를 쓰고, float 이 아니라 Prediction(score=, feedback=) 을 받는다.
# feedback 문자열이 reflection LM 에 전달되는 핵심 채널이라 반드시 채운다.
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    got  = (pred.brand or "").strip().lower()
    want = gold.brand.strip().lower()
    ok = want in got
    return dspy.Prediction(
        score=float(ok),
        feedback="정답" if ok else f"'{gold.text[:40]}' 의 브랜드는 {gold.brand} 인데 {pred.brand!r} 로 답함",
    )

CALLS = 40      # 실측용 소량. 강의 노트북에서는 120 을 쓸 예정이다.
print(f"\n  GEPA compile 시작 — max_metric_calls={CALLS}")
t0 = time.time()
try:
    opt = GEPA(metric=metric, max_metric_calls=CALLS,
               reflection_lm=lm, num_threads=8, track_stats=True)
    optimized = opt.compile(dspy.Predict(Brand), trainset=train, valset=val)
except Exception as e:
    print(f"  ★ GEPA compile 실패: {type(e).__name__}: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)
el = time.time() - t0

def score(p):
    return sum(metric(ex, p(text=ex.text)).score for ex in val) / len(val)

print(f"  compile 완료 — {el:.1f}초 ({CALLS}회 기준)")
print(f"  → 강의에서 120회면 약 {el * 120 / CALLS / 60:.1f}분 예상")
print()
print("  최적화 후 프롬프트(instructions):")
for name, pr in optimized.named_predictors():
    print(f"    [{name}] {pr.signature.instructions[:400]}")
print()
print(f"  val 점수(최적화본): {score(optimized):.2f}")
print("  ※ 4B 하나로 task+reflection 을 겸하므로 개선폭이 작을 수 있습니다.")
print("    점수가 안 오르면 노트북에서는 시작 프롬프트를 더 부실하게 주거나")
print("    직접 구현(OPRO 루프)으로 전환합니다.")
PY
RC=$?

echo
echo "======================================================================"
echo " 결과"
echo "======================================================================"
if [ "$RC" -eq 0 ]; then
  echo " ✅ DSPy 경로 사용 가능"
  echo "    위 '120회 예상 시간' 과 'val 점수' 를 회신해 주세요."
  echo "    그 두 값으로 노트북의 max_metric_calls 와 시작 프롬프트 수준을 정합니다."
else
  echo " ❌ 실패 — 3막을 직접 구현(OPRO 루프)으로 전환합니다."
  echo "    위 에러 메시지 전문을 회신해 주세요."
fi
