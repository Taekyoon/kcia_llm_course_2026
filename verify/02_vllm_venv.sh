#!/usr/bin/env bash
# =====================================================================
# [VESSL] vLLM 전용 venv 구성 — 학습 환경과 분리
#
#   bash verify/02_vllm_venv.sh
#
# 왜 분리하는가
# -------------
# vllm 0.27.1 은 `torch==2.13.0` 을 **등호로 하드핀**한다. 기본 커널에 그냥 설치하면
# 지금 쓰는 torch 2.9.1+cu130 이 삭제되고 2.13.0 으로 교체된다(2~3GB 재다운로드).
# 또 trl 1.10.0 의 vllm extra 는 `vllm<=0.26.0` 상한이라 `pip install trl[vllm]` 은
# vLLM 을 되돌린다. GRPO 에서 정면 충돌한다.
#
# vLLM 은 `vllm serve` 로 **별도 프로세스**이고 노트북들은 HTTP(OpenAI 호환 API)로
# 붙기 때문에, venv 를 분리해도 실습에 아무 지장이 없다.
#
# CUDA 버전에 대하여 (2026-08-25 확인)
# ------------------------------------
# 이 워크스페이스는 CUDA 13.0 / torch 2.9.1+cu130 이다. 그런데
# **vLLM 은 cu130 휠을 배포하지 않는다.** v0.25.0 ~ v0.27.1 전 릴리즈를 확인한 결과
# `+cu129` 빌드만 있다. 따라서 cu129 빌드를 쓴다.
#
# 문제되지 않는 이유: pip 휠이 필요한 CUDA 런타임(nvidia-*-cu12)을 함께 설치하고,
# NVIDIA 드라이버는 하위 호환이라 580.159.03 에서 CUDA 12.9 바이너리가 동작한다.
# 다만 **이 부분은 실측으로 확인해야 하는 지점**이다. 아래 4단계에서 검증한다.
#
# 근거: verify/결과.md · review/02_문제점.md D-10
# =====================================================================
set -euo pipefail

VENV=/opt/vllm-env
MODEL="Qwen/Qwen3-4B-Instruct-2507"   # Apache-2.0, text-only, Qwen3ForCausalLM
PORT=8000

echo "=== 1. 기본 환경의 torch 확인 (건드리면 안 되는 것) ==="
python -c "import torch; print('  기본 커널 torch:', torch.__version__)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== 2. vLLM 전용 venv 생성: $VENV ==="
if [ -d "$VENV" ]; then
  echo "  이미 존재합니다. 재사용합니다."
else
  python -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q -U pip

echo
echo "=== 3. vLLM 0.27.1 설치 (PyPI = cu129 빌드) ==="
echo "  torch 2.13.0 + nvidia-*-cu12 를 이 venv 안에만 설치합니다."
echo "  다운로드가 3~5GB 정도라 몇 분 걸립니다."
pip install "vllm==0.27.1"

echo
echo "=== 4. ★ cu129 빌드가 이 드라이버에서 실제로 도는지 확인 ==="
echo "  (CUDA 13 환경에 12.9 바이너리를 얹는 것이라 여기서 판가름납니다)"
python - <<'PY'
import torch, vllm
print("  vllm :", vllm.__version__)
print("  torch:", torch.__version__, "/ CUDA", torch.version.cuda)
print("  cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU  : {p.name} {p.total_memory/2**30:.1f}GB sm_{p.major}{p.minor}")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    print("  행렬곱 스모크 테스트:", float((x @ x).sum()) is not None)
else:
    raise SystemExit("  ★ GPU를 못 씁니다. cu129/CUDA13 조합 문제일 수 있습니다.")
PY
deactivate

echo
echo "=== 5. 기본 환경 torch 가 그대로인지 재확인 ==="
python -c "import torch; print('  기본 커널 torch:', torch.__version__, '(2.9.1+cu130 이어야 정상)')"

cat <<EOF

======================================================================
서버 기동 (이 터미널에서 계속 진행)

  source $VENV/bin/activate
  nohup vllm serve $MODEL --port $PORT \\
      --gpu-memory-utilization 0.80 \\
      --max-model-len 16384 \\
      > /tmp/vllm.log 2>&1 &

  ★ 두 플래그를 생략하면 실패합니다 (2026-08-25 실측):
      ValueError: Free memory on device cuda:0 (40.2/44.39 GiB) on startup
      is less than desired GPU memory utilization (0.92, 40.84 GiB)
    Jupyter 커널이 3~4GB 만 잡고 있어도 기본값 0.92 로는 못 뜹니다.
    그리고 이 모델의 기본 컨텍스트가 262,144 토큰이라 KV 캐시가 과도합니다.

기동 확인 — 모델 다운로드(약 8GB) 때문에 처음엔 3~6분 걸립니다:

  tail -f /tmp/vllm.log          # "Application startup complete" 가 뜨면 준비됨
  curl -s http://localhost:$PORT/v1/models

준비되면 노트북 커널로 돌아가 verify/05_vllm검증.py 를 실행하세요.
(노트북은 기본 커널에서 그대로 실행합니다. HTTP 로 붙으므로 venv 를 오갈 필요 없음)

문제가 생기면 로그 전체를 회신해 주세요:
  tail -n 100 /tmp/vllm.log
======================================================================
EOF
