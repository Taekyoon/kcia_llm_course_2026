#!/usr/bin/env bash
# =====================================================================
# VESSL 워크스페이스 환경 구성 — 한 번만 실행
#
#   bash setup_vessl.sh
#
# 이 스크립트가 하는 일
#   1. 기본 커널에 학습 스택 설치 (torch 는 건드리지 않음)
#   2. vLLM 은 별도 venv 에 격리 설치
#
# 왜 이렇게 나누는가
# -------------------
# vllm 0.27.1 은 `torch==2.13.0` 을 **등호로 하드핀**한다. 기본 커널에 그냥 설치하면
# 워크스페이스 이미지의 torch 2.9.1+cu130 이 삭제되고 2.13.0 으로 교체된다.
# 또 trl 의 vllm extra 는 `vllm<=0.26.0` 상한이라 `trl[vllm]` 은 버전을 되돌린다.
#
# vLLM 은 `vllm serve` 로 별도 프로세스이고 노트북은 HTTP 로 붙기 때문에
# venv 를 나눠도 실습에 지장이 없다.
#
# 검증 근거: verify/ 아래 점검 스크립트 실행 결과 (2026-08-25)
# =====================================================================
set -euo pipefail

CONSTRAINTS=/tmp/torch_pin.txt
VLLM_VENV=/opt/vllm-env
SERVE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

echo "======================================================================"
echo " 1. 현재 환경 확인"
echo "======================================================================"
python -c "import sys; print('  Python :', sys.version.split()[0])"
python -c "import torch; print('  torch  :', torch.__version__, '/ CUDA', torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  GPU    : /'

# torch 를 pip 의존성 해결에 맡기지 않는다. 이 한 줄이 핵심이다.
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch==${TORCH_VER}" > "$CONSTRAINTS"
echo
echo "  torch 를 constraints 로 고정: torch==${TORCH_VER}"

echo
echo "======================================================================"
echo " 2. 학습 스택 설치 (1·2일차 실습)"
echo "======================================================================"
# flash-attn 은 일부러 뺀다. 휠이 없어 소스 빌드에 30분~2시간 걸리는데,
# attn_implementation="sdpa" (torch 내장) 로 대체 가능해 실습에 불필요하다.
pip install -q -c "$CONSTRAINTS" -U \
    transformers datasets accelerate peft trl evaluate \
    bitsandbytes math-verify seqeval openai \
    tensorboard matplotlib ipywidgets
# tensorboard : 노트북들이 report_to="tensorboard" 로 학습 로그를 남긴다
# matplotlib  : llama-index 의 display_source_node 가 내부에서 import 한다
# ipywidgets  : 없으면 진행바가 IProgress 경고와 함께 깨진다

echo
echo "======================================================================"
echo " 3. RAG 스택 설치 (3일차 실습)"
echo "======================================================================"
# llama-index-llms-vllm(in-process) 대신 openai-like 를 쓴다.
# in-process 로 띄우면 이 커널의 torch 와 충돌한다.
pip install -q -c "$CONSTRAINTS" -U \
    llama-index-core llama-index-retrievers-bm25 llama-index-llms-openai-like PyStemmer

echo
echo "======================================================================"
echo " 4. torch 가 보존됐는지 확인 ★"
echo "======================================================================"
python -c "
import torch
print('  torch:', torch.__version__, '/ CUDA', torch.version.cuda, '/ 사용가능:', torch.cuda.is_available())
assert '${TORCH_VER}' in torch.__version__, '★ torch 가 교체됐습니다. 원인을 찾아야 합니다.'
print('  → 보존됨')
"
echo
echo "  설치된 버전:"
python - <<'PY'
from importlib.metadata import version
for p in ["transformers","datasets","trl","peft","accelerate","evaluate",
          "huggingface_hub","seqeval","llama-index-core"]:
    try:
        print(f"    {p:<24} {version(p)}")
    except Exception:
        print(f"    {p:<24} -")
PY

echo
echo "======================================================================"
echo " 5. vLLM 전용 venv (2·3일차 서빙 실습)"
echo "======================================================================"
# 주의: vLLM 은 cu130 휠을 배포하지 않는다. v0.25.0~v0.27.1 전 릴리즈가 cu129 뿐이다.
# pip 휠이 nvidia-*-cu12 런타임을 함께 설치하고 드라이버는 하위 호환이라 동작한다.
if [ -d "$VLLM_VENV" ]; then
  echo "  이미 존재합니다: $VLLM_VENV (재사용)"
else
  python -m venv "$VLLM_VENV"
fi
# shellcheck disable=SC1091
source "$VLLM_VENV/bin/activate"
pip install -q -U pip
pip install -q "vllm==0.27.1"
python -c "
import torch, vllm
print('  vllm :', vllm.__version__)
print('  torch:', torch.__version__, '(이 venv 안에만 존재)')
print('  GPU  :', torch.cuda.is_available())
assert torch.cuda.is_available(), '★ vLLM venv 에서 GPU 를 못 씁니다.'
"
deactivate

echo
echo "  기본 커널 torch 재확인:"
python -c "import torch; print('    ', torch.__version__)"

cat <<EOF

======================================================================
 준비 완료
======================================================================

실습 노트북은 work/notebook/ 에 있습니다 (2026 스택으로 마이그레이션된 것).

  work/notebook/1일차/  HPC_Classification실습 · HPC_NER실습 · HPC_MiniGPT실습
  work/notebook/2일차/  HPC_SFT실습 · HPC_DPO실습 · HPC_GRPO실습 · HPC_퓨샷실습
  work/notebook/3일차/  HPC_Amazon요약실습 · HPC_BM25_RAG실습

--- 서빙 실습(퓨샷 · Amazon 요약 · BM25 RAG)을 하려면 ---

  source $VLLM_VENV/bin/activate
  nohup vllm serve $SERVE_MODEL --port 8000 \\
      --gpu-memory-utilization 0.80 \\
      --max-model-len 16384 \\
      > /tmp/vllm.log 2>&1 &
  tail -f /tmp/vllm.log        # "Application startup complete" 대기 (3~6분)

  ★ 두 플래그는 생략하지 마세요. 이유가 있습니다(2026-08-25 실측).

    --gpu-memory-utilization 0.80
      기본값 0.92 는 GPU 의 92% 를 미리 확보하려 한다. 그런데 실습에서는
      학생이 노트북을 띄워둔 채 서빙을 쓴다. Jupyter 커널이 3~4GB 만 잡고
      있어도 다음과 같이 실패한다:
        ValueError: Free memory on device cuda:0 (40.2/44.39 GiB) on startup
        is less than desired GPU memory utilization (0.92, 40.84 GiB)
      0.80 이면 vLLM 이 약 35GB 를 쓰고 노트북 몫으로 9GB 가 남는다.

    --max-model-len 16384
      Qwen3-4B-Instruct-2507 의 기본 컨텍스트는 262,144 토큰이다.
      그만큼의 KV 캐시를 잡으려 해서 메모리를 크게 낭비한다.
      강의 실습에는 16K 로 충분하고 기동도 빨라진다.

  노트북은 기본 커널에서 그대로 실행합니다. HTTP 로 붙으므로 venv 를 오갈 필요 없습니다.

--- MiniGPT 실습(1일차)도 이 커널에서 돌아갑니다 ---

  예전에는 Keras/TensorFlow 스택이라 별도 venv 가 필요했지만,
  PyTorch + HuggingFace 로 재작성해서 이제 나머지 8종과 같은 환경을 씁니다.
  추가 설치가 필요 없습니다.
======================================================================
EOF
