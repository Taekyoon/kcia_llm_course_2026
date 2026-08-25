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
{
  echo "torch==${TORCH_VER}"
  # openai 3.x 는 llama-index-llms-openai 가 요구하는 openai<3 과 충돌한다.
  # 노트북이 쓰는 기능(OpenAI 클라이언트, response_format, structured outputs)은
  # 2.x 에서 이미 검증됐다(verify/05_vllm검증.py 통과 당시 openai 2.54.0).
  echo "openai<3"
  # litellm 은 dspy 가 끌고 온다. 상한을 못박아 둔다.
  echo "litellm<2"
} > "$CONSTRAINTS"
echo
echo "  constraints 고정:"
sed 's/^/    /' "$CONSTRAINTS"

# ── 구버전 핀 패키지 정리 (설치보다 먼저) ──────────────────────────────
# llama-index-question-gen-openai 는 혼자 오지 않는다. agent-openai, program-openai
# 같은 형제들을 함께 끌고 오는데 전부 llama-index-core<0.13 에 묶여 있다.
# 남겨두면 이후 모든 pip 호출이 충돌을 보고한다. 여기서 먼저 걷어낸다.
LEGACY="llama-index-question-gen-openai llama-index-agent-openai llama-index-program-openai"
removed=0
for p in $LEGACY; do
  if pip show "$p" >/dev/null 2>&1; then
    echo "  구버전 핀 패키지 제거: $p"
    pip uninstall -q -y "$p"
    removed=1
  fi
done
[ "$removed" = "0" ] && echo "  구버전 핀 패키지: 없음"

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
# ⚠️ llama-index-question-gen-openai 를 설치하지 마세요.
#    SubQuestionQueryEngine.from_defaults() 가 없으면 ImportError 를 내며 이 패키지를
#    권하지만, 최신판(0.3.1)이 llama-index-core<0.13 에 묶여 있어 설치하는 순간
#    core 가 0.14.24 -> 0.12.52 로 끌려 내려가고 llama-index 전체가 깨진다.
#    노트북에서 LLMQuestionGenerator 를 직접 넘기는 방식으로 우회했다.
#    (구버전 형제 패키지 제거는 1단계에서 이미 처리했다)
pip install -q -c "$CONSTRAINTS" -U \
    llama-index llama-index-core llama-index-retrievers-bm25 \
    llama-index-llms-openai-like PyStemmer

echo
echo "  llama-index 정합성 확인:"
python - <<'PY'
import sys
from importlib.metadata import version, PackageNotFoundError

MIN = {                       # 이보다 낮으면 구버전 핀 패키지에 끌려 내려간 것이다
    "llama-index-core": (0, 14),
    "llama-index-llms-openai-like": (0, 7),   # 0.4.x 는 transformers<5 를 요구한다
}
MAX = {                       # 이보다 높으면 다른 쪽과 충돌한다
    "openai": (3, 0),         # llama-index-llms-openai 가 openai<3 을 요구한다
}
bad = []
for pkg, want in MIN.items():
    try:
        v = version(pkg)
    except PackageNotFoundError:
        print(f"    {pkg:<32} (미설치)")
        continue
    got = tuple(int(x) for x in v.split(".")[:2])
    mark = "" if got >= want else f"  ★ {'.'.join(map(str, want))} 이상 필요"
    print(f"    {pkg:<32} {v}{mark}")
    if got < want:
        bad.append(pkg)

for pkg, limit in MAX.items():
    try:
        v = version(pkg)
    except PackageNotFoundError:
        continue
    got = tuple(int(x) for x in v.split(".")[:2])
    mark = "" if got < limit else f"  ★ {'.'.join(map(str, limit))} 미만이어야 함"
    print(f"    {pkg:<32} {v}{mark}")
    if got >= limit:
        bad.append(pkg)

if bad:
    print("\n    ★ 구버전 핀 패키지가 llama-index 를 끌어내렸습니다.")
    print("      llama-index-question-gen-openai 와 그 형제들이 원인입니다. 복구:")
    print("        pip uninstall -y llama-index-question-gen-openai \\")
    print("            llama-index-agent-openai llama-index-program-openai")
    print("        pip install -U llama-index llama-index-core llama-index-llms-openai-like")
    sys.exit(1)
PY

echo
echo "======================================================================"
echo " 3-2. 프롬프트 자동 최적화 (2일차 ⑧-3)"
echo "======================================================================"
# dspy 는 transformers/datasets/trl/peft/huggingface_hub 에 의존하지 않는다.
# 새로 들어오는 것은 litellm 계열뿐이라 학습 스택을 건드리지 않는다.
# optuna 는 MIPROv2 전용이다. 우리는 GEPA 를 쓰므로 필수는 아니지만,
# 강의 중에 MIPROv2 를 만져볼 때 compile() 도중에 죽는 것을 막으려고 같이 넣는다.
pip install -q -c "$CONSTRAINTS" -U "dspy[optuna]"
python -c "
import dspy
from dspy import GEPA
print('  dspy  :', getattr(dspy, '__version__', '?'))
print('  GEPA  : import OK')
"

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
          "huggingface_hub","seqeval","llama-index-core","openai"]:
    try:
        print(f"    {p:<24} {version(p)}")
    except Exception:
        print(f"    {p:<24} -")
PY

echo
echo "  최종 의존성 검사 (pip check):"
if pip check > /tmp/pipcheck.txt 2>&1; then
  echo "    충돌 없음"
else
  # conda CLI 도구(anaconda-cli-base <-> click)는 실습과 무관하므로 걸러낸다
  grep -v "anaconda-cli-base" /tmp/pipcheck.txt | sed 's/^/    /' || true
  if grep -qv "anaconda-cli-base" /tmp/pipcheck.txt; then
    echo
    echo "    ※ 위 충돌이 llama-index 관련이면 아래로 복구하세요:"
    echo "       pip uninstall -y llama-index-question-gen-openai \\"
    echo "           llama-index-agent-openai llama-index-program-openai"
    echo "       bash setup_vessl.sh"
  fi
fi

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

echo
echo "======================================================================"
echo " 6. 데이터 경로 고정 (HPC_DATA)"
echo "======================================================================"
# 2일차 노트북들은 앞 단계가 만든 파일을 뒤 단계가 읽는다. 그런데 Jupyter 도 nbconvert 도
# **커널 cwd 를 노트북이 있는 디렉터리**로 잡아서, 상대경로를 쓰면 노트북마다 다른 곳을 본다.
# 여기서 절대경로를 한 번 못박아 두면 어느 노트북에서 열어도 같은 곳을 가리킨다.
DATA_ABS="$(pwd)/data"
mkdir -p "$DATA_ABS"
export HPC_DATA="$DATA_ABS"
if grep -q "^export HPC_DATA=" ~/.bashrc 2>/dev/null; then
  sed -i "s|^export HPC_DATA=.*|export HPC_DATA=\"$DATA_ABS\"|" ~/.bashrc
  echo "  ~/.bashrc 갱신: HPC_DATA=$DATA_ABS"
else
  echo "export HPC_DATA=\"$DATA_ABS\"" >> ~/.bashrc
  echo "  ~/.bashrc 추가: HPC_DATA=$DATA_ABS"
fi
echo
echo "  ★ Jupyter 커널은 ~/.bashrc 를 안 읽을 수 있습니다."
echo "    노트북에서 경로가 안 잡히면 커널을 재시작하거나, Jupyter 자체를 이 셸에서"
echo "    다시 띄우세요. 노트북은 HPC_DATA 가 없으면 repo 루트를 스스로 찾습니다."

cat <<EOF

======================================================================
 준비 완료
======================================================================

실습 노트북은 work/notebook/ 에 있습니다. **폴더 안의 순서가 곧 진행 순서**입니다.

  1일차 (2단원)  HPC_Classification실습 → HPC_NER실습
  2일차 (3단원)  HPC_데이터처리실습 → HPC_Amazon요약실습
                 → HPC_프롬프트최적화실습 → HPC_MiniGPT실습
  3일차 (4단원)  HPC_평가실습 → HPC_퓨샷실습 → HPC_SFT실습
                 → HPC_DPO실습 → HPC_GRPO실습 → HPC_BM25_RAG실습(심화)

  ★ 2일차는 앞 노트북이 만든 파일을 뒤 노트북이 받습니다. 순서를 지켜주세요.
      데이터처리 → \$HPC_DATA/ko_wiki_clean.jsonl → MiniGPT 이어학습(CPT)
      Amazon    → \$HPC_DATA/amazon_ko_sft.mine.jsonl → 3일차 SFT

--- 서빙 실습(Amazon · 프롬프트최적화 · 퓨샷 · 평가 · RAG)을 하려면 ---

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
