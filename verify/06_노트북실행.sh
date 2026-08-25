#!/usr/bin/env bash
# =====================================================================
# [VESSL 점검 6] 마이그레이션 노트북을 실제로 실행해 완주 여부 확인
#
#   bash verify/06_노트북실행.sh              # 빠른 것부터 (추론계)
#   bash verify/06_노트북실행.sh amazon       # 하나만
#   bash verify/06_노트북실행.sh all          # 학습계 포함 전부 (오래 걸림)
#
# 왜 필요한가
# -----------
# 지금까지 검증한 것은 **부품**이다 — response_format 이 JSON 을 돌려주는가,
# klue ner 이 로드되는가, SFT 가 1스텝 도는가.
# 그런데 노트북은 셀이 **사슬로 엮여** 있다. 특히 Amazon 요약은 7단계가
# Step1 출력 → Step3 입력, Step2 출력 → json.loads() → Step6 입력 식으로
# 이어져서 중간 하나만 형식이 어긋나면 뒤가 전부 무너진다.
#
# 부품 검증으로는 이걸 못 잡는다. 실제로 끝까지 돌려봐야 한다.
#
# 전제
#   - bash setup_vessl.sh 완료
#   - 서빙계(퓨샷·amazon·rag)는 vLLM 서버가 떠 있어야 함:
#       source /opt/vllm-env/bin/activate
#       nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
#           --gpu-memory-utilization 0.80 --max-model-len 16384 \
#           > /tmp/vllm.log 2>&1 &
# =====================================================================
set -uo pipefail   # -e 는 쓰지 않는다. 하나 실패해도 나머지를 계속 돌려 전체 그림을 본다.

OUT=/tmp/nb_exec
mkdir -p "$OUT"
W=work/notebook

# 추론계 — 빠르다 (vLLM 서버 필요)
FAST=(
  "3일차/HPC_Amazon요약실습.ipynb|1200"
  "3일차/HPC_BM25_RAG실습.ipynb|900"
  "2일차/HPC_퓨샷실습.ipynb|1800"
  "2일차/HPC_평가실습.ipynb|900"      # LLM-as-judge 가 vLLM 을 쓴다
)
# 학습계 — 오래 걸린다. vLLM 서버 불필요
# (데이터처리는 GPU 를 쓰지 않지만 vLLM 과 무관하므로 여기 둔다)
SLOW=(
  "3일차/HPC_데이터처리실습.ipynb|1200"
  "1일차/HPC_Classification실습.ipynb|1800"
  "1일차/HPC_NER실습.ipynb|1800"
  "1일차/HPC_MiniGPT실습.ipynb|2400"
  "2일차/HPC_SFT실습.ipynb|2400"
  "2일차/HPC_DPO실습.ipynb|2400"
  "2일차/HPC_GRPO실습.ipynb|3000"
)

case "${1:-fast}" in
  # all 은 일부러 두지 않는다. 서빙계와 학습계는 한 GPU 에서 같이 못 돈다(아래 참조).
  fast)    TARGETS=("${FAST[@]}") ;;
  slow)    TARGETS=("${SLOW[@]}") ;;
  amazon)  TARGETS=("3일차/HPC_Amazon요약실습.ipynb|1200") ;;
  rag)     TARGETS=("3일차/HPC_BM25_RAG실습.ipynb|900") ;;
  fewshot) TARGETS=("2일차/HPC_퓨샷실습.ipynb|1800") ;;
  minigpt) TARGETS=("1일차/HPC_MiniGPT실습.ipynb|2400") ;;
  data)    TARGETS=("3일차/HPC_데이터처리실습.ipynb|1200") ;;
  eval)    TARGETS=("2일차/HPC_평가실습.ipynb|900") ;;
  *)       TARGETS=("$1|1800") ;;
esac

# 실행 전에 정적 검사부터. GPU 를 붙잡기 전에 문법 오류를 걸러낸다.
if command -v uv >/dev/null 2>&1 && [ -f tools/validate_notebooks.py ]; then
  echo " 정적 검사..."
  uv run python tools/validate_notebooks.py 2>&1 | tail -n 4 | sed 's/^/   /' || true
  echo
fi

echo "======================================================================"
echo " 노트북 실행 검증 — 대상 ${#TARGETS[@]}종"
echo "======================================================================"

has_serving=$(printf '%s\n' "${TARGETS[@]}" | grep -cE 'Amazon|BM25|퓨샷' || true)
has_training=$(printf '%s\n' "${TARGETS[@]}" | grep -cE 'Classification|NER|MiniGPT|SFT|DPO|GRPO' || true)
vllm_up=$(curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && echo 1 || echo 0)

nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/ GPU 메모리: /'

# ★ 서빙계와 학습계는 **같이 못 돌린다.**
#    vLLM 은 기동 시 GPU 의 80% 를 선점한다(--gpu-memory-utilization 0.80).
#    그 상태로 학습 노트북을 돌리면 OutOfMemoryError 가 난다. 실제로 그렇게 만들었다:
#      Process 3238 has 35.79 GiB memory in use.  ← vLLM
#      OutOfMemoryError: Tried to allocate 20.00 MiB ... 4.75 MiB is free
#    이건 이 스크립트만의 문제가 아니라 **강의 운영에도 그대로 적용되는 제약**이다.
#    2일차는 서빙 실습(퓨샷)과 학습 실습(SFT/DPO/GRPO)이 같은 날에 있다.
if [ "${has_serving:-0}" -gt 0 ] && [ "${has_training:-0}" -gt 0 ]; then
  echo
  echo " ★ 서빙계와 학습계를 함께 지정했습니다. 한 GPU 에서는 같이 돌 수 없습니다."
  echo "   vLLM 이 GPU 의 80% 를 선점하므로 학습계가 OOM 으로 죽습니다."
  echo "   나눠서 실행하세요:"
  echo "       bash verify/06_노트북실행.sh fast    # 서빙계 — vLLM 켠 상태"
  echo "       (vLLM 종료)"
  echo "       bash verify/06_노트북실행.sh slow    # 학습계 — vLLM 끈 상태"
  echo
  exit 1
fi

if [ "${has_serving:-0}" -gt 0 ]; then
  if [ "$vllm_up" = "1" ]; then
    echo " vLLM 서버: 응답함 (서빙계 실행에 필요)"
  else
    echo " ★ vLLM 서버가 응답하지 않습니다. 서빙계 노트북은 실패합니다."
    echo "   source /opt/vllm-env/bin/activate"
    echo "   nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \\"
    echo "       --gpu-memory-utilization 0.80 --max-model-len 16384 > /tmp/vllm.log 2>&1 &"
    exit 1
  fi
fi

if [ "${has_training:-0}" -gt 0 ] && [ "$vllm_up" = "1" ]; then
  echo
  echo " ★ 학습계를 돌리려는데 vLLM 서버가 떠 있습니다. GPU 를 선점하고 있어 OOM 이 납니다."
  echo "   먼저 내려주세요:"
  echo "       pkill -f 'vllm serve' ; sleep 5 ; nvidia-smi"
  echo
  exit 1
fi
echo

PASS=(); FAIL=()
for item in "${TARGETS[@]}"; do
  rel="${item%|*}"; tmo="${item#*|}"
  name=$(basename "$rel" .ipynb)
  echo "----------------------------------------------------------------------"
  echo " ▶ $rel   (제한 ${tmo}초)"
  echo "----------------------------------------------------------------------"
  t0=$(date +%s)

  # --allow-errors 를 쓰지 않는다. 첫 실패에서 멈춰야 어느 셀이 문제인지 드러난다.
  if timeout "$tmo" jupyter nbconvert \
        --to notebook --execute \
        --ExecutePreprocessor.timeout="$tmo" \
        --output-dir "$OUT" --output "${name}.executed.ipynb" \
        "$W/$rel" > "$OUT/${name}.log" 2>&1; then
    el=$(( $(date +%s) - t0 ))
    echo "   ✅ 완주 (${el}초)"
    PASS+=("$name")

    # ⚠️ 완주 != 정상 동작.
    # Amazon 요약은 각 스텝이 try/except 로 감싸여 있어 LLM 호출이 전부 실패해도
    # 'error' 문자열을 반환하며 끝까지 진행된다. 결과물을 따로 봐야 한다.
    # grep -c 는 매치가 0이면 exit 1 이다. `|| echo 0` 을 붙이면 "0\n0" 이 되어
    # [ "$n" -gt 0 ] 이 integer expression expected 로 깨진다. 실제로 그렇게 만들었다.
    # grep 을 wc 로 받아 항상 숫자 하나가 나오게 한다.
    count_in() { grep -o "$1" "$2" 2>/dev/null | wc -l | tr -d ' '; }

    nerr=$(count_in "\"error\"" "$OUT/${name}.executed.ipynb")
    if [ "${nerr:-0}" -gt 0 ]; then
      echo "   ⚠️  출력에 'error' 가 ${nerr}회 — 예외를 삼키고 진행했을 수 있습니다."
    fi
    ntb=$(count_in "Traceback" "$OUT/${name}.executed.ipynb")
    if [ "${ntb:-0}" -gt 0 ]; then
      echo "   ⚠️  Traceback 흔적 ${ntb}회 — 셀 안에서 잡힌 예외가 있습니다."
    fi
    echo "   출력 확인: python tools/show_nb_outputs.py $OUT/${name}.executed.ipynb"
  else
    el=$(( $(date +%s) - t0 ))
    echo "   ❌ 실패 (${el}초)"
    PASS_FAIL_LOG="$OUT/${name}.log"
    echo "   --- 실패 지점 ---"
    # nbconvert 는 실패한 셀 소스와 예외를 함께 찍는다. 핵심만 뽑는다.
    grep -nE "^(Error|.*Error:|.*Exception:|CellExecutionError)" "$PASS_FAIL_LOG" | head -n 8 | sed 's/^/   /'
    echo "   --- 직전 컨텍스트 ---"
    tail -n 25 "$PASS_FAIL_LOG" | sed 's/^/   /'
    echo "   (전체 로그: $PASS_FAIL_LOG)"
    FAIL+=("$name")
  fi
  echo
done

echo "======================================================================"
echo " 결과"
echo "======================================================================"
echo " 완주 ${#PASS[@]}종: ${PASS[*]:-없음}"
echo " 실패 ${#FAIL[@]}종: ${FAIL[*]:-없음}"
echo
echo " 실행 결과 노트북: $OUT/*.executed.ipynb"
echo " 전체 로그       : $OUT/*.log"
if [ ${#FAIL[@]} -gt 0 ]; then
  echo
  echo " 실패한 것의 로그를 회신해 주세요:"
  for f in "${FAIL[@]}"; do echo "   tail -n 60 $OUT/${f}.log"; done
fi
