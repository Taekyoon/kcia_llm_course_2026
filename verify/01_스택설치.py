# =====================================================================
# [VESSL 점검 2/2] 최신 스택 설치 + 파손 지점 실측
#
#  ※ 00_환경점검.py 결과를 반영해 재작성했습니다 (2026-08-25).
#     환경: Python 3.13.9 / CUDA 13.0 / torch 2.9.1+cu130 / L40S 44.4GB
#           그리고 ML 패키지가 torch 외에는 전혀 설치돼 있지 않음
#
#  ▶ 이 스크립트는 패키지를 설치합니다. 되돌릴 수 있는 상태에서 실행하세요.
#  ▶ 출력 전체를 그대로 복사해서 회신해 주세요.
#  ▶ 예상 소요: 5~15분 (다운로드 포함)
#
#  핵심 안전장치: torch 2.9.1 을 constraints 로 못박아 재설치를 차단합니다.
#     vLLM 0.27.1 은 torch==2.13.0 을 하드핀하므로 여기서 설치하지 않습니다.
#     (설치하면 torch 2.9.1+cu130 이 삭제되고 2.13.0+cu129 로 교체됨)
#     vLLM 은 §E 안내대로 별도 venv 에서 다룹니다.
# =====================================================================
import json
import subprocess
import sys
import traceback

PY = sys.executable
CONSTRAINTS = "/tmp/torch_pin.txt"
RESULT = {}


def step(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def run(cmd, tail=25):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    lines = out.splitlines()
    print("\n".join(lines[-tail:]) if len(lines) > tail else out)
    return r.returncode, out


def probe(key, fn):
    """실패해도 멈추지 않고 에러 원문을 기록한다."""
    try:
        out = fn()
        RESULT[key] = {"ok": True, "detail": str(out)[:300]}
        print(f"  [OK]   {key}: {str(out)[:200]}")
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        RESULT[key] = {"ok": False, "detail": msg[:600]}
        print(f"  [FAIL] {key}")
        print("         " + msg[:600].replace("\n", "\n         "))


# ---------------------------------------------------------------------
step("A. 설치 전 상태 확인")
# ---------------------------------------------------------------------
print("--- torch (보호 대상) ---")
import torch  # noqa: E402

TORCH_BEFORE = torch.__version__
RESULT["torch_before"] = TORCH_BEFORE
print(f"  torch {TORCH_BEFORE}  CUDA {torch.version.cuda}  사용가능={torch.cuda.is_available()}")

print("\n--- 현재 설치된 패키지 (00 점검에서 '전부 미설치'로 나온 것 재확인) ---")
_, pipout = run(f"{PY} -m pip list --format=freeze", tail=200)
RESULT["pip_list_count"] = len(pipout.splitlines())
print(f"\n  총 {len(pipout.splitlines())}개 패키지")

# torch 를 못박는다. 이후 모든 pip install 에 -c 로 적용.
with open(CONSTRAINTS, "w") as f:
    f.write(f"torch=={TORCH_BEFORE.split('+')[0]}\n")
print(f"\n  constraints 파일 생성: {CONSTRAINTS}")
print(f"    torch=={TORCH_BEFORE.split('+')[0]}")

# ---------------------------------------------------------------------
step("B. 학습 스택 설치 (torch 재설치 차단)")
# ---------------------------------------------------------------------
PKGS = "transformers datasets accelerate peft trl evaluate bitsandbytes math-verify openai"
print(f"설치 대상: {PKGS}")
print("※ flash-attn 은 제외했습니다 — 휠이 없어 소스 빌드에 30분~2시간 걸리고,")
print("   attn_implementation='sdpa'(torch 내장)로 대체 가능해 강의에 불필요합니다.\n")

rc, _ = run(f"{PY} -m pip install -c {CONSTRAINTS} -U {PKGS}")
RESULT["install_train_rc"] = rc

# ---------------------------------------------------------------------
step("C. seqeval 설치 — 실패 가능성 있음 (NER 실습 직결)")
# ---------------------------------------------------------------------
print("seqeval 1.2.2 는 sdist 전용(휠 없음)이고 2020년 패키지라 구식 distutils 기반입니다.")
print("Python 3.12에서 distutils 가 표준 라이브러리에서 제거돼 3.13 빌드가 깨질 수 있습니다.\n")
rc, seq_out = run(f"{PY} -m pip install -c {CONSTRAINTS} seqeval")
RESULT["install_seqeval_rc"] = rc
if rc != 0:
    RESULT["seqeval_error"] = seq_out[-1500:]
    print("\n  ★ seqeval 설치 실패 — 위 에러 원문이 판정 근거입니다.")
    print("  ★ 대안: pip install seqeval --no-build-isolation  또는 지표 직접 구현")

# ---------------------------------------------------------------------
step("D. RAG 스택 설치")
# ---------------------------------------------------------------------
rc, _ = run(
    f"{PY} -m pip install -c {CONSTRAINTS} "
    "llama-index-core llama-index-retrievers-bm25 llama-index-llms-openai-like"
)
RESULT["install_rag_rc"] = rc

# ---------------------------------------------------------------------
step("E. torch 가 보존됐는지 확인 — 가장 중요")
# ---------------------------------------------------------------------
rc, tv = run(f"{PY} -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"", tail=5)
RESULT["torch_after"] = tv.strip()
same = TORCH_BEFORE.split("+")[0] in tv
print(f"\n  설치 전: {TORCH_BEFORE}")
print(f"  설치 후: {tv.strip()}")
print(f"  → {'보존됨 ✅' if same else '★ 변경됨! 원인 추적 필요 ★'}")
RESULT["torch_preserved"] = same

print("\n--- 의존성 충돌 검사 ---")
rc, chk = run(f"{PY} -m pip check", tail=30)
RESULT["pip_check"] = chk[:1500] or "(충돌 없음)"

print("\n--- 설치된 버전 ---")
from importlib.metadata import version  # noqa: E402

for p in ["torch", "transformers", "datasets", "trl", "peft", "accelerate",
          "evaluate", "huggingface_hub", "tokenizers", "seqeval",
          "llama-index-core", "llama-index-retrievers-bm25", "bitsandbytes"]:
    try:
        v = version(p)
    except Exception:  # noqa: BLE001
        v = "- (미설치)"
    RESULT[f"ver_{p}"] = v
    print(f"  {p:<32} {v}")

# ---------------------------------------------------------------------
step("F. 데이터셋 8종 로드 실측")
# ---------------------------------------------------------------------
from datasets import load_dataset  # noqa: E402

# 자료의 원본 호출을 그대로 재현하고, 대체안도 함께 시험한다.
probe("ds_nsmc_원본",     lambda: load_dataset("nsmc", split="train[:10]"))
probe("ds_nsmc_e9t원본",  lambda: load_dataset("e9t/nsmc", split="train[:10]"))
probe("ds_nsmc_대체",     lambda: load_dataset("e9t/nsmc", revision="refs/convert/parquet", split="train[:10]"))
probe("ds_kor_ner_원본",  lambda: load_dataset("kor_ner", split="train[:10]"))
probe("ds_klue_ner_대체", lambda: load_dataset("klue/klue", "ner", split="train[:10]"))
probe("ds_kowikitext",    lambda: load_dataset("heegyu/kowikitext", split="train[:10]"))
probe("ds_koalpaca",      lambda: load_dataset("beomi/KoAlpaca-v1.1a", split="train[:10]"))
probe("ds_kodpo",         lambda: load_dataset("sionic-ai/ko-dpo-mix-7k-trl-style", split="train[:10]"))
probe("ds_numina",        lambda: load_dataset("OLAIR/numina_math_ko_verifiable_540k", split="train[:10]"))
probe("ds_korquad",       lambda: load_dataset("KorQuAD/squad_kor_v1", split="train[:10]"))
probe("ds_amazon",        lambda: load_dataset("Taekyoon/test_amazon", split="train[:10]"))

print("\n--- klue/klue ner 라벨 스킴 (NER 노트북 재작성에 필요) ---")
try:
    d = load_dataset("klue/klue", "ner", split="train[:1]")
    print("  컬럼:", d.column_names)
    print("  features:", d.features)
    RESULT["klue_ner_columns"] = str(d.column_names)
    RESULT["klue_ner_labels"] = str(d.features)[:800]
except Exception as e:  # noqa: BLE001
    print("  실패:", e)

# ---------------------------------------------------------------------
step("G. API 변경 실측 (transformers / TRL)")
# ---------------------------------------------------------------------
import inspect  # noqa: E402


def sig_has(cls, name):
    return name in inspect.signature(cls.__init__).parameters


from transformers import Trainer, TrainingArguments  # noqa: E402

probe("Trainer_tokenizer인자",       lambda: sig_has(Trainer, "tokenizer"))
probe("Trainer_processing_class인자", lambda: sig_has(Trainer, "processing_class"))
probe("Trainer_tokenizer속성",       lambda: hasattr(Trainer, "tokenizer"))
probe("TA_eval_strategy",            lambda: sig_has(TrainingArguments, "eval_strategy"))
probe("TA_evaluation_strategy",      lambda: sig_has(TrainingArguments, "evaluation_strategy"))

from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer  # noqa: E402

probe("SFTConfig_max_seq_length", lambda: sig_has(SFTConfig, "max_seq_length"))
probe("SFTConfig_max_length",     lambda: sig_has(SFTConfig, "max_length"))
probe("SFTTrainer_tokenizer인자", lambda: sig_has(SFTTrainer, "tokenizer"))
probe("SFTTrainer_peft_config",   lambda: sig_has(SFTTrainer, "peft_config"))
probe("DPOTrainer_tokenizer인자", lambda: sig_has(DPOTrainer, "tokenizer"))
probe("GRPOConfig_vllm_mode기본", lambda: inspect.signature(GRPOConfig.__init__).parameters["vllm_mode"].default)
probe("GRPOTrainer_시그니처",     lambda: list(inspect.signature(GRPOTrainer.__init__).parameters))
probe("DPOConfig_필드",           lambda: list(inspect.signature(DPOConfig.__init__).parameters)[:25])

print("\n--- evaluate + datasets 5.x 런타임 공존 (미검증 조합) ---")
import evaluate  # noqa: E402

probe("evaluate_accuracy", lambda: evaluate.load("accuracy").compute(references=[0, 1], predictions=[0, 1]))
probe("evaluate_seqeval", lambda: evaluate.load("seqeval").compute(
    references=[["O", "B-PS"]], predictions=[["O", "B-PS"]]))

# ---------------------------------------------------------------------
step("H. 모델 로드 실측 (L40S bf16)")
# ---------------------------------------------------------------------
from transformers import AutoTokenizer  # noqa: E402

for mid in [
    "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "Qwen/Qwen3-0.6B-Base",
    "Qwen/Qwen2.5-0.5B-Instruct",
]:
    probe(f"tok_{mid.split('/')[-1]}", lambda m=mid: AutoTokenizer.from_pretrained(m).__class__.__name__)

probe("bf16_지원", lambda: torch.cuda.is_bf16_supported())

# ---------------------------------------------------------------------
step("I. vLLM — 설치하지 않았습니다. 별도 처리 필요")
# ---------------------------------------------------------------------
print("""
vllm 0.27.1 은 `torch==2.13.0` 을 등호로 하드핀하고, PyPI 기본 휠은 CUDA 12.9 빌드입니다.
그냥 `pip install vllm` 하면 지금 쓰는 torch 2.9.1+cu130 이 삭제되고 2.13.0+cu129 로 교체됩니다.
이 환경은 CUDA 13.0 이므로 cu130 전용 휠을 써야 합니다.

또한 trl 1.10.0 의 vllm extra 는 `vllm<=0.26.0` 상한이라 `pip install trl[vllm]` 은
vLLM 을 0.26.0 으로 되돌립니다. GRPO 실습에서 정면 충돌합니다.

▶ 권장: vLLM 전용 venv 분리. 학습용(torch 2.9.1)과 서빙용(torch 2.13.0)을 나눕니다.
  vLLM 은 `vllm serve` 로 별도 프로세스이므로 분리해도 실습에 지장이 없고,
  퓨샷/Amazon/RAG 노트북은 HTTP(OpenAI 호환 API)로 붙기 때문에 문제없습니다.

  python -m venv /opt/vllm-env
  source /opt/vllm-env/bin/activate
  pip install \\
    https://github.com/vllm-project/vllm/releases/download/v0.27.1/vllm-0.27.1+cu130-cp38-abi3-manylinux_2_28_x86_64.whl \\
    --extra-index-url https://download.pytorch.org/whl/cu130
  vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000        # EXAONE 은 NC 라이선스라 교체 대상

  # 다른 터미널/커널에서 확인 — guided_choice 가 제거됐는지
  from openai import OpenAI
  c = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
  M = "Qwen/Qwen2.5-3B-Instruct"
  try:
      r = c.chat.completions.create(model=M, messages=[{"role":"user","content":"좋아요"}],
                                    extra_body={"guided_choice": ["긍정","부정"]})
      print("guided_choice 아직 동작:", r.choices[0].message.content)
  except Exception as e:
      print("guided_choice 실패(예상됨):", type(e).__name__, e)

  r = c.chat.completions.create(model=M, messages=[{"role":"user","content":"좋아요"}],
          extra_body={"structured_outputs": {"choice": ["긍정","부정"]}})
  print("structured_outputs 결과:", r.choices[0].message.content)

▶ MiniGPT(keras-hub)도 별도 venv 를 권합니다.
  keras-hub → tensorflow-text<2.21 → tensorflow 2.20.x → nvidia-* cu12 패키지를 끌어옵니다.
  torch cu130 과 LD_LIBRARY_PATH 가 섞이면 충돌할 수 있습니다.
  (tensorflow-text 는 cp313 Linux 휠이 있으므로 설치 자체는 가능합니다)

  python -m venv /opt/keras-env && source /opt/keras-env/bin/activate
  pip install keras-hub          # tensorflow 를 명시하지 마세요. 짝이 자동으로 맞춰집니다
""")

# ---------------------------------------------------------------------
step("결과 요약")
# ---------------------------------------------------------------------
fails = [k for k, v in RESULT.items() if isinstance(v, dict) and not v.get("ok")]
print(f"torch 보존: {'예' if RESULT.get('torch_preserved') else '아니오 ★'}")
print(f"seqeval 설치: {'성공' if RESULT.get('install_seqeval_rc') == 0 else '실패 ★'}")
print(f"\n실패 항목 {len(fails)}건:")
for k in fails:
    print(f"  - {k}")
print()
print("--- 아래 JSON 도 함께 회신해 주세요 ---")
print(json.dumps(RESULT, ensure_ascii=False, indent=1)[:10000])
