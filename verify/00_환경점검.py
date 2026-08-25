# =====================================================================
# [VESSL 점검 1/2] 환경 스펙 확인 — 노트북 셀에 통째로 붙여넣고 실행
#
#  * 아무것도 설치하지 않습니다. 읽기만 합니다.
#  * 출력 전체를 그대로 복사해서 회신해 주세요.
# =====================================================================
import json
import os
import platform
import shutil
import subprocess
import sys

REPORT = {}


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<실행 실패: {e}>"


print("=" * 72)
print("1. 시스템")
print("=" * 72)
REPORT["python"] = sys.version.split()[0]
REPORT["platform"] = platform.platform()
print(f"Python      : {sys.version}")
print(f"실행 경로   : {sys.executable}")
print(f"플랫폼      : {platform.platform()}")
print(f"CPU 코어    : {os.cpu_count()}")

total, used, free = shutil.disk_usage("/")
REPORT["disk_free_gb"] = round(free / 2**30, 1)
print(f"디스크(/)   : 전체 {total/2**30:.1f}GB / 사용 {used/2**30:.1f}GB / 여유 {free/2**30:.1f}GB")

home = os.path.expanduser("~")
t2, u2, f2 = shutil.disk_usage(home)
print(f"디스크(홈)  : 여유 {f2/2**30:.1f}GB   ({home})")

print()
print("=" * 72)
print("2. GPU / CUDA   ※ 계획서 요구사양: A100 40G 이상")
print("=" * 72)
nv = sh("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap "
        "--format=csv,noheader")
print(f"nvidia-smi  : {nv}")
REPORT["nvidia_smi"] = nv
print(f"CUDA 컴파일러: {sh('nvcc --version | tail -n 2')}")

try:
    import torch

    REPORT["torch"] = torch.__version__
    REPORT["cuda"] = torch.version.cuda
    print(f"torch       : {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"GPU 사용가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gb = p.total_memory / 2**30
            print(f"  [{i}] {p.name}  {gb:.1f}GB  sm_{p.major}{p.minor}")
            REPORT[f"gpu{i}"] = f"{p.name} {gb:.1f}GB"
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"  현재 여유 VRAM: {free_b/2**30:.1f}GB / {total_b/2**30:.1f}GB")
except ImportError:
    print("torch       : 미설치")
    REPORT["torch"] = None

print()
print("=" * 72)
print("3. 사전 설치된 패키지 버전")
print("=" * 72)
PKGS = [
    "transformers", "datasets", "trl", "peft", "accelerate", "evaluate",
    "huggingface_hub", "tokenizers", "safetensors", "vllm", "openai",
    "llama-index-core", "llama-index-retrievers-bm25", "llama-index-llms-vllm",
    "keras", "keras-hub", "keras-nlp", "tensorflow", "tensorflow-text",
    "seqeval", "math-verify", "PyStemmer", "bitsandbytes", "flash-attn",
]
try:
    from importlib.metadata import version
except ImportError:
    version = None

for p in PKGS:
    try:
        v = version(p)
    except Exception:  # noqa: BLE001
        v = "-"
    REPORT[p] = v
    print(f"  {p:<32} {v}")

print()
print("=" * 72)
print("4. 네트워크 / 허브 접근")
print("=" * 72)
print(f"HF_HOME     : {os.environ.get('HF_HOME', '(미설정)')}")
print(f"HF_TOKEN    : {'설정됨' if os.environ.get('HF_TOKEN') else '(미설정)'}")
print(f"프록시      : {os.environ.get('HTTPS_PROXY', '(없음)')}")

for name, url in [
    ("huggingface.co", "https://huggingface.co/api/models/Qwen/Qwen3-0.6B-Base"),
    ("pypi.org", "https://pypi.org/pypi/trl/json"),
]:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=15) as r:
            print(f"  {name:<20} 도달 가능 (HTTP {r.status})")
    except Exception as e:  # noqa: BLE001
        print(f"  {name:<20} 실패: {type(e).__name__} {e}")

print()
print("=" * 72)
print("5. 판정 요약")
print("=" * 72)
gpu_line = REPORT.get("nvidia_smi", "")
print(f"GPU        : {gpu_line or '확인 실패'}")
print("A100 40G 이상 충족 여부 → 위 GPU 정보로 판단 (자동 판정하지 않음)")
print()
print("--- 아래 JSON 한 줄도 함께 회신해 주세요 ---")
print(json.dumps(REPORT, ensure_ascii=False))
