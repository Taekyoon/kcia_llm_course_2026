# HPC LLM 강의 실습 자료 (2026)

LLM 데이터 처리·파인튜닝 3일 과정(21H)의 **실습 노트북과 실행 환경**입니다.
VESSL 워크스페이스에서 clone 해서 바로 돌리는 것을 전제로 구성했습니다.

> 이 브랜치(`main`)가 **수강생용**입니다. 강의 슬라이드는 별도로 제공됩니다.

---

## 빠른 시작

```bash
git clone https://github.com/Taekyoon/kcia_llm_course_2026.git
cd kcia_llm_course_2026
bash setup_vessl.sh
```

`setup_vessl.sh` 가 학습 스택을 설치하고 vLLM 을 별도 venv 에 격리합니다. 5~10분 걸립니다.
**처음 한 번만** 실행하면 됩니다.

그다음 Jupyter 에서 `work/notebook/` 의 노트북을 **일차 순서 → 파일명 앞 숫자 순서**로 엽니다.

---

## 구성

```
work/notebook/     ← 실습 노트북. 일차별 폴더, 파일명 앞 숫자가 진행 순서
  1일차/  2일차/  3일차/
assets/            ← SFT 실습용 사전생성 데이터 (강사 제공)
setup_vessl.sh     ← 환경 구성 (처음 한 번)
```

노트북은 자유롭게 고치고 실험해도 됩니다. 원래대로 돌리고 싶으면 `git checkout -- work/notebook/` 으로 되돌리세요.

---

## 실습 목록 (진행 순서)

| 일차 | 순서 | 노트북 | 내용 | 모델 | vLLM 서버 |
|---|---|---|---|---|---|
| 1 | 0 | `Classification실습` | 감정 분류 (NSMC) | `jhu-clsp/mmBERT-base` | 불필요 |
| 1 | 1 | `NER실습` | 개체명 인식 (KLUE-NER) | `jhu-clsp/mmBERT-base` | 불필요 |
| 2 | 0 | `데이터처리실습` | 한국어 위키 코퍼스 정제 | — | 불필요 |
| 2 | 1 | `MiniGPT실습` | 미니 GPT 사전학습 + 이어학습(CPT) | 자체 학습 (PyTorch) | 불필요 |
| 2 | 2 | `Amazon요약실습` | LLM 으로 학습 데이터 만들기 | `Qwen/Qwen3-4B-Instruct-2507` | **필요** |
| 2 | 3 | `프롬프트최적화실습` | 프롬프트 자동 최적화 (DSPy GEPA) | `Qwen/Qwen3-4B-Instruct-2507` | **필요** |
| 3 | 0 | `BM25_RAG실습` | 검색 증강 생성 (RAG) | `Qwen/Qwen3-4B-Instruct-2507` | **필요** |
| 3 | 1 | `평가실습` | BLEU/ROUGE · LLM-as-judge | `Qwen/Qwen3-4B-Instruct-2507` (채점) | **필요** |
| 3 | 2 | `퓨샷실습` | 프롬프트 엔지니어링 · 구조화 출력 | `Qwen/Qwen3-4B-Instruct-2507` | **필요** |
| 3 | 3 | `SFT실습` | 인스트럭션 파인튜닝 (LoRA) | `Qwen/Qwen3-0.6B-Base` | 불필요 |
| 3 | 4 | `DPO실습` | 선호 기반 학습 | `Qwen/Qwen3-0.6B-Base` | 불필요 |
| 3 | 5 | `GRPO실습` | 리즈닝 학습 (채점 함수) | `Qwen/Qwen3-0.6B-Base` | 불필요 |
| 3 | 6 | `RAG개선실습` | 학습 전후를 RAG 에 꽂아 비교 | `Qwen3-0.6B-Base` + SFT 어댑터 | 불필요 |

모델은 전부 **상업 이용 가능**(MIT / Apache-2.0)하고 로그인(HF 토큰) 없이 받아집니다.

---

## 노트북 사이의 의존 관계 — 순서를 지켜야 하는 이유

앞 실습이 만든 파일을 뒤 실습이 읽습니다. 순서를 건너뛰면 "파일이 없다"에서 멈춥니다.

| 앞 실습 → 만드는 것 | 쓰는 뒤 실습 |
|---|---|
| 데이터처리 → `ko_wiki_clean.jsonl` | MiniGPT (이어학습 코퍼스) |
| Amazon요약 → `amazon_ko_sft.mine.jsonl` | SFT (학습 데이터) — `assets/` 에 강사 사전생성본이 있어 **Amazon 을 못 돌려도 SFT 는 됩니다** |
| BM25_RAG → `bm25_retriever/` | RAG개선 (같은 검색 인덱스를 다시 씀) |
| SFT → `data/sft_model` | DPO · RAG개선 — 평가실습은 SFT 가 없으면 **예시 응답으로 진행**됩니다 |

---

## vLLM 서버 켜기 · 끄기

**GPU 가 하나라 서빙 실습과 학습 실습을 같이 못 돌립니다.** vLLM 이 기동하면서 GPU 80% 를 선점하기 때문에, 서버를 띄운 채 학습 노트북을 돌리면 이렇게 죽습니다:

```
OutOfMemoryError: Tried to allocate 20.00 MiB ... 4.75 MiB is free
```

날마다 전환 지점이 한 번씩 있습니다:

| 일차 | 서버 **끄고** | 서버 **켜고** |
|---|---|---|
| 2 | 데이터처리 · MiniGPT | Amazon요약 · 프롬프트최적화 |
| 3 | SFT · DPO · GRPO · RAG개선 | BM25_RAG · 평가 · 퓨샷 |

즉 **2일차는 MiniGPT 가 끝나면 켜고, 3일차는 퓨샷이 끝나면 끕니다.**

### 켜기

```bash
source /opt/vllm-env/bin/activate
nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 \
    --gpu-memory-utilization 0.80 \
    --max-model-len 16384 \
    > /tmp/vllm.log 2>&1 &

tail -f /tmp/vllm.log     # "Application startup complete" 가 뜨면 준비 완료
```

두 플래그는 **반드시** 붙이세요.
- `--gpu-memory-utilization 0.80` — 기본값 0.92 는 Jupyter 커널이 GPU 를 조금만 잡고 있어도 기동에 실패합니다.
- `--max-model-len 16384` — 이 모델의 기본 컨텍스트가 262,144 토큰이라 그대로 두면 메모리를 크게 낭비합니다.

### 끄기

```bash
pkill -f 'vllm serve' ; sleep 5 ; nvidia-smi     # 메모리가 반환됐는지 확인
```

학습 노트북으로 넘어가기 전에 **Jupyter 커널도 재시작**하는 편이 안전합니다 (커널이 3~4GB 를 쥐고 있습니다).

---

## 실행 환경 (참고)

```
Python 3.13 · CUDA 13.0 · torch 2.9 (cu130)
GPU    NVIDIA L40S 44GB
```

- vLLM 은 `setup_vessl.sh` 가 **별도 venv**(`/opt/vllm-env`)에 격리해 둡니다. 기본 커널에 `pip install vllm` 하지 마세요 — torch 가 통째로 교체됩니다.
- 위 서빙 명령의 `source /opt/vllm-env/bin/activate` 가 그 venv 를 여는 것입니다.
