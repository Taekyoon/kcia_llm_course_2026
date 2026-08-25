# HPC LLM 강의 실습 자료 (26년)

LLM 데이터 처리·파인튜닝 3일 과정(21H)의 **실습 노트북과 실행 환경**입니다.
VESSL 워크스페이스에서 clone 해서 바로 돌리는 것을 전제로 구성했습니다.

> ⚠️ **private repo 입니다.** 강의 슬라이드(`ppt/`), 발주처 커리큘럼(`*.hwp`),
> 내부 검토 문서(`review/`)는 `.gitignore` 로 **의도적으로 제외**했습니다.
> 그 자산들은 로컬에만 보관합니다.

---

## 빠른 시작

```bash
git clone https://github.com/Taekyoon/kcia_llm_course_2026.git
cd kcia_llm_course_2026
bash setup_vessl.sh
```

`setup_vessl.sh` 가 학습 스택을 설치하고 vLLM 을 별도 venv 에 격리합니다.
5~10분 걸립니다.

그다음 `work/notebook/` 의 노트북을 Jupyter 에서 열면 됩니다.

---

## 구성

```
work/notebook/     ← 실습에 쓸 노트북 (2026 스택으로 마이그레이션됨)
notebook/          ← 25년 원본 (읽기 전용 · 재생성 소스)
tools/             ← 마이그레이션 · 추출 스크립트
verify/            ← 환경 점검 및 검증 스크립트
setup_vessl.sh     ← 환경 구성 (한 번만)
```

### work/notebook/ 은 손으로 고치지 마세요

`tools/migrate_notebooks.py` 가 `notebook/` 원본에서 **매번 새로 생성**합니다.
직접 편집하면 다음 실행에서 덮어써집니다.

```bash
uv run python tools/migrate_notebooks.py --dry-run   # 미리보기
uv run python tools/migrate_notebooks.py             # 생성
```

수정 내용을 바꾸려면 스크립트 안의 **규칙표**를 고치세요.
규칙마다 변경 사유가 붙어 있고 실행하면 그대로 출력됩니다.

---

## 실습 목록

| 일차 | 노트북 | 모델 |
|---|---|---|
| 1 | `HPC_Classification실습` 감정분류(NSMC) | `jhu-clsp/mmBERT-base` |
| 1 | `HPC_NER실습` 개체명인식(KLUE-NER) | `jhu-clsp/mmBERT-base` |
| 1 | `HPC_MiniGPT실습` 미니 GPT 학습 | 자체 학습 (keras-hub) |
| 2 | `HPC_퓨샷실습` 프롬프트 엔지니어링 | `Qwen/Qwen3-4B-Instruct-2507` (서빙) |
| 2 | `HPC_SFT실습` 인스트럭션 파인튜닝 | `Qwen/Qwen3-0.6B-Base` |
| 2 | `HPC_DPO실습` 선호기반 학습 | `Qwen/Qwen3-0.6B-Base` |
| 2 | `HPC_GRPO실습` 리즈닝 학습 | `Qwen/Qwen3-0.6B-Base` |
| 3 | `HPC_Amazon요약실습` 데이터 생성 파이프라인 | `Qwen/Qwen3-4B-Instruct-2507` (서빙) |
| 3 | `HPC_BM25_RAG실습` 검색 증강 생성 | `Qwen/Qwen3-4B-Instruct-2507` (서빙) |

**모델은 전부 상업 이용 가능**합니다 (MIT / Apache-2.0, ungated).

---

## 25년 자료에서 바뀐 것 — 49건

2025년 5월 이후 HuggingFace 생태계가 메이저를 두 번 올려서 상당수가 그냥은 돌지 않습니다.

| 라이브러리 | 25년 자료 | 현재 |
|---|---|---|
| transformers | 4.51 | **5.15** |
| datasets | 3.5.1 (핀) | **5.0.1** |
| trl | 0.17 | **1.10** |
| vLLM | 0.8 | **0.27.1** |

### 주요 변경

**데이터셋** — `datasets` 5.x 가 로딩 스크립트를 지원하지 않아 4건이 실패했습니다.
- `nsmc` → `e9t/nsmc` 의 `refs/convert/parquet` 리비전
- `kor_ner` → `klue/klue` (`ner`)
- `heegyu/kowikitext` → `wikimedia/wikipedia` (`20231101.ko`)
- `beomi/KoAlpaca-v1.1a` → `nlpai-lab/kullm-v2` (라이선스 사유)

**API** — transformers 5 / TRL 1.x 변경
- `Trainer(tokenizer=)` · `trainer.tokenizer` → `processing_class`
- `SFTConfig.max_seq_length` → `max_length`
- `get_peft_model()` 선감싸기 → `peft_config=` 인자

**vLLM** — structured output 방식 변경
- `guided_choice` → `structured_outputs`
- `guided_json` → `response_format` (OpenAI 표준)
- `python -m vllm.entrypoints.openai.api_server` → `vllm serve`

**LlamaIndex** — 조용히 실패하던 코드 수정
- `get_prompts()` 는 deepcopy 를 반환해 반환값을 고쳐도 **에러 없이 무시**됩니다.
  다시 출력하면 바뀐 것처럼 보여서 더 위험했습니다 → `update_prompts()` 로 교체

**모델** — 라이선스 및 커리큘럼 정합성
- `EXAONE-3.5` 는 **NC(비상업) 라이선스**라 상업 강의에 쓸 수 없습니다(4.0 도 NC).
- 분류·NER 을 decoder-only 에서 **인코더로 전환**했습니다. 커리큘럼이 "Encoder는 BERT,
  MLM+NSP" 를 가르친 직후에 decoder 에 분류 헤드를 붙이고 있어 배운 것과 어긋났습니다.

---

## 환경 특성 (2026-08 실측)

```
Python 3.13.9 · CUDA 13.0 · torch 2.9.1+cu130
GPU    NVIDIA L40S 44.4GB (sm_89, Ada)
```

**주의할 점 두 가지:**

1. **vLLM 을 기본 커널에 설치하면 안 됩니다.** `torch==2.13.0` 을 등호로 하드핀해서
   torch 가 통째로 교체됩니다. `setup_vessl.sh` 가 별도 venv 로 격리합니다.
2. **vLLM 은 cu130 휠을 배포하지 않습니다.** cu129 빌드를 쓰며, pip 휠이 가져오는
   `nvidia-*-cu12` 런타임과 드라이버 하위 호환에 의존합니다.

---

## 검증

`verify/` 에 환경 점검 스크립트가 있습니다. 노트북 셀에 붙여넣어 실행합니다.

| 파일 | 용도 |
|---|---|
| `00_환경점검.py` | GPU · CUDA · 패키지 버전 (아무것도 설치하지 않음) |
| `01_스택설치.py` | 스택 설치 + 데이터셋·API 실측 |
| `02_vllm_venv.sh` | vLLM venv 구성 (setup_vessl.sh 에 포함됨) |
| `03_수정본검증.py` | 마이그레이션 결과 검증 |
| `04_인코더비교.py` | 인코더 모델 성능 비교 |
| `05_vllm검증.py` | 서빙·structured output·RAG 프롬프트 검증 |
