# HPC LLM 강의 실습 자료 (2026) — 제작·검토용 브랜치 (`dev`)

LLM 데이터 처리·파인튜닝 3일 과정(21H) 자료를 **만들고 검증하는** 쪽의 브랜치입니다.
수강생이 받는 것은 `main` 이고, 이 브랜치는 그것을 **생성하는 도구·원본·검증 기록**을 함께 가집니다.

> 수강생 안내(실습 순서 · 의존 관계 · 서버 전환)는 `main` 의 README 를 보세요. 여기서는 반복하지 않습니다.

---

## 브랜치 구조

| ref | 용도 | 내용 |
|---|---|---|
| `main` | **수강생 배포** | README · `setup_vessl.sh` · `assets/` · `work/notebook/` 13종 (19파일) |
| `2026` | 2026 에디션 | main 의 2026년 확정 시점 (브랜치) |
| `v2026` | 2026 확정본 태그 | 위와 같은 커밋의 **불변** 스냅샷 |
| `dev` | **제작·검토** | main + `tools/` · `verify/` · 25년 원본 `notebook/` · 슬라이드 자산 |

`main` 은 dev 에서 제작용 파일을 `git rm` 한 뒤의 트리입니다. 히스토리는 공유합니다 —
과거 커밋에는 전부 남아 있고, `main` 의 **현재 트리**에서만 제외됩니다.

수강생용 파일(`work/notebook/`·`setup_vessl.sh`·`assets/`)을 고치면 **dev 에서 만들고 main 으로 옮깁니다.**

---

## 구성

```
work/notebook/     ← 생성된 실습 노트북 13종 (수강생에게 가는 것 · 손으로 고치지 않음)
notebook/          ← 25년 원본 9종 (읽기 전용 · 재생성의 출발점)
tools/             ← 노트북·슬라이드 생성기 + 추출·검증 스크립트 (uv 프로젝트)
verify/            ← VESSL 점검 스크립트와 실측 결과
assets/            ← 강사 사전생성 데이터 (SFT 용 amazon_ko_sft.jsonl.gz)
work/assets/       ← 슬라이드에 넣는 이미지 (p25 사과 · 스케일링 법칙 그래프)
setup_vessl.sh     ← 수강생 환경 구성
```

**git 에 없는 것** (`.gitignore`, 로컬에만 보관):
- `ppt/` — 25년 원본 슬라이드 3덱. **슬라이드 빌드에 필요**하므로 clone 만으로는 슬라이드를 못 만듭니다.
- `work/ppt/` — 빌드된 슬라이드 산출물
- `*.hwp` — 발주처 커리큘럼 (승인 확정본)
- `review/` — 내부 검토 리포트

---

## 노트북 생성

### `work/notebook/` 은 손으로 고치지 마세요

이 폴더는 **스크립트가 매번 새로 생성**합니다. 직접 편집하면 다음 빌드에서 덮어써집니다.
항상 pristine 원본에서 시작하므로 몇 번을 돌려도 결과가 같습니다.

노트북마다 생성원이 다릅니다:

| 노트북 | 생성원 | 방식 |
|---|---|---|
| Classification · NER · 퓨샷 · SFT · DPO · GRPO · Amazon요약 · BM25_RAG (8종) | `tools/migrate_notebooks.py` | 25년 원본에 **규칙 기반 치환** (규칙마다 `why` 사유 기록) |
| MiniGPT | `tools/build_minigpt_notebook.py` | **새로 작성** (Keras → PyTorch/HF 전면 재작성) |
| 데이터처리 | `tools/build_datacleaning_notebook.py` | 새로 작성 |
| 평가 | `tools/build_eval_notebook.py` | 새로 작성 |
| 프롬프트최적화 | `tools/build_promptopt_notebook.py` | 새로 작성 |
| RAG개선 | `tools/build_ragcheck_notebook.py` | 새로 작성 — **SFT 노트북을 읽으므로 migrate 다음에** 실행 |

### 빌드

```bash
# repo 루트에서 실행합니다
uv run --project tools python tools/migrate_notebooks.py --dry-run   # 규칙 적용 미리보기
uv run --project tools python tools/migrate_notebooks.py             # 8종 생성 (먼저)
uv run --project tools python tools/build_minigpt_notebook.py
uv run --project tools python tools/build_datacleaning_notebook.py
uv run --project tools python tools/build_eval_notebook.py
uv run --project tools python tools/build_promptopt_notebook.py
uv run --project tools python tools/build_ragcheck_notebook.py       # SFT 노트북 의존 → 마지막
uv run --project tools python tools/validate_notebooks.py            # 정적 검사
uv run --project tools python tools/check_prose.py                   # 분량·용어·설명 공백 검사
```

내용을 바꾸려면 노트북이 아니라 **해당 생성 스크립트**를 고칩니다.

### 배치와 경로는 한 곳에서

- **`tools/layout.py`** — 노트북의 일자 배치(SSOT). 파일명 앞 숫자(`0_`, `1_`…)가 여기서 정해집니다.
  일자 문자열을 다른 파일에 적지 마세요 — 재배치 때 한 곳이 빠지면 같은 노트북이 두 위치에 남습니다.
- **`tools/nbcommon.py`** — 여러 노트북이 글자 하나까지 같아야 하는 셀(`DATA_DIR` 확정 등).
  노트북 사이에 파일이 흐르므로(데이터처리→MiniGPT, Amazon→SFT, RAG→RAG개선, SFT→DPO·RAG개선)
  경로 코드는 이 한 벌만 씁니다. `HPC_DATA` 환경변수 → repo 루트 탐색 순이고, 둘 다 실패하면 raise 합니다.

---

## 슬라이드 생성

원본 3덱(`ppt/`, 로컬)에서 새 3덱(`work/ppt/`)을 **매번 다시** 조립합니다.

```bash
uv run --project tools python tools/migrate_slides.py            # 재조립 (1·2·3일차)
uv run --project tools python tools/validate_slides.py --twice   # 장수·rId·라운드트립·원본불가침·결정론
```

- **`tools/slide_layout.py`** — 장 순서 SSOT. 원본 참조(`refs("3일차", 25, 53)`)와 신규 장 마커(`NewSlide`)를 여기에만 적습니다.
- **`tools/migrate_slides.py`** — 치환 규칙(`SlideRule`·`GlobalSlideRule`), 신규 장 원고(`NEW_SLIDES`), 도형 덧붙임(`EXTRAS`), 도해 함수.
  지적을 반영할 때는 새 덱 페이지가 아니라 **원본 (덱, 페이지)** 로 주소를 잡습니다 — 재배치와 무관하게 맞습니다.
- 빌드 시 시각 서식을 통일합니다(코드 폰트 Consolas · 글머리표 ▪/✓ · 제목을 소제목 박스로). 원본 pptx 는 건드리지 않습니다.
- **산출물이 PowerPoint 에 열려 있으면 빌드가 막힙니다.** 닫고 실행하세요.
  (닫았는데도 막히면 `POWERPNT` 프로세스가 남은 것 — 강제 종료 후 재시도)

원본 불가침 기준선(크기·해시)은 `validate_slides.py` 가 매번 확인합니다.

---

## 도구 목록 (`tools/`)

| 스크립트 | 용도 |
|---|---|
| `migrate_notebooks.py` | 25년 원본 → 2026 스택 규칙 기반 마이그레이션 (8종) |
| `build_*_notebook.py` | 신규 노트북 5종 생성 (minigpt · datacleaning · eval · promptopt · ragcheck) |
| `build_amazon_pregen.py` | Amazon 파이프라인을 대량으로 돌려 `assets/` 사전생성 데이터 제작 |
| `layout.py` · `nbcommon.py` | 노트북 배치 SSOT · 공용 셀 소스 |
| `validate_notebooks.py` · `check_prose.py` | 노트북 정적 검사 · 산문 점검 |
| `migrate_slides.py` · `slide_layout.py` · `pptxcommon.py` | 슬라이드 재조립기 · 배치 SSOT · OPC 복제 프리미티브 |
| `validate_slides.py` · `check_slide_sync.py` | 슬라이드 무결성 · 슬라이드↔노트북 코드 싱크 검사 |
| `extract_pptx.py` · `extract_ipynb.py` · `extract_hwp.py` · `extract_all.py` | 원본을 텍스트로 추출 (HWP 는 COM 없이 CFB 직접 파싱) |
| `check_amazon_*.py` · `check_minhash.py` · `count_targets.py` | 개별 정합성 검사 |
| `show_nb_outputs.py` | 실행된 노트북의 셀 출력 확인 |
| `build_report_docx.py` | 발주처 제출 보고서 마크다운 → .docx |

전부 `tools/` 의 uv 프로젝트로, **repo 루트에서** `uv run --project tools python tools/…` 로 돌립니다. 로컬에 시스템 Python 을 두지 않습니다.

---

## 검증 (`verify/`)

VESSL 워크스페이스에서 **셀에 붙여넣어 한 번에** 실행하는 점검입니다. 결과는 `verify/결과.md` 에 기록합니다.

| 파일 | 용도 |
|---|---|
| `00_환경점검.py` | GPU · CUDA · 패키지 버전 (설치 없음) |
| `01_스택설치.py` | 스택 설치 + 데이터셋·API 실측 |
| `02_vllm_venv.sh` | vLLM 별도 venv 구성 (`setup_vessl.sh` 에 포함) |
| `03_수정본검증.py` | 마이그레이션 결과 검증 |
| `04_인코더비교.py` | 인코더 모델 성능 비교 (mmBERT vs KoELECTRA) |
| `05_vllm검증.py` | 서빙 · structured output · RAG 프롬프트 검증 |
| `06_노트북실행.sh` | 노트북 13종 무인 완주 (`nbconvert`) |
| `07_dspy설치.sh` | DSPy/GEPA 설치 및 실측 |

원칙: **실행 결과를 받지 못한 항목은 "미검증"으로 적습니다.** 추측을 사실처럼 쓰지 않고, 실패는 에러 원문을 그대로 인용합니다.

---

## 25년 자료에서 바뀐 것

2025년 5월 이후 HuggingFace 생태계가 메이저를 두 번 올려서 상당수가 그냥은 돌지 않았습니다.

| 라이브러리 | 25년 자료 | 2026 |
|---|---|---|
| transformers | 4.51 | **5.16** |
| datasets | 3.5.1 (핀) | **5.0.1** |
| trl | 0.17 | **1.12** |
| vLLM | 0.8 | **0.27.1** |

**데이터셋** — `datasets` 5.x 가 로딩 스크립트를 지원하지 않아 4건 교체
- `nsmc` → `e9t/nsmc` (`refs/convert/parquet`) · `kor_ner` → `klue/klue` (`ner`)
- `heegyu/kowikitext` → `wikimedia/wikipedia` (`20231101.ko`) · `beomi/KoAlpaca-v1.1a` → `nlpai-lab/kullm-v2` (NC 라이선스)

**API** — transformers 5 / TRL 1.x
- `Trainer(tokenizer=)` · `trainer.tokenizer` → `processing_class` · `SFTConfig.max_seq_length` → `max_length`
- `warmup_ratio` · `overwrite_output_dir` · `GRPOConfig.max_prompt_length` 제거

**vLLM** — structured output 이 **조용히 무시되던 것** (에러 없이 자유 생성)
- `guided_choice` → `structured_outputs` · `guided_json` → `response_format` · `api_server` → `vllm serve`
- `response_format` 쓸 때 `max_tokens` 하드캡 + 스키마 `maxLength` + `finish_reason=="length"` 검사가 필요

**LlamaIndex** — `get_prompts()` 는 deepcopy 를 돌려줘 고쳐도 무시됨 → `update_prompts()`

**모델** — 라이선스·커리큘럼 정합성
- EXAONE 계열은 NC 라이선스라 제외. 분류·NER 은 커리큘럼(BERT/MLM)에 맞춰 **인코더로 전환** (`jhu-clsp/mmBERT-base`)
- 학습 `Qwen/Qwen3-0.6B-Base` · 서빙 `Qwen/Qwen3-4B-Instruct-2507` (전부 MIT/Apache-2.0, ungated)

**LoRA 재로드 함정** — 커스텀 챗 템플릿은 코드 지정이라 어댑터 저장에 안 따라옴. 재로드 후 다시 넣지 않으면 빈 출력 (RAG개선에서 실제로 겪음).

상세 실측은 `verify/결과.md`, 노트북별 변경 건수는 `work/README.md` 를 보세요.

---

## 실행 환경 (VESSL, 2026-08 실측)

```
Python 3.13 · CUDA 13.0 · torch 2.9.1+cu130
GPU    NVIDIA L40S 44.4GB (sm_89) — 계획서 표기 "A100 40G" 와 다름
```

- vLLM 은 `torch==2.13` 을 등호로 하드핀하므로 기본 커널에 설치하면 torch 가 통째로 교체됩니다 → `setup_vessl.sh` 가 `/opt/vllm-env` 로 격리.
- vLLM 은 cu130 휠이 없어 cu129 빌드를 씁니다 — 드라이버 하위 호환으로 **실측 정상 동작**.
- 로컬 Windows 는 분석·저작 전용입니다. 코드가 도는지의 기준은 **VESSL** 뿐입니다.
