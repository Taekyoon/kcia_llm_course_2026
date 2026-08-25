# work/ — 수정 작업 영역

`notebook/` 원본은 읽기 전용입니다(CLAUDE.md §1). 모든 수정은 여기서 합니다.

## work/notebook/ 는 손으로 고치지 마세요

이 폴더는 **스크립트가 매번 새로 생성**합니다. 직접 편집하면 덮어써집니다.
노트북에 따라 생성원이 다릅니다.

| 노트북 | 생성원 |
|---|---|
| 8종 (MiniGPT 제외) | `tools/migrate_notebooks.py` — 원본에서 규칙 기반 치환 |
| **MiniGPT** | `tools/build_minigpt_notebook.py` — **새로 작성** (원본 변환 아님) |

```bash
uv run python tools/migrate_notebooks.py --dry-run   # 미리보기
uv run python tools/migrate_notebooks.py             # 8종 생성
uv run python tools/build_minigpt_notebook.py        # MiniGPT 생성
uv run python tools/validate_notebooks.py            # 9종 정적 검사
```

수정 내용을 바꾸려면 해당 스크립트를 고치세요.
`migrate_notebooks.py` 는 규칙마다 변경 사유(`why`)가 붙어 있고 실행하면 그대로 출력됩니다.

### 왜 이렇게 만들었나

사본을 제자리에서 고치는 방식으로 먼저 만들었다가 문제를 겪었습니다.
규칙의 `new` 문자열이 `old` 를 포함하면 **재실행할 때마다 중복 적용**되어
`update_prompts()` 호출이 여러 번 삽입되는 식으로 사본이 오염됐습니다.

항상 pristine 원본에서 시작하면 이 문제가 원천적으로 사라지고,
몇 번을 돌려도 결과가 같습니다(해시 동일 확인).

---

## 현재 적용된 변경 — 49건 / 노트북 9종

| 노트북 | 건수 | 주요 변경 |
|---|---:|---|
| 1일차 Classification | 6 | 인코더 전환 · nsmc parquet · processing_class · 체크포인트 하드코딩 제거 |
| 1일차 NER | 9 | 인코더 전환 · klue/klue ner · `.replace()` 버그 · test→validation |
| 1일차 MiniGPT | **전면 재작성** | Keras/TF → **PyTorch + HuggingFace**. 아래 참조 |
| 2일차 SFT | 6 | kullm-v2 교체 · **`input` 컬럼 병합** · `max_seq_length`→`max_length` |
| 2일차 DPO | 3 | pip 정리 · processing_class |
| 2일차 GRPO | 4 | Qwen3 통일 · processing_class |
| 2일차 퓨샷 | 6 | Qwen3-4B(18곳) · guided_choice/json 제거 · vllm serve |
| 3일차 Amazon 요약 | 10 | Qwen3-4B · `.beta.parse`→`create` · guided_json 6곳 → response_format |
| 3일차 BM25 RAG | 5 | OpenAILike HTTP · **wikimedia/wikipedia** · **update_prompts()** |

상세 근거: [review/02_문제점.md](../review/02_문제점.md) · [verify/결과.md](../verify/결과.md)

### MiniGPT 전면 재작성 (2026-08-25)

원본은 9종 중 **혼자만 Keras/TensorFlow** 였습니다. keras-hub API 자체는 유효했지만:

- `keras-hub` → `tensorflow-text` → `tensorflow 2.20` → `nvidia-*-cu12` 를 끌어와
  기본 커널의 torch cu130 과 충돌 → **별도 venv 가 필요**했습니다
- 수강생이 나머지 8종에서 배운 도구(transformers·datasets·Trainer)를
  여기서만 다시 배워야 했습니다
- 슬라이드도 어차피 재작성 대상이었습니다 (C-2: 1일차 p63·p69-90 이 구 `keras_nlp` API)

**교육 목표는 그대로 유지**하고 스택만 바꿨습니다.

| 항목 | 원본 (Keras) | 신규 (PyTorch/HF) |
|---|---|---|
| 토크나이저 | `compute_word_piece_vocabulary` | `tokenizers` **ByteLevel BPE** 직접 학습 |
| BOS/EOS | `StartEndPacker` | `TemplateProcessing` post-processor |
| 데이터 | `tf.data` 파이프라인 | `datasets` map/filter + 청킹 |
| 모델 | `TransformerDecoder` 레이어 조립 | **`nn.Module` 로 직접 구현** |
| 학습 | `model.compile/fit` | `Trainer` |
| 생성 | `keras_hub.samplers` 5종 | `model.generate()` 플래그 5종 |
| 데이터셋 | SimpleBooks (영어, S3 직링크) | `g0ster/TinyStories-Korean` (MIT) |

**의도적으로 바꾼 것 두 가지:**

- `NUM_HEADS` 3 → **4**. PyTorch 표준 구현은 `n_embd` 가 `n_head` 로 나누어떨어져야
  합니다(256/3 은 안 됩니다). Keras `MultiHeadAttention` 은 `key_dim` 이 별도라 3이 가능했습니다.
- WordPiece → **ByteLevel BPE**. 한국어는 음절 종류가 많아 사전 5,000 짜리 WordPiece 로는
  `[UNK]` 가 대량 발생합니다. ByteLevel 은 UNK 가 원천적으로 없습니다.

**Causal Self-Attention 을 직접 구현**하는 것이 핵심 변화입니다.
1일차 p49-58 에서 배운 Self-Attention·Multi-head 가 코드로 바로 연결됩니다.
`keras_hub.layers.TransformerDecoder` 한 줄로는 그 연결이 보이지 않았습니다.

---

## 검증 상태 — 2026-08-25 VESSL 실측

[verify/03_수정본검증.py](../verify/03_수정본검증.py) 실행 결과 **13개 중 12개 통과**.

| 확인 | 결과 |
|---|---|
| mmBERT 인코더 로드 · `is_fast` · `word_ids()` | ✅ |
| **mmBERT 한국어 NSMC 정확도** | **0.8545** — 기준선 0.85 통과, 단 여유 없음 |
| `klue/klue` ner 로드 · `.feature.names` 접근 | ✅ |
| SFT 1스텝 (`peft_config=` 경로) | ✅ |
| `kullm-v2` 로드 | ✅ — 단 `input` 컬럼 발견 → 병합 처리함 |
| `kowikitext` parquet 직접 지정 | ❌ → `wikimedia/wikipedia` 로 교체 |

### ✅ 인코더 확정 — mmBERT 유지 (2026-08-25)

`monologg/koelectra-base-v3-discriminator`(Apache-2.0, 한국어 네이티브)와
동일 조건 비교한 결과입니다.

| 모델 | NER F1 | NSMC acc |
|---|---:|---:|
| **mmBERT-base** | **0.8850** | 0.8525 |
| KoELECTRA-v3 | 0.6734 | 0.8710 |

**토큰화 우려는 사실이 아니었습니다.** "음절 단위라 NER 이 더 타격받을 것"으로
봤는데 정반대로 NER 에서 크게 앞섰습니다. 한국어는 교착어라 개체명 경계가
어절 내부에 걸리는데, 잘게 쪼개는 것이 오히려 유리했던 것으로 보입니다.

---

## ✅ 9종 전부 실행 검증 완료 (2026-08-25)

`nbconvert` 로 실제 실행해 셀 사슬이 끝까지 도는 것을 확인했습니다. **실패 0건.**

| 일차 | 노트북 | 소요 |
|---|---|---:|
| 1 | Classification · NER · MiniGPT | 140 · 149 · 112초 |
| 2 | 퓨샷 · SFT · DPO · GRPO | 44 · **475** · 285 · **491**초 |
| 3 | Amazon 요약 · BM25 RAG | 187 · 82초 |

**합계 약 32분** (무인 실행 기준). 실제 강의에서는 설명·질의응답이 붙어 **3~5배**를 잡아야 합니다.

> ⚠️ **2일차가 무겁습니다.** SFT(8분) + GRPO(8분) + 퓨샷(서빙, GPU 전환 필요).
> 시간을 줄이려면 `num_train_epochs` 나 데이터 개수를 낮추면 되지만,
> 너무 줄이면 결과가 안 나옵니다 — MiniGPT 에서 실제로 겪었습니다.

---

## 아직 안 된 것

| 항목 | 상태 |
|---|---|
| **출력 품질 확인** | 완주는 확인. 각 노트북의 **결과가 제대로 나왔는지**는 별도 확인 필요 |
| **`Taekyoon/test_amazon`** | 라이선스 미표기 — 소유자 판단으로 **현행 유지 결정** |
| **슬라이드(PPT) 수정** | 미착수. 불일치 78건 (C축) |
| **⑧ Pre-training 데이터 처리 신규 개발** | 계획서 협의 결과 대기 |
| **RAG 처리 방향** | 발주처 회신 대기 (결-1) |
