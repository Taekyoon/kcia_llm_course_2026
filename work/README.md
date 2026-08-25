# work/ — 수정 작업 영역

`notebook/` 원본은 읽기 전용입니다(CLAUDE.md §1). 모든 수정은 여기서 합니다.

## work/notebook/ 는 손으로 고치지 마세요

이 폴더는 **`tools/migrate_notebooks.py` 가 원본에서 매번 새로 생성**합니다.
직접 편집하면 다음 실행에서 덮어써집니다.

```bash
uv run python tools/migrate_notebooks.py            # 생성
uv run python tools/migrate_notebooks.py --dry-run  # 미리보기
```

수정 내용을 바꾸려면 `tools/migrate_notebooks.py` 의 **규칙표**를 고치세요.
규칙마다 변경 사유(`why`)가 붙어 있고 리포트에 그대로 출력됩니다.

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
| 1일차 MiniGPT | 0 | **변경 없음** — keras-hub API 유효. 문제는 슬라이드 쪽 |
| 2일차 SFT | 6 | kullm-v2 교체 · **`input` 컬럼 병합** · `max_seq_length`→`max_length` |
| 2일차 DPO | 3 | pip 정리 · processing_class |
| 2일차 GRPO | 4 | Qwen3 통일 · processing_class |
| 2일차 퓨샷 | 6 | Qwen3-4B(18곳) · guided_choice/json 제거 · vllm serve |
| 3일차 Amazon 요약 | 10 | Qwen3-4B · `.beta.parse`→`create` · guided_json 6곳 → response_format |
| 3일차 BM25 RAG | 5 | OpenAILike HTTP · **wikimedia/wikipedia** · **update_prompts()** |

상세 근거: [review/02_문제점.md](../review/02_문제점.md) · [verify/결과.md](../verify/결과.md)

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

## 아직 안 된 것

| 항목 | 상태 |
|---|---|
| **`Taekyoon/test_amazon` 교체** | 라이선스 미표기. 노트북에 TODO 주석만. **자체 합성 데이터 필요** |
| **vLLM 계열 실행 검증** | `verify/02_vllm_venv.sh` 로 서버 띄운 뒤 확인 |
| **슬라이드(PPT) 수정** | 미착수. 불일치 78건 (C축) |
| **⑧ Pre-training 데이터 처리 신규 개발** | 계획서 협의 결과 대기 |
| **RAG 처리 방향** | 발주처 회신 대기 (결-1) |
