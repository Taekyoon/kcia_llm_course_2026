# assets — 강사가 미리 만들어 배포하는 자산

`data/` 와 혼동하지 마세요.

| | 성격 | git |
|---|---|---|
| `data/` | 실습을 실행하면 생기는 것. 언제든 재생성 가능 | **무시** |
| `assets/` | 강사가 수업 **전에** 만들어 두는 것. 수업 중에 못 만듦 | **커밋** |

---

## `amazon_ko_sft.jsonl.gz`

3일차 SFT 실습이 읽는 학습 데이터입니다.

| | |
|---|---|
| 만드는 법 | `python tools/build_amazon_pregen.py --n 300` (vLLM 필요) |
| 원본 | `Taekyoon/test_amazon` 의 영문 상품 설명 |
| 생성 | `Qwen/Qwen3-4B-Instruct-2507` 로 7단계 파이프라인 |
| 형식 | `{"instruction": ..., "output": ...}` JSONL, gzip |

### 왜 미리 만드나

상품 1개에 LLM 호출이 16번쯤 들어갑니다. 300건이면 5,000번 가까이 되어
강의 시간에 할 수 없습니다. 수강생은 2일차 Amazon 실습에서 **10건**만 직접 만들고
(`data/amazon_ko_sft.mine.jsonl`), 3일차 SFT 는 **이 파일과 합쳐서** 학습합니다.

직접 만든 것이 그 안에 들어가 있다는 점이 중요합니다 —
남의 데이터가 아니라 **내가 만든 데이터로 학습**하는 흐름입니다.

### 다시 만들어야 할 때

노트북의 **프롬프트를 고쳤다면** 다시 만드세요. 그러지 않으면 수강생이 만든 10건과
배포본 300건의 성격이 달라집니다.

스크립트는 프롬프트·스키마를 `work/notebook/2일차/HPC_Amazon요약실습.ipynb` 에서
직접 가져오므로, 노트북만 고치면 됩니다. 복사본이 따로 있지 않습니다.

```bash
python tools/build_amazon_pregen.py --n 300            # 처음부터
python tools/build_amazon_pregen.py --n 300 --resume   # 중단된 것 이어받기
```

### 원본 데이터에 대해

`Taekyoon/test_amazon` 은 영문 상품 카탈로그입니다. `instruction` 안에 원문 일부
(앞 1,200자)가 들어갑니다. 원본에 고유명사가 글자 단위로 쪼개진 결함이 있어
(`- Brand: F, a, t,  , S, h, a, r, k`) 변환 시 복원합니다.
