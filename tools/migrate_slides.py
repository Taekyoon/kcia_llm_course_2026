"""슬라이드 재조립기 — 원본 3덱에서 새 3덱을 **매번 다시** 만든다.

    uv run --project tools python tools/migrate_slides.py

`migrate_notebooks.py` 와 같은 규율:
- 원본(`ppt/`)은 읽기만 한다. 산출물은 `work/ppt/` 에 통째로 다시 쓴다.
- 치환 규칙에는 근거(why)를 붙인다. 앵커를 못 찾으면 조용히 넘어가지 않고 멈춘다.
- 배치는 `slide_layout.PLACEMENT` 가 단독으로 정한다.

신규 슬라이드는 **도너 복제** 방식이다: 그림 없는 텍스트 전용 장(1일차 p19)을
복제해 헤더·본문 텍스트만 갈아 끼운다. placeholder 방식은 이 덱들과 맞지 않는다 —
전 장이 Blank 레이아웃 + 슬라이드 내 장식 도형 구조라, 도너 쪽이 장식을 보존한다.
"""

from __future__ import annotations

import copy
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.slide import Slide
from pptx.util import Pt

import pptxcommon as pc
from slide_layout import DECKS, DECK_PAGES, PLACEMENT, WORK_PPT, NewSlide, SlideRef, out_path

# 신규 슬라이드의 도너 — 그림 없는 텍스트 전용 장 (2026-08-27 도형 실사로 선정)
DONOR = ("1일차", 19)
BOILER = "쉬운 HPC 활용교육"


# ---------------------------------------------------------------------------
# 치환 규칙
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlideRule:
    """원본 (덱, 페이지) 한 장 안의 문자열 치환. 임포트 시점에 적용되므로
    재배치로 번호가 밀릴 일이 없다. 그 장에서 정확히 count 회 치환돼야 한다."""
    deck: str
    page: int
    old: str
    new: str
    why: str
    count: int = 1


@dataclass(frozen=True)
class GlobalSlideRule:
    """재사용되는 원본 전 장 대상. 총 치환 횟수가 expect 와 다르면 중단."""
    old: str
    new: str
    why: str
    expect: int


# 코드 슬라이드의 현행 노트북 동기화 (E-2, 2026-08-27 XML 수확 기준).
# 문단(=줄) 단위로만 치환된다 — 여러 줄 교체는 줄별 규칙으로 나눈다.
RULES: list[SlideRule] = [
    SlideRule('2일차', 8, '!pip install openai vllm datasets',
              "%pip install -q -U 'openai<3' datasets",
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('2일차', 8, '!nohup python -m vllm.entrypoints.openai.api_server --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct &',
              '# 서버는 별도 venv 터미널에서: nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 --gpu-memory-utilization 0.80 --max-model-len 16384 &',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다 — api_server deprecated + torch 하드핀 venv 분리'),
    SlideRule('2일차', 9, '실행된 모델은 Colab 실습 환경 안에 모델 서비스로 실행중에 있습니다.',
              '실행된 모델은 실습 환경(VESSL) 안에 모델 서비스로 실행 중입니다.',
              '실습 환경이 Colab → VESSL 로 바뀌었다 (E-2)'),
    SlideRule('2일차', 11, 'dataset = load_dataset("e9t/nsmc", trust_remote_code=True)',
              'dataset = load_dataset("e9t/nsmc", revision="refs/convert/parquet")',
              'datasets 5.x 스크립트 로딩 제거 — parquet 리비전 (CLAUDE.md §9 실측)'),
    SlideRule('2일차', 31, 'raw_datasets = load_dataset("beomi/KoAlpaca-v1.1a")',
              'raw_datasets = load_dataset("json", data_files="data/amazon_ko_sft.mine.jsonl")',
              'KoAlpaca 는 CC BY-NC. 현행 SFT 는 2일차에 직접 만든 데이터로 학습한다'),
    SlideRule('2일차', 31, '학습 데이터는 KoAlpaca 데이터셋을 활용 합니다.',
              '학습 데이터는 2일차에 직접 만든 상품 요약 데이터셋입니다.',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('2일차', 75, '"Qwen/Qwen2.5-0.5B-Instruct"',
              '"Qwen/Qwen3-0.6B-Base"',
              'GRPO 만 구세대 모델이었다 — Qwen3 계열 통일 (CLAUDE.md §10)'),
    SlideRule('3일차', 23, '! nohup python -m vllm.entrypoints.openai.api_server --model LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct &',
              '# 서버는 별도 venv 터미널에서: nohup vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000 &',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('3일차', 57, '여기서는 LG 엑사원 모델을 활용합니다.',
              '여기서는 vLLM 서버의 Qwen3-4B 모델을 활용합니다.',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('3일차', 57, 'from llama_index.llms.vllm import Vllm',
              'from llama_index.llms.openai_like import OpenAILike',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다 — 현행 RAG 는 vLLM 서버에 HTTP 로 붙는다'),
    SlideRule('3일차', 57, 'Settings.llm = Vllm(',
              'Settings.llm = OpenAILike(',
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('3일차', 57, "dtype='float16',",
              "model='Qwen/Qwen3-4B-Instruct-2507', api_base='http://localhost:8000/v1',",
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('3일차', 57, "model='LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct'",
              "api_key='EMPTY', is_chat_model=True,",
              '현행 노트북과 동기화 (E-2). 슬라이드 코드가 그대로는 안 돈다'),
    SlideRule('1일차', 38, 'dataset = load_dataset("kor_ner")',
              'dataset = load_dataset("klue/klue", "ner")',
              'kor_ner 스크립트형 제거 — klue/klue ner (컬럼명 동일, CLAUDE.md §9 실측)'),
]

# 전역 규칙. expect 는 재사용 257장의 XML 결합 텍스트 실측값 (2026-08-27).
# 섹션 규칙은 이름으로 구분되어 서로의 출력을 다시 잡지 않는다 (충돌 분석 완료).
GLOBAL_RULES: list[GlobalSlideRule] = [
    GlobalSlideRule('3.\t트렌스포머와 GPT', '2.\t트랜스포머와 ChatGPT',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 14),
    GlobalSlideRule('2.\t자연어처리 머신러닝 소개', '3.\t자연어처리 머신러닝 소개',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('2. 자연어처리 머신러닝 소개', '3. 자연어처리 머신러닝 소개',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 26),
    GlobalSlideRule('4. 미니 GPT 만들기', '3. 미니 GPT 만들기',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 28),
    GlobalSlideRule('1. 도메인 최적화 프리트레인에 대해서', '4. 도메인 최적화 프리트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측) + 표기 정정', 1),
    GlobalSlideRule('1. 도메인 최적화 프리트레이닝에 대해서', '4. 도메인 최적화 프리트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 13),
    GlobalSlideRule('2. VLLM을 활용한 데이터처리 실습', '5. vLLM을 활용한 데이터 생성 실습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측). Amazon 은 정제가 아니라 생성 축이라 명칭도 맞춘다', 35),
    GlobalSlideRule('3. Llama Index를 활용한 RAG 실습', '1. Llama Index를 활용한 RAG 실습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 18),
    GlobalSlideRule('4. 지속적인 LLM 챗봇 개발을 위해서', '2. 태스크 정의와 평가',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측). 커리큘럼 세부항목명과 일치', 10),
    GlobalSlideRule('1. 퓨샷 러닝에 대해서', '3. 퓨샷 러닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('1. 퓨샷러닝에 대해서', '3. 퓨샷러닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 14),
    GlobalSlideRule('2. 포스트 트레이닝에 대해서', '4. 포스트 트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 11),
    GlobalSlideRule('3.\t 인스트럭션 모델 학습', '5.\t 인스트럭션 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('3. 인스트럭션 모델학습', '5. 인스트럭션 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 22),
    GlobalSlideRule('4.\t 선호기반 모델 학습', '6.\t 선호기반 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('4. 선호기반 모델학습', '6. 선호기반 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 17),
    GlobalSlideRule('5.\t 리즈닝 모델 학습', '7.\t 리즈닝 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('5. 리즈닝 모델학습', '7. 리즈닝 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 18),
    GlobalSlideRule('Vllm', 'vLLM',
                    '용어 표준 (CLAUDE.md §7). p57 줄 교체가 먼저 2곳을 없애 13', 13),
    GlobalSlideRule('evaluation_strategy', 'eval_strategy',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측) — transformers 5 에서 제거, 슬라이드 코드가 깨진다 (C-3)', 2),
    GlobalSlideRule('trainer.tokenizer', 'trainer.processing_class',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측)', 3),
    GlobalSlideRule('max_seq_length', 'max_length',
                    '라이브러리 API 변경 (CLAUDE.md §9 실측) — SFTConfig 인자명 변경', 1),
    GlobalSlideRule('RLPH', 'RLHF',
                    '용어 표준 (CLAUDE.md §7)', 1),
    GlobalSlideRule('높히', '높이',
                    '용어 표준 (CLAUDE.md §7)', 1),
    GlobalSlideRule('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct',
                    'Qwen/Qwen3-4B-Instruct-2507',
                    'EXAONE NC 라이선스 → Qwen3 (CLAUDE.md §10). 서빙 줄 교체(RULES)로 3곳이 먼저 사라져 7', 7),
    GlobalSlideRule('"beomi/gemma-ko-2b"',
                    '"jhu-clsp/mmBERT-base"',
                    '분류·NER 모델 교체 — 인코더 전환 (CLAUDE.md §10)', 3),
    GlobalSlideRule('client.beta.chat.completions.parse(',
                    'client.chat.completions.create(',
                    'openai SDK 2.x — .beta 네임스페이스 이동 (노트북과 동일 처리)', 6),
    GlobalSlideRule('extra_body={"guided_json": feature_type_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "feature_type_list", "schema": feature_type_schema}},',
                    'vLLM 0.12 에서 guided_json 정식 제거 — 조용히 무시되는 함정 (CLAUDE.md §9)', 1),
    GlobalSlideRule('extra_body={"guided_json": subsectoin_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "subsection_list", "schema": subsectoin_schema}},',
                    'guided_json 제거 (위와 동일)', 1),
    GlobalSlideRule('extra_body={"guided_json": extracted_feature_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "extracted_feature_list", "schema": extracted_feature_schema}},',
                    'guided_json 제거 (위와 동일)', 1),
    GlobalSlideRule('extra_body={"guided_json": consumer_category_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "consumer_category_list", "schema": consumer_category_schema}},',
                    'guided_json 제거 (위와 동일)', 1),
    GlobalSlideRule('extra_body={"guided_json": summary_schema},',
                    'response_format={"type": "json_schema", "json_schema": {"name": "summary", "schema": summary_schema}},',
                    'guided_json 제거 (위와 동일)', 2),
    GlobalSlideRule('extra_body={"guided_choice": ["긍정", "부정"]},',
                    'extra_body={"structured_outputs": {"choice": ["긍정", "부정"]}},',
                    'guided_choice 제거 — structured_outputs (CLAUDE.md §9 실측)', 2),
    GlobalSlideRule('공유드린 Github 실습자료를 Colab에서 불러옵니다.',
                    '실습 노트북(work/notebook)을 VESSL 워크스페이스에서 엽니다.',
                    '실습 환경 Colab → VESSL (E-2)', 3),
    GlobalSlideRule('%pip install -q bitsandbytes trl peft math_verify',
                    '%pip install -q math_verify   # 나머지 학습 스택은 사전 설치됨',
                    'setup_vessl.sh 가 스택을 설치한다 — 긴 줄 먼저 (접두사 충돌)', 1),
    GlobalSlideRule('%pip install -q bitsandbytes trl peft',
                    '# torch 는 constraints 로 고정되어 재설치되지 않습니다',
                    'setup_vessl.sh 가 스택을 설치한다', 2),
    GlobalSlideRule('%pip install -q transformers[torch] datasets',
                    '# 학습 스택은 setup_vessl.sh 가 설치해 둡니다 (transformers·datasets·trl·peft)',
                    'setup_vessl.sh 가 스택을 설치한다', 3),
]

# ---------------------------------------------------------------------------
# 신규 슬라이드 레지스트리 — key → 장 목록
#   각 장: dict(header=우상단 헤더, title=소제목, bullets=[(level, text), ...])
#   Phase F 에서 원고로 대체한다. 지금은 파이프라인 검증용 스텁.
# ---------------------------------------------------------------------------
def _stub(key: str, memo: str, n: int) -> list[dict]:
    return [{
        "header": "(신규) " + memo.split("(")[0].strip(),
        "title": f"{memo.split('(')[0].strip()} — 초안 {i + 1}/{n}",
        "bullets": [(0, "원고는 Phase F 에서 작성됩니다."),
                    (1, f"배치 키: {key}")],
    } for i in range(n)]


NEW_SLIDES: dict[str, list[dict]] = {
    # 목차 3장은 각 덱의 원본 목차(p2)를 도너로 쓴다 — 목차 서식을 그대로 물려받는다
    # (본문형 도너로 찍었더니 정리가 안 돼 보인다는 사용자 지적, 2026-08-27).
    "toc1": [{"donor": ("1일차", 2), "toc": [
        '1.\tAI 소프트웨어 개발 개론',
        '2.\t트랜스포머와 ChatGPT',
        '3.\t자연어처리 머신러닝 소개와 실습',
    ]}],
    "toc2": [{"donor": ("2일차", 2), "toc": [
        '1.\tPre-training 등장 배경과 목적',
        '2.\tPre-training 데이터 처리',
        '3.\t미니 GPT 만들기',
        '4.\t도메인 최적화 프리트레이닝(CPT)',
        '5.\tvLLM을 활용한 데이터 생성 실습',
        '6.\t프롬프트 자동 최적화',
    ]}],
    "toc3": [{"donor": ("3일차", 2), "toc": [
        '1.\tLlama Index를 활용한 RAG 실습',
        '2.\t태스크 정의와 평가',
        '3.\t퓨샷 러닝',
        '4.\t포스트 트레이닝',
        '5.\t인스트럭션 모델 학습',
        '6.\t선호기반 모델 학습',
        '7.\t리즈닝 모델 학습',
        '8.\t마무리 — 방법 선택 가이드와 RAG 개선 확인',
    ]}],
    "ml2dl": _stub("ml2dl", "④ ML→DL 발전사", 4),
    "pretrain_intro": _stub("pretrain_intro", "⑦ Pre-training 등장 배경과 목적", 5),
    "datacleaning": _stub("datacleaning", "⑧ Pre-training 데이터 처리", 10),
    "promptopt": [
        {"header": '6.\t프롬프트 자동 최적화',
         "title": '학습 전에 최선을 다하는 방법',
         "bullets": [
            (0, '파인튜닝은 비쌉니다 — GPU 만이 아니라 데이터 · 평가 체계 · 유지보수'),
            (0, '실무 순서: 프롬프트로 해결되는가? → 예: 끝 / 아니오: 그때 학습'),
            (0, '이 실습은 왼쪽 가지를 끝까지 밀어붙입니다 — 그것도 자동으로'),
            (1, '프롬프트를 파라미터로 보고, 점수를 목적함수로 삼아 탐색합니다'),
        ]},
        {"header": '6.\t프롬프트 자동 최적화',
         "title": '3막 구조',
         "bullets": [
            (0, '1막 — 번역: 영문 상품 설명을 한국어로. 없던 한국어 데이터를 만든다'),
            (1, '고유명사는 원문 유지 · 쪼개진 브랜드명(F, a, t, S, h, a, r, k) 복원'),
            (0, '2막 — LLM-as-judge: 번역 품질을 채점하고 나쁜 것을 거른다'),
            (1, '같은 번역을 세 번 채점해 점수가 흔들리는 것도 직접 확인'),
            (0, '3막 — DSPy GEPA: 프롬프트를 자동으로 고쳐 쓴다'),
            (1, '산출물(amazon_ko.jsonl)은 3일차 SFT 의 학습 데이터가 된다'),
        ]},
        {"header": '6.\t프롬프트 자동 최적화',
         "title": '핵심 규칙 — 목적함수에 노이즈가 없어야 한다',
         "bullets": [
            (0, 'judge 점수를 그대로 목적함수로 쓰면? — 옵티마이저는 노이즈를 맞춘다'),
            (1, '점수는 오르는데 품질은 그대로인 결과. 실무에서 실제로 저지르는 실수'),
            (0, '그래서 최적화 대상을 바꿉니다: 번역 → 구조화 추출'),
            (1, '형식 준수 · 항목 수 · 원문 근거 · 한국어 — 전부 프로그램 판정 (노이즈 0)'),
            (0, '출력 형식은 고정하고, 지시문만 최적화합니다'),
            (1, '요구사항을 안 알려주고 맞히게 하면 개선이 나지 않습니다'),
        ]},
        {"header": '6.\t프롬프트 자동 최적화',
         "title": '결과 — 학습 없이 얻은 개선',
         "bullets": [
            (0, 'GEPA 120회 호출, 약 8분 (L40S 실측) — val 0.80 → 1.00'),
            (0, 'reflection LM 이 feedback 을 읽고 지시문을 스스로 고쳐 썼다'),
            (1, 'category 는 대분류로 · features 는 명사 기반으로 간결하게'),
            (0, '이 점수가 내일 학습과 비교할 기준선이 됩니다'),
            (1, '학습이 사주는 것은 프롬프트 위에 얹히는 만큼 — 그 차이가 판단 기준'),
        ]},
    ],
    "lora": [
        {"header": '5.\t인스트럭션 모델 학습 – LoRA 이론',
         "title": 'LoRA — 전부 학습시키지 않는다',
         "bullets": [
            (0, '원래 가중치 W 는 얼려 두고, 옆에 작은 행렬 두 개(A·B)만 붙여 학습한다'),
            (1, '출력 = W·x + B·A·x   (B·A 가 학습되는 부분 — 저랭크 보정)'),
            (0, '왜 되는가: 파인튜닝이 실제로 바꾸는 양은 저차원으로 근사된다'),
            (0, '저장은 어댑터(A·B)만 — 수백 MB 가 아니라 수십 MB'),
            (1, 'SFT 실습에서 trainable 2.99% 였던 것이 바로 이 구조입니다'),
        ]},
        {"header": '5.\t인스트럭션 모델 학습 – LoRA 이론',
         "title": 'r 과 alpha — 두 개의 손잡이',
         "bullets": [
            (0, 'r: 붙이는 행렬의 크기. 클수록 표현력이 늘고 무거워진다'),
            (0, 'alpha: 그 결과를 얼마나 세게 반영할지 (스케일 = alpha / r)'),
            (0, '이 과정의 실습값:'),
            (1, 'SFT: r=64, alpha=16 — 데이터가 877건이라 표현력을 넉넉히'),
            (1, 'GRPO: r=8, alpha=32 — 생성 부담이 커서 가볍게'),
            (0, '정답이 있는 값이 아닙니다 — 작업과 자원에 따라 고릅니다'),
        ]},
        {"header": '5.\t인스트럭션 모델 학습 – LoRA 이론',
         "title": '어디에 붙이고, 어떻게 배포하나',
         "bullets": [
            (0, 'target_modules: 어디에 붙일지 — 관례는 어텐션 4곳 (q·k·v·o proj)'),
            (1, '피드포워드까지 붙이면 표현력↑ 비용↑ — 실습은 어텐션만'),
            (0, '배포 두 가지 길:'),
            (1, '어댑터만 배포 — 베이스는 공용, 작업마다 어댑터 교체'),
            (1, 'merge_and_unload() — 베이스에 합쳐 단일 모델로 (추론 오버헤드 0)'),
            (0, 'DPO 실습에서 참조 모델이 필요 없던 이유 — 어댑터를 끄면 그게 원래 모델'),
        ]},
        {"header": '5.\t인스트럭션 모델 학습 – LoRA 이론',
         "title": 'QLoRA 와 알아둘 성질',
         "bullets": [
            (0, 'QLoRA: 베이스를 4bit 로 양자화해 올리고 그 위에 LoRA — 메모리 대폭 절감'),
            (1, '0.6B 실습엔 불필요하지만, 7B+ 를 GPU 한 장으로 다룰 때의 표준'),
            (0, '성질: LoRA 는 full FT 보다 도메인 밖 성능을 잘 보존한다'),
            (1, '망각 정도가 rank 로 조절된다 (Biderman 2024)'),
            (1, '단 무조건 덜 잊는다는 과장 — 비교 조건(LR)에 따라 달라진다'),
        ]},
    ],
    "method_guide": [
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '"우리 문제에는 뭘 써야 하죠?"',
         "bullets": [
            (0, '3일 동안 방법을 배웠습니다 — 프롬프트, SFT, DPO, ORPO, GRPO, RAG, CPT'),
            (0, '이 장은 그 사이에서 고르는 법입니다. 핵심 메시지는 셋:'),
            (1, '① 데이터가 방법을 정한다 — 방법을 먼저 고르지 않는다'),
            (1, '② 위에서부터 시도한다 — 프롬프트로 되면 학습하지 않는다'),
            (1, '③ DPO 는 SFT 를 전제한다 — ORPO 는 그 전제를 없앤 것'),
            (0, '"GRPO 가 최신이니 GRPO" 는 실수입니다 — 검증 함수 없이는 시작도 못 합니다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '의사결정 표 — 가진 것이 방법을 정한다',
         "bullets": [
            (0, '아무것도 없음  →  프롬프트 최적화  (학습 안 함 · 2일차)'),
            (0, '입력–정답 쌍  →  SFT  (3일차)'),
            (0, '선호 쌍 + SFT 완료 모델  →  DPO  (2단계 · 3일차)'),
            (0, '선호 쌍만, SFT 아직  →  ORPO  (1단계 · DPO 실습 비교 절)'),
            (0, '정답을 검증하는 함수  →  GRPO  (가장 비쌈 · 3일차)'),
            (0, '최신 사실이 필요  →  RAG  (학습 안 함 · 3일차)'),
            (0, '도메인 언어 자체가 다름  →  CPT  (가장 큼 · 2일차)'),
            (1, '위에서 아래로 갈수록 비싸집니다 — 위에서부터 시도합니다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '비용 감각 — 이 과정에서 직접 돌린 시간 (L40S 실측)',
         "bullets": [
            (0, '프롬프트 최적화 (GEPA 120회)   약 8분  /  실무: 수십 분~수 시간'),
            (0, 'SFT (0.6B · 877건 · LoRA)      약 8분  /  실무: 수 시간~수 일'),
            (0, 'DPO (0.6B · LoRA)              약 5분  /  SFT 를 마친 뒤에 다시 그만큼'),
            (0, 'GRPO (0.6B)                    약 8분  /  가장 비쌈 — 생성이 학습 안에 있다'),
            (0, 'CPT (2M 미니GPT · replay 3회)  약 2분  /  실무: 수 주 (llama-2-ko 는 40B+ 토큰)'),
            (1, 'GRPO 가 비싼 이유: 매 스텝 답을 여러 개 생성하고 각각 채점합니다'),
            (1, '실습 시간은 2026-08 실측, 실무 규모는 참고 수치입니다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '전제와 함정 ① — 프롬프트 최적화 · SFT',
         "bullets": [
            (0, '프롬프트 최적화'),
            (1, '전제: 채점할 수 있어야 한다 — 점수를 못 매기면 최적화도 없다'),
            (1, '함정: 노이즈 큰 점수를 목적함수로 삼으면 노이즈를 최적화한다'),
            (1, '→ 가능하면 프로그램으로 검증되는 metric (JSON 유효성 · 필수 항목 · 형식)'),
            (0, 'SFT'),
            (1, '전제: 입력–정답 쌍. 정답이 하나로 정해지는 작업에 맞는다'),
            (1, '함정: 정답이 여럿인 작업에 SFT 만 쓰면 한 표현만 정답이라고 가르치게 된다'),
            (1, '함정: 긴 입력이 잘려 정답이 날아가면 loss 는 낮은데 아무것도 안 배운다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '전제와 함정 ② — DPO · ORPO · GRPO',
         "bullets": [
            (0, 'DPO — 전제: SFT 완료 모델 + 선호 쌍'),
            (1, '베이스에 바로 걸면 잘 안 된다. 참조 모델로 메모리도 두 배 (LoRA 면 절약)'),
            (0, 'ORPO — 전제: 선호 쌍만'),
            (1, 'SFT 없이 한 단계로. 선호 데이터부터 모인 상황이라면 파이프라인이 짧아진다'),
            (0, 'GRPO — 전제: 정답을 코드로 검증할 수 있어야 한다'),
            (1, '수학(답 검증) · 코드(테스트) · 형식(스키마) — 이게 없으면 못 쓴다'),
            (1, '리워드 모델을 새로 학습시키는 길은 학습을 하나 더 얹는 것 — 비용이 다시 뛴다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '전제와 함정 ③ — 학습의 대안과 극단',
         "bullets": [
            (0, 'RAG — 문제가 "모델이 모른다" 일 때. 사실이 자주 바뀔 때. 학습이 아니다'),
            (0, 'CPT — 도메인 언어 자체가 다를 때 (법률 · 의료 · 특정 언어)'),
            (0, '자주 틀리는 판단: "우리 회사 문서를 학습시키자"'),
            (1, '대개는 RAG 가 맞습니다'),
            (1, '파인튜닝은 지식을 넣는 도구가 아니라 행동을 바꾸는 도구입니다'),
            (1, '오늘 아침 RAG 로 시작해 방금 그 시스템에 학습 모델을 꽂아 본 이유입니다'),
        ]},
        {"header": '8.\t마무리 — 방법 선택 가이드',
         "title": '실무 순서 — 이 장의 결론',
         "bullets": [
            (0, '① 프롬프트로 해결되는가?  예 → 끝. 학습하지 않는다'),
            (0, '② 모델이 모르는 게 문제인가?  예 → RAG'),
            (0, '③ 입력–정답 쌍이 있는가?  예 → SFT'),
            (0, '④ 선호 쌍이 있는가?  예 → SFT 했으면 DPO, 안 했으면 ORPO'),
            (0, '⑤ 검증 함수를 짤 수 있는가?  예 → GRPO'),
            (0, '⑥ 전부 아니오 → 데이터부터 만든다 (2일차로 돌아간다)'),
            (1, '마지막 가지가 중요합니다 — 방법이 없는 게 아니라 데이터가 없는 것'),
            (1, '이 과정이 2일차 하루를 데이터에 쓴 이유입니다'),
        ]},
    ],
    "ragcheck": [
        {"header": '8.\t마무리 — RAG 개선 확인',
         "title": '아침의 시스템에 오후의 모델을 꽂는다',
         "bullets": [
            (0, '같은 BM25 인덱스 · 같은 질문 · 같은 프롬프트 — 바꾸는 것은 모델 하나'),
            (0, '학습 전 (Qwen3-0.6B-Base): 지시 개념이 없어 이어쓰기만 한다'),
            (1, '검색이 완벽한 문서를 찾아줘도 시스템은 동작하지 않는다'),
            (0, '학습 후 (오늘 만든 SFT): 질문을 받았다는 걸 알고 답하려 한다'),
            (1, '학습 데이터는 상품 요약 877건 — 그런데도 답하는 행동이 옮겨 왔다'),
            (0, '개선의 실체는 지식이 아니라 행동입니다'),
        ]},
        {"header": '8.\t마무리 — RAG 개선 확인',
         "title": '3일의 결론',
         "bullets": [
            (0, '지식은 검색으로 넣고, 행동은 학습으로 바꾼다'),
            (0, '프롬프트로 되는지 먼저 — 안 되면 그때 학습'),
            (0, '무엇을 하든, 바꾸기 전에 잴 수단부터 갖춘다'),
            (1, '오늘 평가 실습을 학습보다 먼저 한 이유'),
            (0, '그 판단을 스스로 내릴 수 있게 된 것 — 이 3일의 목표였습니다'),
        ]},
    ],
}


# ---------------------------------------------------------------------------
# 텍스트 채우기 (도너 복제 후)
# ---------------------------------------------------------------------------
def _set_text(tf, lines: list[tuple[int, str]], base_size: Pt | None) -> None:
    """텍스트 프레임을 비우고 (level, text) 목록으로 다시 채운다.
    서식은 도너 첫 run 의 크기만 물려받는다 — 텍스트 초안 수준(사용자 확정)."""
    first_run = None
    for p_el in tf.paragraphs:
        for r in p_el.runs:
            first_run = r
            break
        if first_run:
            break
    font_name = first_run.font.name if first_run else None
    tf.clear()
    for i, (level, text) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = min(level, 4)
        for r in p.runs:
            if font_name:
                r.font.name = font_name
            if base_size:
                r.font.size = base_size


def fill_new_slide(part, spec: dict) -> None:
    slide = Slide(part._element, part)

    if "toc" in spec:
        # 목차 도너: "목 차" 제목은 그대로 두고, 문단이 가장 많은 본문 상자만 교체
        boxes = [sh for sh in slide.shapes
                 if sh.has_text_frame and BOILER not in sh.text_frame.text
                 and sh.text_frame.text.strip()]
        body = max(boxes, key=lambda b: len(b.text_frame.paragraphs))
        _set_text(body.text_frame, [(0, line) for line in spec["toc"]], None)
        return

    boxes = [sh for sh in slide.shapes
             if sh.has_text_frame and BOILER not in sh.text_frame.text]
    boxes = [b for b in boxes if b.text_frame.text.strip()]
    if len(boxes) < 3:
        raise SystemExit(f"[중단] 도너 장의 텍스트 상자가 예상과 다릅니다: {len(boxes)}개")
    boxes.sort(key=lambda b: (b.top or 0))
    header, body, tagline = boxes[0], max(boxes[1:], key=lambda b: (b.width or 0) * (b.height or 0)), None
    tagline = next((b for b in boxes[1:] if b is not body), None)

    _set_text(header.text_frame, [(0, spec["header"])], None)
    lines = [(0, spec["title"])] + [(lv + 1, tx) for lv, tx in spec["bullets"]]
    _set_text(body.text_frame, lines, None)
    if tagline is not None:
        _set_text(tagline.text_frame, [(0, "")], None)


# ---------------------------------------------------------------------------
# 빌드
# ---------------------------------------------------------------------------
def expected_counts() -> dict[str, int]:
    out = {}
    for day, items in PLACEMENT.items():
        n = 0
        for it in items:
            n += len(NEW_SLIDES[it.key]) if isinstance(it, NewSlide) else 1
        out[day] = n
    return out


def build(dst_dir: Path | None = None, verbose: bool = True) -> dict[str, Path]:
    dst_dir = dst_dir or WORK_PPT
    pc.guard_masters_identical(DECKS)

    srcs = {d: Presentation(str(p)) for d, p in DECKS.items()}
    zips = {d: zipfile.ZipFile(p) for d, p in DECKS.items()}
    for d, prs in srcs.items():
        if len(prs.slides) != DECK_PAGES[d]:
            raise SystemExit(f"[중단] {d} 원본 장수 {len(prs.slides)} ≠ 기대 {DECK_PAGES[d]}")

    rules_by_page: dict[tuple[str, int], list[SlideRule]] = {}
    for r in RULES:
        rules_by_page.setdefault((r.deck, r.page), []).append(r)
    global_hits = {id(g): 0 for g in GLOBAL_RULES}

    # PowerPoint 가 산출물을 열어 두면 저장이 막힌다 — 시작 전에 전부 확인한다
    for day in PLACEMENT:
        target = (dst_dir) / out_path(day).name
        if target.exists():
            try:
                target.rename(target)   # Windows: 열려 있으면 여기서 실패한다
            except PermissionError:
                raise SystemExit(
                    f"[중단] {target.name} 이 PowerPoint 에 열려 있습니다. "
                    f"파일을 닫고 다시 실행하세요.")

    outputs = {}
    for day, items in PLACEMENT.items():
        shell = Presentation(str(DECKS[day]))
        pc.drop_all_slides(shell)
        media = pc.MediaCache(shell.part.package)
        sid = 256
        n_ref = n_new = 0

        for it in items:
            if isinstance(it, SlideRef):
                def mutate(el, _ref=it):
                    for rule in rules_by_page.get((_ref.deck, _ref.page), []):
                        n = pc.replace_in_tree(el, rule.old, rule.new)
                        if n != rule.count:
                            raise SystemExit(
                                f"[중단] 규칙 불일치 — {_ref.deck} p{_ref.page}: "
                                f"{rule.old!r} 를 {n}회 치환 (기대 {rule.count})\n  사유: {rule.why}")
                    for g in GLOBAL_RULES:
                        global_hits[id(g)] += pc.replace_in_tree(el, g.old, g.new)
                part = pc.clone_slide(srcs[it.deck], zips[it.deck], it.page, shell,
                                      media, mutate_element=mutate)
                pc.append_slide_part(shell, part, sid)
                n_ref += 1
            else:
                for spec in NEW_SLIDES[it.key]:
                    dd, dp = spec.get("donor", DONOR)
                    part = pc.clone_slide(srcs[dd], zips[dd], dp, shell, media)
                    fill_new_slide(part, spec)
                    pc.append_slide_part(shell, part, sid)
                    sid += 1
                    n_new += 1
                sid -= 1
            sid += 1

        dst_dir.mkdir(parents=True, exist_ok=True)
        out = dst_dir / out_path(day).name
        shell.save(str(out))
        outputs[day] = out
        if verbose:
            print(f"{day}: 재사용 {n_ref} + 신규 {n_new} = {n_ref + n_new}장 → {out}")

    for g in GLOBAL_RULES:
        if global_hits[id(g)] != g.expect:
            raise SystemExit(
                f"[중단] 전역 규칙 {g.old!r}: {global_hits[id(g)]}회 치환 (기대 {g.expect})\n"
                f"  사유: {g.why}")
    if verbose and GLOBAL_RULES:
        print(f"전역 규칙 {len(GLOBAL_RULES)}건 — 전부 기대 횟수와 일치")
    return outputs


if __name__ == "__main__":
    build()
    print("\n검증: uv run --project tools python tools/validate_slides.py")
