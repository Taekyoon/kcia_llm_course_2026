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
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.slide import Slide
from pptx.util import Cm, Pt

import pptxcommon as pc
from slide_layout import ROOT, DECKS, DECK_PAGES, PLACEMENT, WORK_PPT, NewSlide, SlideRef, out_path

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
class SlideExtra:
    """원본 한 장에 **도형을 더한다**. `SlideRule` 은 기존 텍스트 치환만 하므로
    "원본은 그대로 두고 내용을 덧붙인다" 를 표현할 수 없다 (2026-09-01).

    - `add_bullets`  가장 큰 텍스트 상자 끝에 (level, text) 를 덧붙인다
    - `body_h_cm`    그 상자의 높이 (덧붙인 만큼 늘려 준다)
    - `move_picture` 가장 큰 그림을 (left, top, w, h) cm 로 옮긴다 — 자리를 비우려고
    - `table`        `_add_table` 과 같은 사양. `left_cm` 을 따로 줄 수 있다
    """
    deck: str
    page: int
    why: str
    add_bullets: tuple = ()
    body_h_cm: float | None = None
    move_picture: tuple | None = None
    table: dict | None = None
    # 위 선언형으로 안 되는 자유 배치는 빌더 함수로 — slide 를 받아 도형을 그린다.
    # (frozen dataclass 라 함수도 값으로 담긴다)
    build_fn: object | None = None
    # 라운드트립 검증이 "이 문구들이 출력에 있어야 한다" 로 확인할 목록
    expect_texts: tuple = ()


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
              'raw_datasets = load_dataset("json", data_files=["assets/amazon_ko_sft.jsonl.gz", "data/amazon_ko_sft.mine.jsonl"])',
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
    SlideRule('2일차', 15, '[10:30]',
              '[100:105]',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 예시 20개가 아니라 5개', 2),
    SlideRule('2일차', 16, 'for i in range(100):',
              'for i in range(10):',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 노트북은 10건 (통계적 무의미도 명시)', 1),
    SlideRule('2일차', 32, 'indices = range(0,1000)',
              "n_train = min(1000, int(len(raw_datasets['train']) * 0.9))",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 고정 인덱스는 평가셋 0건 사고 전력, 비율 분할로', 1),
    SlideRule('2일차', 32, 'test_indices = range(1000, 1050)',
              'indices, test_indices = range(0, n_train), range(n_train, n_train + 50)',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 46, "output_dir = 'data/test_model'",
              "output_dir = 'data/sft_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 경로 분리 (DPO 이어받기의 전제)', 1),
    SlideRule('2일차', 46, 'overwrite_output_dir=True,',
              '',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — transformers 5 에서 제거된 인자', 1),
    SlideRule('2일차', 49, "'data/test_model'",
              "'data/sft_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 50, 'input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")',
              'inputs = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — transformers 5: return_dict + **inputs', 1),
    SlideRule('2일차', 50, 'input_ids=input_ids,',
              '**inputs,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 55, 'indices = range(0,500)',
              "n_train = min(500, int(len(raw_datasets['train']) * 0.9))",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 55, 'test_indices = range(500,550)',
              'indices, test_indices = range(0, n_train), range(n_train, n_train + 50)',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 63, 'from peft import LoraConfig, get_peft_model',
              'from peft import LoraConfig, PeftModel',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 63, 'model = get_peft_model(model, lora_config)',
              "model = PeftModel.from_pretrained(model, 'data/sft_model', is_trainable=True)  # SFT 이어받기",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — DPO 는 SFT 를 전제한다 (베이스에서 새 어댑터가 아니라)', 1),
    SlideRule('2일차', 64, "'data/test_model'",
              "'data/dpo_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 64, 'num_train_epochs=5,',
              'num_train_epochs=2,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 64, 'overwrite_output_dir=True,',
              '',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 67, "'data/test_model'",
              "'data/dpo_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 68, 'input_ids = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_tensors="pt").to("cuda")',
              'inputs = tokenizer.apply_chat_template(messages, truncation=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 68, 'input_ids=input_ids,',
              '**inputs,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 83, 'r=64,',
              'r=8,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 노트북: 생성 부담이 커서 가볍게', 1),
    SlideRule('2일차', 83, 'lora_alpha=16,',
              'lora_alpha=32,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 83, 'target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],',
              'target_modules=["q_proj", "v_proj"],',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 84, "'data/test_model'",
              "'data/grpo_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 84, 'max_prompt_length=128, ',
              '',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — GRPOConfig 에서 제거된 인자 (실측)', 1),
    SlideRule('2일차', 87, "'data/test_model'",
              "'data/grpo_model'",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 88, 'input_ids = tokenizer.apply_chat_template(messages, truncation=True,',
              'inputs = tokenizer.apply_chat_template(messages, truncation=True,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 88, 'add_generation_prompt=True, return_tensors="pt").to("cuda")',
              'add_generation_prompt=True, return_dict=True, return_tensors="pt").to("cuda")',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('2일차', 88, 'input_ids=input_ids,',
              '**inputs,',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치', 1),
    SlideRule('3일차', 56, '%pip install llama-index-llms-vllm',
              '%pip install llama-index-llms-openai-like',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — 현행 RAG 는 vLLM 서버에 HTTP 로 붙는다', 1),
    SlideRule('3일차', 56, '%pip install vllm',
              '# vLLM 서버는 별도 venv 에서 — 노트북 환경에는 설치하지 않습니다',
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — torch 하드핀 충돌 (venv 분리)', 1),
    SlideRule('3일차', 56, '%pip install datasets',
              '%pip install datasets PyStemmer matplotlib',
              '노트북 대조(2026-09-01) — p9 의 import Stemmer(PyStemmer) 와 display_source_node(matplotlib) 에 필요한데 설치 목록에 빠져 있었다', 1),
    SlideRule('3일차', 58, "dataset = load_dataset('heegyu/kowikitext', trust_remote_code=True, split='train[:1000]’)",
              "dataset = load_dataset('wikimedia/wikipedia', '20231101.ko', split='train[:1000]')",
              '슬라이드↔현행 노트북 대조(E-3)에서 확인된 불일치 — kowikitext 는 datasets 5.x 에서 로딩 불가 (실측)', 1),
    SlideRule('2일차', 65, 'eval_dataset=eval_dataset,',
              'eval_dataset=eval_dataset, processing_class=tokenizer,',
              '슬라이드↔현행 노트북 대조(E-3) — DPOTrainer 는 processing_class 필요', 1),
    SlideRule('1일차', 28, '먼저 개체명 추출 모델에 필요한 파이썬 패키지들을 선언합니다.',
              '먼저 감정 분류 모델에 필요한 파이썬 패키지들을 선언합니다.',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 28, 'Transformers: 개체명 추출 모델을 학습하기 위한 도구',
              'Transformers: 분류 모델을 학습하기 위한 도구',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 28, 'Datasets: 개체명 추출 모델 학습을 위한 오픈소스 데이터 전처리 도구',
              'Datasets: 학습 데이터를 불러오고 전처리하는 도구',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 28, 'Evaluate: 개체명 추출 평가 점수를 산출 하기 위한 도구',
              'Evaluate: 평가 점수를 산출하기 위한 도구',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 28, 'from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer',
              'from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer',
              '분류 실습인데 TokenClassification 을 import (C-4)', 1),
    SlideRule('1일차', 31, '학습 데이터는 “nsmc” 이라고 하는 허깅페이스 허브 내 데이터를 활용합니다.',
              '학습 데이터는 e9t/nsmc라는 허깅페이스 허브 내 데이터를 활용합니다.',
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    SlideRule('1일차', 31, 'dataset = load_dataset("nsmc")',
              'dataset = load_dataset("e9t/nsmc", revision="refs/convert/parquet")',
              'datasets 5.x 스크립트 로딩 제거 (CLAUDE.md §9)', 1),
    SlideRule('1일차', 39, 'OG: 그룹 또는 집단, TI: 시간, LC: 위치, DT: 날짜, PS: 사람',
              'PS 인물 · LC 장소 · OG 기관 · DT 날짜 · TI 시간 · QT 수량  (B-/I- 접두 + O = 13종)',
              'klue/klue ner 의 라벨 체계로 (kor_ner 7종 → 13종)', 1),
    SlideRule('1일차', 39, "Sequence(feature=ClassLabel(names=['I', 'O', 'B_OG', 'B_TI', 'B_LC', 'B_DT', 'B_PS'], id=None), length=-1, id=None)",
              '# 13개: O, B-DT, I-DT, B-LC, I-LC, B-OG, I-OG, B-PS, I-PS, B-QT, I-QT, B-TI, I-TI',
              '옛 kor_ner 출력 → klue 라벨 (실측 CLAUDE.md §9)', 1),
    SlideRule('1일차', 40, 'tokenized_inputs["labels"] = new_labels.replace()',
              'tokenized_inputs["labels"] = new_labels',
              '원본 슬라이드 오타 — list 에 replace 없음 (복사 시 AttributeError)', 1),
    SlideRule('1일차', 43, "['O', 'O', 'O', 'O', 'B-PS', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']",
              "# (예: ['O', 'O', 'B-PS', 'I-PS', 'O', ...] — 실행해 실제 나열을 확인하세요)",
              '옛 kor_ner 데이터의 출력 예시 — klue 와 다르다', 1),
    SlideRule('1일차', 47, "[{'entity_group': 'B_LC', 'score': 0.8979496, 'word': '대한민국 서울', 'start': 6, 'end': 13}, {'entity_group': 'B_PS', 'score': 0.8987875, 'word': '홍길동', 'start': 18, 'end': 21}]",
              '# 출력: 병합된 개체와 라벨(LC, PS 등)을 확인하세요 — 쪼갠 토큰이 다시 하나로 합쳐집니다',
              '옛 kor_ner 라벨 체계(B_LC)의 출력 예시', 1),
    SlideRule('2일차', 17, '여기서 정확한 예측값을 “긍정“ 또는 “부정＂으로 출력값을 받기 위해 Structured Guide 생성을 적용해봅니다.',
              "예측값을 '긍정'/'부정' 둘 중 하나로만 받도록 제약 디코딩(structured outputs)을 적용합니다 — 부탁이 아니라 문법으로 막는 것입니다.",
              '자기검토(3덱 전문 통독, 2026-08-27)에서 확인 — 다음 신규 장(강제력 3단계)과 용어 일치', 1),
    SlideRule('2일차', 50, '저장된 학습 모델을 활용하여 실제 학습이 잘 되었는지 확인해 봅니다.',
              '학습 데이터가 상품 요약이었으므로, 일부러 다른 주제(정보 이론)를 물어봅니다 — 파인튜닝의 경계가 보입니다.',
              '생성 예제가 학습 도메인과 무관하다는 오해 방지 (대조 지적)', 1),
    SlideRule('2일차', 55, '# remove this when done debugging',
              '',
              '원본 디버그 잔재 (노트북에서는 제거됨)', 1),
    SlideRule('2일차', 63, '효율적인 학습을 위해 LoRA 파인튜닝으로 진행합니다.',
              'SFT 에서 만든 LoRA 어댑터를 이어받아 학습을 계속합니다. (아래 LoraConfig 는 SFT 가 쓴 값의 참고용입니다)',
              '이어받기로 바뀐 코드와 설명 일치 (자기검토)', 1),
    SlideRule('2일차', 81, 'rewards_list = [1.0 if match else 0.0 for match in matches]',
              '',
              '미사용 죽은 줄 (바로 아래 같은 식을 return)', 1),
    SlideRule('3일차', 19, 'pip install openai vllm datasets',
              "%pip install -q -U 'openai<3' datasets    (vLLM 서버는 별도 venv)",
              'Amazon 환경 세팅 장의 옛 설치 명령 (E-2 누락분)', 1),
    SlideRule('3일차', 65, '만약 프롬프트를 수정하고자 한다면 템플릿 값에 새로운 프롬프트를 할당해주면 됩니다.',
              '프롬프트 교체는 update_prompts() 로 합니다 — 대입은 에러 없이 무시됩니다 (다음 장의 함정).',
              '바로 뒤 신규 장과 자기모순이던 옛 대입 방식 (자기검토)', 1),
    SlideRule('3일차', 65, 'prompts[\'response_synthesizer:refine_template\'].default_template.template = """원 질의는 다음과 같습니다.: {query_str}',
              'REFINE_KO = """원 질의는 다음과 같습니다.: {query_str}',
              '위와 동일', 1),
    SlideRule('3일차', 65, "print(prompts['response_synthesizer:refine_template'].default_template.template)",
              "query_engine.update_prompts({'response_synthesizer:refine_template': PromptTemplate(REFINE_KO)})   # from llama_index.core import PromptTemplate",
              '위와 동일 — 확인 print 대신 실제 등록', 1),
    # 원본 덱 코드 문자열의 닫는 따옴표가 곱슬(“, U+201C)이라 복붙 시 SyntaxError —
    # 곧은따옴표로 교정 (Fable 더블체크에서 발견, XML 실측 확정 2026-09-03)
    SlideRule('2일차', 13, '표현해주세요.“', '표현해주세요."',
              'TASK_PROMPT 닫는 따옴표 곱슬 → 곧은', 1),
    SlideRule('2일차', 14, '""“', '"""',
              '퓨샷 프롬프트 삼중따옴표 닫힘 세 번째가 곱슬 → 곧은', 1),
    SlideRule('2일차', 15, '""“', '"""',
              '퓨샷 프롬프트 삼중따옴표 닫힘 세 번째가 곱슬 → 곧은', 1),
    SlideRule('2일차', 37, '{% endfor %}“', '{% endfor %}"',
              'SFT DEFAULT_CHAT_TEMPLATE 닫는 따옴표 곱슬 → 곧은', 1),
    SlideRule('2일차', 57, '{% endfor %}“', '{% endfor %}"',
              'DPO DEFAULT_CHAT_TEMPLATE 닫는 따옴표 곱슬 → 곧은', 1),
    SlideRule('3일차', 49, 'json format""“', 'json format"""',
              'Amazon 프롬프트 삼중따옴표 닫힘 세 번째가 곱슬 → 곧은', 1),
    SlideRule('3일차', 65, '개선된 답변: ""“', '개선된 답변: """',
              'RAG REFINE_KO 삼중따옴표 닫힘 세 번째가 곱슬 → 곧은', 1),
    SlideRule('3일차', 68, '왜 꼬리질문이 필요할까요? \uf0e8 Q&A를 활용한 챗봇을 만든다면, 사용자가 직접 질문을 구체적으로 하기가 쉽지 않습니다.',
              '한 번의 검색으로 안 되는 질문이 있습니다 — "백남준과 맥스웰은 각각 어떤 분야의 인물인가요?" 를 통째로 검색하면 두 인물이 같이 나오는 문서를 찾게 됩니다.',
              '노트북의 프레이밍(복합 질문 분해)과 일치시킴', 1),
    SlideRule('3일차', 68, '꼬리질문 생성을 통해 사용자에게 제안을 할 수 있는 기능을 추가하여 Q&A 챗봇의 활성도를 높힐 기회를 얻을 수 있습니다.',
              '질문을 하위 질문으로 쪼개 각각 검색하고 합치면 됩니다 — 그 분해를 LLM 이 합니다.',
              '위와 동일 + 높힐 표기', 1),
    SlideRule('1일차', 27, '개체명 추출 모델은 허깅페이스에 공개된 모델과 데이터셋 위주로 학습을 진행합니다.',
              '모델과 데이터셋은 허깅페이스에 공개된 것을 씁니다.',
              '자체 리뷰 2회차(1·2일차 통독, 2026-08-27) — 분류 실습 도입 장인데 개체명 얘기를 하고 있었다', 1),
    SlideRule('1일차', 28, 'import evaluate',
              'import evaluate, numpy as np',
              '자체 리뷰 2회차(1·2일차 통독, 2026-08-27) — 이 import 로 시작한 분류 실습이 뒤에서 np.argmax 를 쓴다 (노트북엔 있다)', 1),
    # ── 1일차 2섹션(트랜스포머·BERT·GPT) 통독, 2026-09-01 ──────────────
    SlideRule('1일차', 50, '방식을 활 용하여', '방식을 활용하여',
              '1일차 통독 (2026-09-01)' + ' — 단어 가운데 공백 (원본 줄바꿈 흔적)'),
    SlideRule('1일차', 51, '단어들 간 의 관계가', '단어들 간의 관계가',
              '1일차 통독 (2026-09-01)' + ' — 단어 가운데 공백'),
    SlideRule('1일차', 53, 'Self-Attention 연산을 각각 N번 하는 과정',
              '입력 표현을 N개로 나눠 각각 Self-Attention을 구하고 다시 이어붙이는 과정',
              '1일차 통독 (2026-09-01)' + ' — 옆 그림은 Linear 로 쪼개 Concat 하는 구조인데 본문이 "각각 N번" 이라, '
                     '연산량이 N배 든다는 오해를 준다'),
    SlideRule('1일차', 53, 'N번 하는 것을 multi-head 수로 설정',
              '나누는 개수 N을 multi-head 수로 설정 — 나눠 하므로 전체 연산량은 크게 늘지 않음',
              '1일차 통독 (2026-09-01)' + ' — 위 규칙과 한 벌'),
    SlideRule('1일차', 53, '여러개로 보고자', '여러 개로 보고자', '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
    SlideRule('1일차', 54, '관계를 봐야하기 때문에', '관계를 봐야 하기 때문에', '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
    SlideRule('1일차', 54, 'Position Embedding을 이용하여',
              'Positional Encoding을 이용하여',
              '1일차 통독 (2026-09-01)' + ' — 옆 그림(원 논문 Figure 1)은 Positional Encoding 이다. '
                     '트랜스포머 원 논문은 학습하지 않는 사인·코사인 encoding 이고, '
                     '학습되는 position embedding 은 BERT·GPT 방식이라 개념이 갈린다'),
    SlideRule('1일차', 55, '3.\t트렌스포머와 GPT – GPT에 대해서',
              '2.\t트랜스포머와 ChatGPT – Transformer 소개',
              '1일차 통독 (2026-09-01)' + ' — 이 장 내용은 "Transformer 의의" 인데 헤더가 GPT 였다'),
    SlideRule('1일차', 56, '3.\t트렌스포머와 GPT – GPT에 대해서',
              '2.\t트랜스포머와 ChatGPT – BERT에 대해서',
              '1일차 통독 (2026-09-01)' + ' — 원본은 p55~p61 일곱 장이 모두 "GPT에 대해서" 인데 '
                     '실제로는 p55 Transformer 의의 · p56~58 BERT · p59~61 GPT 다. '
                     '바로 앞 장이 "Encoder는 BERT, Decoder는 GPT" 로 분기한 직후라 더 헷갈린다'),
    SlideRule('1일차', 57, '3.\t트렌스포머와 GPT – GPT에 대해서',
              '2.\t트랜스포머와 ChatGPT – BERT에 대해서', '1일차 통독 (2026-09-01)' + ' — 위와 한 벌'),
    SlideRule('1일차', 58, '3.\t트렌스포머와 GPT – GPT에 대해서',
              '2.\t트랜스포머와 ChatGPT – BERT에 대해서', '1일차 통독 (2026-09-01)' + ' — 위와 한 벌'),
    SlideRule('1일차', 58, 'KorQuad (Reading Comprehension Task)',
              'KorQuAD (Reading Comprehension Task)',
              '1일차 통독 (2026-09-01)' + ' — 정식 표기. 옆 그림도 KorQuAD 다'),
    SlideRule('1일차', 58, '대부분의 모델들이 Bert 기반 모델을',
              '대부분의 모델들이 BERT 기반 모델을', '1일차 통독 (2026-09-01)' + ' — 표기'),
    SlideRule('1일차', 56, 'Transformers Encoder 모델 활용한',
              'Transformer Encoder 모델 활용한',
              '1일차 통독 (2026-09-01)' + ' — 아키텍처는 단수 Transformer. 복수 Transformers 는 허깅페이스 '
                     '라이브러리 이름이고 같은 덱 실습 장에서 그 뜻으로 쓰인다'),
    SlideRule('1일차', 56, 'NSP (Next Sentence Prediction) 으로',
              'NSP (Next Sentence Prediction)으로', '1일차 통독 (2026-09-01)' + ' — 조사 붙여쓰기'),
    SlideRule('1일차', 59, 'Transformers Decoder 모델 활용한',
              'Transformer Decoder 모델 활용한', '1일차 통독 (2026-09-01)' + ' — 위 Encoder 와 한 벌'),
    SlideRule('1일차', 59, '초기 GPT-1은 Bert와 같이 Pre-training이후 finetune 방식으로 소개',
              '초기 GPT-1은 BERT와 같이 Pre-training 이후 fine-tuning 방식으로 소개',
              '1일차 통독 (2026-09-01)' + ' — Bert 표기 · "이후" 는 조사가 아니라 명사라 띄어야 한다 · '
                     'finetune 은 앞 장들의 fine-tuning 과 표기가 갈린다'),
    SlideRule('1일차', 60, '크기가 200M 수준으로 굉장히 작아',
              '크기가 최대 1.5B 수준으로 작아',
              '1일차 통독 (2026-09-01)' + ' — GPT-2 는 117M(small)~1.5B(XL) 이고 논문 대표 모델이 1.5B 다. '
                     '아래 그림이 GPT-3 175B 를 보여주는데 200M 이면 875배 도약으로 읽혀 '
                     '실제(약 100배)와 어긋난다'),
    SlideRule('1일차', 60, '태스크를 학습없이 해결할', '태스크를 학습 없이 해결할',
              '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
    SlideRule('1일차', 61, 'Reinforcement Learning Human Feedback)',
              'Reinforcement Learning from Human Feedback, RLHF)',
              '1일차 통독 (2026-09-01)' + ' — 정확한 명칭은 from Human Feedback. 그리고 3일차 내내 쓰는 '
                     'RLHF 약어가 이 덱 어디에도 도입되지 않아 여기서 준다'),

    # ── 1일차 3섹션(실습) 통독 — 노트북과 어긋난 곳 ────────────────────
    SlideRule('1일차', 46, 'eval_dataset=tokenized_dataset["test"],',
              'eval_dataset=tokenized_dataset["validation"],',
              '1일차 통독 (2026-09-01)' + ' — KLUE 에는 test 스플릿이 없다. 같은 덱의 NER 데이터 장이 이미 '
                     '"평가는 validation 으로 합니다" 라고 말해 놓고 여기서 test 를 쓴다. '
                     '노트북은 validation 을 쓴다 (실행 검증 완료)'),
    SlideRule('1일차', 36,
              'classifier = pipeline("sentiment-analysis", model="./nsmc_classifier/checkpoint-938")',
              'best = trainer.state.best_model_checkpoint or training_args.output_dir\n'
              'classifier = pipeline("sentiment-analysis", model=best, tokenizer=tokenizer)',
              '1일차 통독 (2026-09-01)' + ' — 노트북에 "checkpoint-938 을 하드코딩하면 배치·에폭·데이터 크기가 '
                     '조금만 달라져도 깨진다" 는 주석이 있는데 슬라이드가 그대로 하고 있었다'),
    SlideRule('1일차', 47, '# Replace this with your own checkpoint',
              '# 하드코딩하면 재현되지 않는다 — 학습 결과에서 받아온다',
              '1일차 통독 (2026-09-01)' + ' — 노트북과 동기화'),
    SlideRule('1일차', 47, 'model_checkpoint = "./ner_seq_classifier/checkpoint-183"',
              'model_checkpoint = trainer.state.best_model_checkpoint or training_args.output_dir',
              '1일차 통독 (2026-09-01)' + ' — 위와 한 벌. 노트북은 best_model_checkpoint 를 쓴다'),
    SlideRule('1일차', 34, '이 때 라벨 정보에 대한', '이때 라벨 정보에 대한', '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
    SlideRule('1일차', 33, '“accuracy”이라는 함수를', '“accuracy”라는 함수를',
              '노트북 설명과 용어·조사 일치'),
    SlideRule('1일차', 34, 'AutoModelSequenceClassification 클래스를',
              'AutoModelForSequenceClassification 클래스를',
              '현재 노트북의 실제 클래스명과 일치'),
    SlideRule('1일차', 35, 'load_best_model_at_end=True',
              'load_best_model_at_end=True,\nreport_to="tensorboard",',
              'Classification 노트북 셀 25와 완전 일치'),
    SlideRule('1일차', 35, '모델 학습을 위한 학습 설정값들을 TrainingArguments에 선언해줍니다.',
              'TrainingArguments로 학습 설정을 선언하고 Trainer.train()을 실행합니다.',
              '학습 코드 클리핑 방지를 위한 설명 압축'),
    SlideRule('1일차', 35, '학습 설정값들을 선언했다면, Trainer 라는 객체를 생성하고 train이라는 함수를 실행합니다.', '',
              '위 설명에 통합'),
    SlideRule('1일차', 37, 'Gemma-2-ko GPT 모델을 활용하여 모델 학습을 실습하고자 합니다.',
              'mmBERT 인코더 모델을 활용하여 사전학습+파인튜닝 방식 그대로 실습합니다.\n'
              'from transformers import AutoTokenizer\n'
              'from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer\n'
              'tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")',
              'NER 노트북 셀 3~4의 핵심 초기화를 실행 순서대로 반영'),
    SlideRule('1일차', 39, 'ner_feature = dataset["train"].features["ner_tags"]',
              'ner_feature = dataset["train"].features["ner_tags"]\n'
              'label_names = ner_feature.feature.names\nlabel_names',
              'NER 노트1 셀 12~13과 일치 — label_names를 사용하기 전에 정의'),
    SlideRule('1일차', 45, '이 때 라벨 정보에 대한', '이때 라벨 정보에 대한', '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
    SlideRule('1일차', 45, 'AutoModelTokenClassification 클래스를',
              'AutoModelForTokenClassification 클래스를',
              '현재 노트1의 실제 클래스명과 일치'),
    SlideRule('1일차', 45, 'label_names = ner_feature.feature.names', '',
              'label_names는 노트1 순서대로 p39에서 먼저 정의함'),
    SlideRule('1일차', 46, 'load_best_model_at_end=True',
              'load_best_model_at_end=True,\nreport_to="tensorboard",',
              'NER 노트1 셀 25와 완전 일치'),
    SlideRule('1일차', 46, '모델 학습을 위한 학습 설정값들을 TrainingArguments에 선언해줍니다.',
              'TrainingArguments로 학습 설정을 선언하고 Trainer.train()을 실행합니다.',
              '학습 코드 클리핑 방지를 위한 설명 압축'),
    SlideRule('1일차', 46, '학습 설정값들을 선언했다면, Trainer 라는 객체를 생성하고 train이라는 함수를 실행합니다.', '',
              '위 설명에 통합'),
    SlideRule('1일차', 27, '의도 분류 모델 대신 감정 분류 모델로 대신하여 실습합니다.',
              '의도 분류 대신 감정 분류로 실습합니다.', '중복 표현 정리'),
    SlideRule('1일차', 24, '의도 분류 예시', '개체명 추출 예시',
              '개체명 추출 장의 잘못된 예시 제목'),
    SlideRule('1일차', 40, '띄어쓰기 단위로 토큰화 하여 태깅을',
              '띄어쓰기 단위로 토큰화하여 태깅을', '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
    SlideRule('1일차', 42, 'load_dataset 으로 받은 dataset 객체는 map 이라는 함수를',
              'load_dataset으로 받은 dataset 객체는 map이라는 함수를', '1일차 통독 (2026-09-01)' + ' — 조사 붙여쓰기'),
    SlideRule('1일차', 24, '필요한 정보를 추출 하기 위한 태스크',
              '필요한 정보를 추출하기 위한 태스크', '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
    SlideRule('1일차', 26, '질의 텍스트로 부터 의도 분류',
              '질의 텍스트로부터 의도 분류', '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
    SlideRule('1일차', 32, '맞춰주도록 패딩작업을 수행합니다.',
              '맞춰주도록 패딩 작업을 수행합니다.', '1일차 통독 (2026-09-01)' + ' — 띄어쓰기'),
]

# 전역 규칙. expect 는 재사용 257장의 XML 결합 텍스트 실측값 (2026-08-27).
# 섹션 규칙은 이름으로 구분되어 서로의 출력을 다시 잡지 않는다 (충돌 분석 완료).
GLOBAL_RULES: list[GlobalSlideRule] = [
    GlobalSlideRule('3.\t트렌스포머와 GPT', '2.\t트랜스포머와 ChatGPT',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여. 통독(2026-09-01)에서 p55~58 헤더 4장을 내용에 맞게 페이지 규칙(Transformer 소개·BERT에 대해서)으로 바꿔 전역에서 빠진다 → 14에서 10', 10),
    GlobalSlideRule('2.\t자연어처리 머신러닝 소개', '3.\t자연어처리 머신러닝 소개',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('2. 자연어처리 머신러닝 소개', '3. 자연어처리 머신러닝 소개',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여. 자기검토에서 3장 드롭 → 23', 23),
    GlobalSlideRule('1. 도메인 최적화 프리트레인에 대해서', '4. 도메인 최적화 프리트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측) + 표기 정정', 1),
    GlobalSlideRule('1. 도메인 최적화 프리트레이닝에 대해서', '4. 도메인 최적화 프리트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 13),
    GlobalSlideRule('2. VLLM을 활용한 데이터처리 실습', '5. vLLM을 활용한 데이터 생성 실습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측). Amazon 은 정제가 아니라 생성 축이라 명칭도 맞춘다', 35),
    GlobalSlideRule('3. Llama Index를 활용한 RAG 실습', '1. Llama Index를 활용한 RAG 실습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여. 자기검토에서 출력 3장 드롭 → 15', 15),
    GlobalSlideRule('4. 지속적인 LLM 챗봇 개발을 위해서', '2. 태스크 정의와 평가',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측). 커리큘럼 세부항목명과 일치. 자기검토에서 p80 까지 드롭 → 간지만 1', 1),
    GlobalSlideRule('1. 퓨샷 러닝에 대해서', '3. 퓨샷 러닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('1. 퓨샷러닝에 대해서', '3. 퓨샷러닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 14),
    GlobalSlideRule('2. 포스트 트레이닝에 대해서', '4. 포스트 트레이닝에 대해서',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 11),
    GlobalSlideRule('3.\t 인스트럭션 모델 학습', '5.\t 인스트럭션 모델 학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여 (expect 는 2026-08-27 XML 실측)', 1),
    GlobalSlideRule('3. 인스트럭션 모델학습', '5. 인스트럭션 모델학습',
                    '재배치로 섹션 순서가 바뀌었다 — 새 덱 기준 번호·명칭 재부여. E-3 에서 KoAlpaca 4장 드롭 → 18', 18),
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
    GlobalSlideRule('실습에서는 ChatML 템플릿을 적용합니다.',
                    '실습에서는 커스텀 챗 템플릿을 정의해 적용합니다.',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인 — ChatML 이 아니라 커스텀 템플릿 (명칭 오류)', 4),
    GlobalSlideRule('파라메터',
                    '파라미터',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인 — 표기 통일 (노트북과 동일)', 8),
    GlobalSlideRule('Bert 모델',
                    'BERT 모델',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인 — 표기 (p56·57·58 제목 + p57 본문)', 4),
    GlobalSlideRule('Bert를',
                    'BERT를',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    GlobalSlideRule('Bert는',
                    'BERT는',
                    '자기검토(3덱 전문 통독, 2026-08-27)에서 확인', 1),
    GlobalSlideRule('Gemma-2-ko GPT 모델을 활용하여 모델 학습을 실습하고자 합니다.',
                    'mmBERT 인코더 모델을 활용하여 사전학습+파인튜닝 방식 그대로 실습합니다.',
                    '분류·NER 본문 서술이 여전히 Gemma 였다 (C-1 잔여). NER p37은 페이지 규칙에서 초기화 코드까지 함께 교체', 1),
    GlobalSlideRule('tokenizer=tokenizer,',
                    'processing_class=tokenizer,',
                    'transformers 5 — Trainer(tokenizer=) 제거 (CLAUDE.md §9)', 2),
    GlobalSlideRule('LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct',
                    'Qwen/Qwen3-4B-Instruct-2507',
                    'EXAONE NC 라이선스 → Qwen3 (CLAUDE.md §10). 서빙 줄 교체(RULES)로 3곳이 먼저 사라져 7', 7),
    GlobalSlideRule('"beomi/gemma-ko-2b"',
                    '"jhu-clsp/mmBERT-base"',
                    '분류·NER 모델 교체 — 인코더 전환. 자기검토에서 p29 드롭 → 2', 2),
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
    GlobalSlideRule('Trainer 라는', 'Trainer라는',
                    '조사 붙여쓰기 — 1일차 학습 설명 두 곳은 페이지 규칙에서 압축하여 전역 치환 대상이 사라졌다', 0),
    GlobalSlideRule('pipeline 이라는', 'pipeline이라는',
                    '조사 붙여쓰기 (위와 동일)', 2),
    GlobalSlideRule('RNN모델에서', 'RNN 모델에서',
                    '통독(2026-09-01) — 붙어 있어야 할 곳이 아니라 띄어야 할 곳', 1),
]

# 원본 장에 도형을 덧붙이는 규칙 (SlideExtra). 원본 내용은 그대로 두고 더한다.
def _kq_run(p, text, *, size, bold=False, color=None, hl=None):
    """문단에 run 하나를 더한다. hl 은 형광(highlight) 색 hex."""
    r = p.add_run()
    r.text = text
    r.font.name = TABLE_FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    if color is not None:
        r.font.color.rgb = RGBColor.from_string(color)
    rPr = r._r.get_or_add_rPr()
    rPr.set(qn("a:lang"), "ko-KR")
    ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", TABLE_FONT)
    if hl is not None:
        h = etree.Element(qn("a:highlight"))
        clr = etree.SubElement(h, qn("a:srgbClr"))
        clr.set("val", hl)
        # highlight 는 rPr 안에서 fill 계열 뒤, latin 앞 순서를 지켜야 한다 — 맨 앞에 둔다
        rPr.insert(0, h)
    return r


def _kq_card(slide, l, t, w, h, *, fill, line=None, radius=0.08):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(l), Cm(t), Cm(w), Cm(h))
    box.adjustments[0] = radius
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = RGBColor.from_string(line)
        box.line.width = Pt(1)
    box.shadow.inherit = False
    return box


import json as _json

# Amazon 구조화 출력 코드는 노트북에서 §9 함정(길이 상한·finish_reason·temperature)을
# 고쳤는데 슬라이드는 옛 버그 코드였다. 슬라이드 코드박스를 **노트북 셀로 통째 교체**해
# 완전 동기화한다 — 전사 오류 0, 노트북이 바뀌면 자동 반영 (2026-09-01 사용자 지시).
_AMZ_NB = ROOT / "work" / "notebook" / "2일차" / "2_HPC_Amazon요약실습.ipynb"
_amz_cache = None


def _amz_cells():
    global _amz_cache
    if _amz_cache is None:
        nb = _json.loads(_AMZ_NB.read_text(encoding="utf-8"))
        _amz_cache = ["".join(c.get("source", [])) for c in nb["cells"]
                      if c.get("cell_type") == "code"]
    return _amz_cache


def _amz_src(anchor):
    for s in _amz_cells():
        if anchor in s:
            return s.rstrip("\n")
    raise SystemExit(f"[중단] Amazon 노트북에서 앵커 못 찾음: {anchor!r}")


# 노트북은 pydantic 클래스와 gen 함수를 **한 셀**에 넣는다. 슬라이드는 둘을 다른
# 장에 보여주므로, 셀을 class 부분과 gen 부분으로 나눠 가져온다.
def _amz_pydantic_src(anchor):
    s = _amz_src(anchor)
    i = s.find("\ndef ")
    return (s[:i] if i != -1 else s).rstrip("\n")


def _amz_gen_src(fn_name):
    s = _amz_src("def " + fn_name)
    i = s.find("def " + fn_name)
    return s[i:].rstrip("\n")


def _set_code_box(slide, box_anchor, source_text):
    """코드박스(문단=줄)를 source_text 로 재구성한다. 첫 코드 문단을 서식 템플릿으로."""
    box = next((sh for sh in slide.shapes
                if sh.has_text_frame and box_anchor in sh.text_frame.text), None)
    if box is None:
        raise SystemExit(f"[중단] Amazon 코드박스 못 찾음: {box_anchor!r}")
    tb = box.text_frame._txBody
    ps = tb.findall(qn("a:p"))
    tmpl = next((pe for pe in ps if pe.findall(".//" + qn("a:t"))), None)
    if tmpl is None:
        raise SystemExit(f"[중단] 코드박스에 템플릿 문단 없음: {box_anchor!r}")
    new_ps = []
    for ln in source_text.rstrip("\n").split("\n"):
        c = copy.deepcopy(tmpl)
        for br in c.findall(".//" + qn("a:br")):
            br.getparent().remove(br)
        runs = c.findall(".//" + qn("a:t"))
        runs[0].text = ln if ln else " "
        # 들여쓰기 공백 보존 (xml:space=preserve) — qn 은 xml 접두사를 모른다
        runs[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        for r in runs[1:]:
            r.text = ""
        new_ps.append(c)
    for pe in ps:
        tb.remove(pe)
    for np_ in new_ps:
        tb.append(np_)


def sync_amazon_pydantic(box_anchor, class_anchor, imports):
    def _fn(slide):
        _set_code_box(slide, box_anchor, imports + "\n\n" + _amz_pydantic_src(class_anchor))
    return _fn


def sync_amazon_gen(def_anchor, fn_name):
    def _fn(slide):
        box = next((sh for sh in slide.shapes
                    if sh.has_text_frame and def_anchor in sh.text_frame.text), None)
        had_call = box is not None and ("= " + fn_name + "(") in box.text_frame.text
        src = _amz_gen_src(fn_name)
        if had_call:
            src += "\n\n" + _amz_src("= " + fn_name + "(")
        _set_code_box(slide, def_anchor, src)
    return _fn


def sync_amazon_full(box_anchor, cell_anchor, imports):
    # 원본 슬라이드가 class+gen 을 한 박스에 둔 경우 — 셀 전체(class+gen)로 교체.
    def _fn(slide):
        _set_code_box(slide, box_anchor, imports + "\n\n" + _amz_src(cell_anchor))
    return _fn


def _code_line_after(box, anchor_substr, new_text):
    """코드 상자에서 anchor_substr 이 든 문단 바로 뒤에 new_text 한 줄을 넣는다.
    코드 슬라이드는 줄마다 별도 <a:p> 라 문단을 복제해 삽입하면 서식이 유지된다."""
    tb = box.text_frame._txBody
    for pe in tb.findall(qn("a:p")):
        txt = "".join(t.text or "" for t in pe.findall(".//" + qn("a:t")))
        if anchor_substr in txt:
            c = copy.deepcopy(pe)
            for br in c.findall(".//" + qn("a:br")):
                br.getparent().remove(br)
            runs = c.findall(".//" + qn("a:t"))
            runs[0].text = new_text
            for r in runs[1:]:
                r.text = ""
            pe.addnext(c)
            return
    raise SystemExit(f"[중단] 코드 앵커 못 찾음: {anchor_substr!r}")


def add_subq_question_gen(slide):
    """p69(SubQuestionQueryEngine) — 노트북과 맞춘다.

    SubQuestionQueryEngine 의 기본 질문 생성기는 OpenAI 전용이라, 로컬 vLLM 환경에서는
    question_gen 을 직접 넘겨야 한다(노트북 주석). 슬라이드 코드엔 이 줄과 import 가
    빠져 있어 그대로는 깨진다 → import 1줄 + 인자 1줄을 노트북과 동일하게 삽입한다.
    """
    box = next((sh for sh in slide.shapes
                if sh.has_text_frame
                and "SubQuestionQueryEngine.from_defaults" in sh.text_frame.text), None)
    if box is None:
        raise SystemExit("[중단] p69 SubQuestionQueryEngine 코드 상자를 못 찾음")
    _code_line_after(box, "from llama_index.core.tools import",
                     "from llama_index.core.question_gen import LLMQuestionGenerator")
    ind = chr(160) + " " + chr(160) + " "
    _code_line_after(box, "query_engine_tools=query_engine_tools,",
                     ind + "question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),")


def build_korquad_example(slide):
    """p58(KorQuAD)에 예제를 세련되게 얹는다 — 리더보드 그림은 뺀다.

    핵심 메시지: "정답은 새로 쓰는 것이 아니라 지문 속 '구간'". 그래서 지문에서
    정답 세 곳을 형광으로 칠하고, 아래 질문 배지의 색을 같게 맞춘다 (눈으로 연결).
    원본 텍스트(상단 4줄)는 건드리지 않는다 (2026-09-01 사용자 지시).
    """
    # ① 낡은 KorQuAD 2.0 리더보드 캡처 제거
    pics = [sh for sh in slide.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE
            and (sh.width or 0) > Cm(8)]
    for pic in pics:
        pic._element.getparent().remove(pic._element)

    Y = 15.2  # 예제 영역 시작 (원본 본문 4줄 아래)
    C1, C2, C3 = "FFE08A", "F7C6D9", "BCE5C6"   # 정답 3색: 노랑·분홍·초록

    # ② 소제목
    cap = slide.shapes.add_textbox(Cm(10.6), Cm(Y), Cm(38), Cm(1.0))
    p = cap.text_frame.paragraphs[0]
    _kq_run(p, "예제 ", size=17, bold=True, color="1F4E79")
    _kq_run(p, "— BERT가 시작·끝 위치를 골라 ", size=15, color="595959")
    _kq_run(p, "지문 속 '구간(span)'", size=15, bold=True, color="1F4E79")
    _kq_run(p, "을 찾습니다  (KorQuAD 1.0 · 문서 \"파우스트_서곡\")", size=15, color="595959")

    # ③ 지문 카드 — 정답 세 곳을 형광으로
    card = _kq_card(slide, 10.6, Y + 1.2, 38.6, 4.6, fill="F4F8FC", line="C9DCEC")
    tf = card.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    for m in (0, 1, 2, 3):
        tf.margin_left = tf.margin_right = Cm(0.5)
        tf.margin_top = tf.margin_bottom = Cm(0.3)
    p = tf.paragraphs[0]
    p.line_spacing = 1.3
    seg = [
        ("1839년 바그너는 괴테의 파우스트를 읽고 그 내용에 끌려 하나의 ", None),
        ("교향곡", C1),
        ("을 쓰려는 뜻을 갖는다. 파리에서 ", None),
        ("베토벤의 교향곡 9번", C3),
        ("을 듣고 깊은 감명을 받았고, 이 곡의 ", None),
        ("1악장", C2),
        ("을 쓴 뒤에 중단했다.", None),
    ]
    for txt, hl in seg:
        # 도형 안 run 의 기본 글자색이 흰색이라 연한 카드에 묻힌다 — 검정 명시 (2026-09-01)
        _kq_run(p, txt, size=15, bold=hl is not None, color="1A1A1A", hl=hl)

    # ④ 질문 배지 세 줄 — 색이 지문 형광과 매칭된다
    rows = [
        (C1, "교향곡", "무엇을 쓰고자 했는가?", "54"),
        (C3, "베토벤의 교향곡 9번", "어떤 곡의 영향을 받았는가?", "194"),
        (C2, "1악장", "어디까지 쓴 뒤에 중단했는가?", "421"),
    ]
    qy = Y + 6.2
    for i, (color, ans, q, pos) in enumerate(rows):
        y = qy + i * 1.75
        badge = _kq_card(slide, 10.6, y, 12.5, 1.4, fill=color, radius=0.5)
        bt = badge.text_frame
        bt.vertical_anchor = MSO_ANCHOR.MIDDLE
        bp = bt.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        _kq_run(bp, ans, size=15, bold=True, color="1A1A1A")
        qbox = slide.shapes.add_textbox(Cm(23.6), Cm(y), Cm(25.6), Cm(1.4))
        qt = qbox.text_frame
        qt.vertical_anchor = MSO_ANCHOR.MIDDLE
        qp = qt.paragraphs[0]
        _kq_run(qp, "Q. ", size=14, bold=True, color="8C8C8C")
        _kq_run(qp, q, size=15, color="1A1A1A")
        _kq_run(qp, f"    answer_start = {pos}", size=12, color="A6A6A6")


def compact_training_code_slide(slide):
    """학습 코드 상자의 빈 문단을 제거한다.

    설명 두 줄을 한 줄로 합친 뒤 남은 빈 불릿이 코드를 아래로 밀어
    history = trainer.train()을 클리핑했다. 코드·글자 크기는 그대로 두고 빈 문단만 뺀다.
    """
    code_box = None
    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        tx_body = shape.text_frame._txBody
        for p in list(tx_body.findall(qn("a:p"))):
            text = "".join((t.text or "") for t in p.findall(".//" + qn("a:t")))
            if not text.strip() and len(tx_body.findall(qn("a:p"))) > 1:
                tx_body.remove(p)
        if "training_args = TrainingArguments" in shape.text_frame.text:
            code_box = shape
    if code_box is None:
        raise SystemExit("[중단] 학습 코드 상자를 못 찾았습니다")
    code_box.top -= Cm(1.2)


# ── p24·p25 도해: 같은 문장을 RNN(직렬) vs 어텐션(전연결) 으로 대비 ────────
_TOKENS = ["나", "는", "사과", "를", "먹었다"]
_TOK_W, _TOK_H, _GAP = 5.4, 1.5, 1.0
_ROW_W = len(_TOKENS) * _TOK_W + (len(_TOKENS) - 1) * _GAP   # 31.0cm
_ROW_L = 10.6 + (38.6 - _ROW_W) / 2                          # 본문 폭 중앙 정렬


def _tok_centers(top_cm):
    xs = [_ROW_L + i * (_TOK_W + _GAP) + _TOK_W / 2 for i in range(len(_TOKENS))]
    return xs, top_cm + _TOK_H / 2


def _draw_tokens(slide, top_cm, *, highlight=None, dim=False):
    for i, tok in enumerate(_TOKENS):
        l = _ROW_L + i * (_TOK_W + _GAP)
        hot = highlight is not None and i == highlight
        box = _kq_card(slide, l, top_cm, _TOK_W, _TOK_H,
                       fill=("D6E8FF" if hot else ("F0F0F0" if dim else "EAF2FB")),
                       line=("6BA6E8" if hot else "D0D8E0"))
        box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = box.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        _kq_run(p, tok, size=15, bold=hot, color=("134A8E" if hot else "3A3A3A"))


def _line(slide, x1, y1, x2, y2, *, color, width_pt, dashed=False, arrow=True):
    conn = slide.shapes.add_connector(2, Cm(x1), Cm(y1), Cm(x2), Cm(y2))  # 2 = straight
    conn.line.color.rgb = RGBColor.from_string(color)
    conn.line.width = Pt(width_pt)
    ln = conn.line._get_or_add_ln()
    if dashed:
        d = etree.SubElement(ln, qn("a:prstDash")); d.set("val", "dash")
    if arrow:
        tail = etree.SubElement(ln, qn("a:tailEnd"))
        tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    return conn


def draw_rnn_limit(slide):
    """RNN — 한 칸씩 순서대로 전달, 멀면 흐려진다 (p25 어텐션의 동기)."""
    top = 18.0
    xs, cy = _tok_centers(top)
    # 위 캡션: 병렬화 한계
    cap = slide.shapes.add_textbox(Cm(_ROW_L), Cm(top - 2.4), Cm(_ROW_W), Cm(1.0))
    cp = cap.text_frame.paragraphs[0]
    cp.alignment = PP_ALIGN.CENTER
    _kq_run(cp, "앞 단어부터 한 칸씩 전달 — 앞 계산이 끝나야 다음이 시작된다 ", size=12, color="595959")
    _kq_run(cp, "(병렬화가 어렵다)", size=12, bold=True, color="595959")
    # 멀리 떨어진 관계 (연한 점선 아치) — 첫 단어에서 마지막 단어로
    _line(slide, xs[0], top - 0.7, xs[-1], top - 0.7,
          color="C4B0D8", width_pt=1.5, dashed=True, arrow=True)
    lab = slide.shapes.add_textbox(Cm(_ROW_L), Cm(top - 1.5), Cm(_ROW_W), Cm(0.8))
    lp = lab.text_frame.paragraphs[0]
    lp.alignment = PP_ALIGN.CENTER
    _kq_run(lp, "멀리 떨어진 단어일수록 관계가 흐려진다", size=11, color="8A7AA6")
    # 인접 전달 화살표
    for i in range(len(xs) - 1):
        _line(slide, xs[i] + _TOK_W / 2, cy, xs[i + 1] - _TOK_W / 2, cy,
              color="5B6B7B", width_pt=1.8)
    _draw_tokens(slide, top)
    cap2 = slide.shapes.add_textbox(Cm(_ROW_L), Cm(top + _TOK_H + 0.4), Cm(_ROW_W), Cm(1.0))
    c2 = cap2.text_frame.paragraphs[0]
    c2.alignment = PP_ALIGN.CENTER
    _kq_run(c2, "문장이 길어질수록 두 한계가 함께 조여 온다 — 그래서 ", size=12, color="595959")
    _kq_run(c2, "어텐션", size=12, bold=True, color="1F4E79")
    _kq_run(c2, "이 나온다", size=12, color="595959")


APPLE_ASSET = ROOT / "work" / "assets" / "p25_apple.png"


def add_apple_image(slide):
    """p25(어텐션 개념)에 사과 장면 이미지를 넣는다 — 있으면.

    브라우저에서 자동 회수가 막혀(2026-09-01) 강사가 work/assets/p25_apple.png 로
    저장한다. 파일이 없으면 조용히 건너뛴다 — 빌드는 깨지지 않는다.
    """
    if not APPLE_ASSET.exists():
        print(f"  [안내] {APPLE_ASSET.name} 없음 — p25 사과 이미지 건너뜀")
        return
    slide.shapes.add_picture(str(APPLE_ASSET), Cm(20.0), Cm(15.6), Cm(19.6), Cm(10.9))


def draw_attention(slide):
    """어텐션 — 질의 단어 '먹었다'가 각 단어를 보는 정도를 막대로.

    교차선 대신 막대로 그린다 — 서로 겹치지 않고, 앞서 보여준 위젯과도 같은 형태다.
    """
    top = 18.9
    xs, _ = _tok_centers(top)
    q = 4  # 질의 = '먹었다'
    weights = [0.30, 0.05, 0.45, 0.05, 0.15]

    cap = slide.shapes.add_textbox(Cm(_ROW_L), Cm(top - 1.5), Cm(_ROW_W), Cm(1.0))
    cp = cap.text_frame.paragraphs[0]
    cp.alignment = PP_ALIGN.CENTER
    _kq_run(cp, "질의 단어 ", size=12, color="595959")
    _kq_run(cp, "'먹었다'", size=12, bold=True, color="1F4E79")
    _kq_run(cp, "가 각 단어를 보는 정도 (어텐션 점수)", size=12, color="595959")

    _draw_tokens(slide, top, highlight=q)

    y0 = top + _TOK_H + 0.35        # 막대 시작(위)
    scale = 7.1                     # 최대 0.45 → 3.2cm
    for i, w in enumerate(weights):
        h = max(w * scale, 0.12)
        strong = w >= 0.25
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                     Cm(xs[i] - 0.8), Cm(y0), Cm(1.6), Cm(h))
        bar.fill.solid()
        bar.fill.fore_color.rgb = RGBColor.from_string("378ADD" if strong else "C8D4E0")
        bar.line.fill.background()
        bar.shadow.inherit = False
        num = slide.shapes.add_textbox(Cm(xs[i] - 1.0), Cm(y0 + h + 0.05), Cm(2.0), Cm(0.7))
        npar = num.text_frame.paragraphs[0]
        npar.alignment = PP_ALIGN.CENTER
        _kq_run(npar, f"{w:.2f}", size=11,
                bold=strong, color=("1F4E79" if strong else "8A8A8A"))

    cap2 = slide.shapes.add_textbox(Cm(_ROW_L), Cm(y0 + 3.2 + 0.7), Cm(_ROW_W), Cm(1.0))
    c2 = cap2.text_frame.paragraphs[0]
    c2.alignment = PP_ALIGN.CENTER
    _kq_run(c2, "주어 '나'와 목적어 '사과'에 높은 점수 — 순서가 아니라 의미로, 한 번에 본다",
            size=12, color="595959")


def _box(slide, l, t, w, h, text, *, fill, line=None, size=13, bold=False,
         tcolor="1A1A1A", radius=0.08):
    b = _kq_card(slide, l, t, w, h, fill=fill, line=line, radius=radius)
    b.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    b.text_frame.word_wrap = True
    p = b.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    for i, seg in enumerate(text.split("\n")):
        para = p if i == 0 else b.text_frame.add_paragraph()
        para.alignment = PP_ALIGN.CENTER
        _kq_run(para, seg, size=size, bold=bold, color=tcolor)
    return b


def _label(slide, l, t, w, text, *, size=12, color="595959", align=PP_ALIGN.CENTER, bold=False):
    tb = slide.shapes.add_textbox(Cm(l), Cm(t), Cm(w), Cm(0.9))
    p = tb.text_frame.paragraphs[0]
    p.alignment = align
    _kq_run(p, text, size=size, bold=bold, color=color)


def draw_lora(slide):
    """LoRA — 얼린 W 옆에 저랭크 A·B 만 학습 (출력 = W·x + B·A·x)."""
    cy = 20.2
    _box(slide, 11.0, cy - 0.9, 3.4, 1.8, "입력\nx", fill="EDEDED", line="C9C9C9", size=13)
    # 위 경로: 얼린 W
    _box(slide, 18.0, cy - 3.0, 9.0, 2.2, "W  (원래 가중치)\n❄ 얼림 · 수억 개",
         fill="ECECEC", line="B8B8B8", size=13, tcolor="6B6B6B")
    # 아래 경로: 저랭크 A·B (학습)
    _box(slide, 18.0, cy + 0.9, 4.1, 2.0, "A\n(d×r)", fill="D6E8FF", line="378ADD", size=12, tcolor="134A8E")
    _box(slide, 22.9, cy + 0.9, 4.1, 2.0, "B\n(r×d)", fill="D6E8FF", line="378ADD", size=12, tcolor="134A8E")
    # 합류 ⊕ → 출력
    _box(slide, 30.5, cy - 1.0, 1.9, 2.0, "＋", fill="FFFFFF", line="9AA6B2", size=20, radius=0.5)
    _box(slide, 34.5, cy - 0.9, 3.6, 1.8, "출력", fill="EAF2FB", line="AEBECE", size=13)
    # 연결선
    _line(slide, 14.4, cy, 18.0, cy - 1.9, color="9AA6B2", width_pt=1.4)          # x→W
    _line(slide, 14.4, cy, 18.0, cy + 1.9, color="378ADD", width_pt=1.4)          # x→A
    _line(slide, 22.1, cy + 1.9, 22.9, cy + 1.9, color="378ADD", width_pt=1.4)    # A→B
    _line(slide, 27.0, cy - 1.9, 30.5, cy, color="9AA6B2", width_pt=1.4)          # W→＋
    _line(slide, 27.0, cy + 1.9, 30.5, cy, color="378ADD", width_pt=1.4)          # B→＋
    _line(slide, 32.4, cy, 34.5, cy, color="5B6B7B", width_pt=1.6)                # ＋→출력
    _label(slide, 18.0, cy - 3.9, 9.0, "학습 안 함", size=11, color="9A9A9A")
    _label(slide, 18.0, cy + 3.0, 9.0, "여기만 학습 — 저랭크 보정 (SFT trainable 2.99%)",
           size=12, color="134A8E", bold=True)
    _label(slide, 11.0, cy - 4.6, 27.0, "출력 = W·x  +  B·A·x", size=15, color="1F4E79",
           align=PP_ALIGN.CENTER, bold=True)


def draw_grpo_group(slide):
    """GRPO — 한 문제에 답 N개, 그룹 평균 대비 상대 평가로 확률을 올리고 내린다."""
    ans = [("답 1", "0.9", True), ("답 2", "0.2", False),
           ("답 3", "0.7", True), ("답 4", "0.0", False)]
    ys = [17.7, 20.0, 22.3, 24.6]                 # 본문 아래에서 시작 (겹침 방지)
    q_cy = (ys[0] + ys[-1] + 1.9) / 2             # 답 박스들의 세로 중앙
    _box(slide, 11.0, q_cy - 1.0, 4.6, 2.0, "문제\n(프롬프트)", fill="EDEDED", line="C9C9C9", size=13)
    for (name, rw, up), y in zip(ans, ys):
        col = ("D9F0DF", "3E9E5A") if up else ("FAD9E0", "C65B77")
        _box(slide, 19.5, y, 9.5, 1.9, f"{name}    보상 {rw}    {'▲ 확률↑' if up else '▼ 확률↓'}",
             fill=col[0], line=col[1], size=12, tcolor="1A1A1A")
        _line(slide, 15.6, q_cy, 19.5, y + 0.95, color="9AA6B2", width_pt=1.2)
    # 그룹 평균 기준선
    _line(slide, 30.2, ys[0], 30.2, ys[-1] + 1.9, color="8A8A8A", width_pt=1.2, dashed=True, arrow=False)
    _label(slide, 30.4, q_cy - 0.5, 8.0, "그룹 평균 (기준선)", size=12, color="6B6B6B", align=PP_ALIGN.LEFT)
    _label(slide, 11.0, ys[-1] + 2.1, 27.0,
           "num_generations=4 — 그룹 안에서 상대 평가라 별도 기준선 모델이 필요 없다",
           size=12, color="595959")


def draw_rag_pipeline(slide):
    """RAG — 질문 → BM25 검색 → top-k 문서 → LLM → 답 (전체 뼈대)."""
    cy = 21.5
    steps = [
        (11.0, 6.2, "질문", "EDEDED", "C9C9C9", "1A1A1A"),
        (18.4, 6.6, "BM25 검색", "FBEED0", "E0B84E", "7A5A10"),
        (26.2, 6.6, "top-3 문서", "EAF2FB", "AEBECE", "1A1A1A"),
        (34.0, 5.0, "LLM\n답 생성", "D6E8FF", "378ADD", "134A8E"),
    ]
    xs_right = []
    for l, w, txt, fill, line, tc in steps:
        _box(slide, l, cy - 1.1, w, 2.2, txt, fill=fill, line=line, size=13, tcolor=tc)
        xs_right.append(l + w)
    _line(slide, xs_right[0], cy, steps[1][0], cy, color="5B6B7B", width_pt=1.6)
    _line(slide, xs_right[1], cy, steps[2][0], cy, color="5B6B7B", width_pt=1.6)
    _line(slide, xs_right[2], cy, steps[3][0], cy, color="5B6B7B", width_pt=1.6)
    _box(slide, 39.6, cy - 0.9, 3.4, 1.8, "답", fill="EAF2FB", line="AEBECE", size=13)
    _line(slide, xs_right[3], cy, 39.6, cy, color="5B6B7B", width_pt=1.6)
    # 위키 인덱스가 BM25 를 받쳐 준다
    _box(slide, 17.6, cy + 2.6, 8.2, 1.7, "위키 인덱스 (아침에 저장)", fill="F4F4F4", line="D0D0D0",
         size=11, tcolor="6B6B6B")
    _line(slide, 21.7, cy + 2.6, 21.7, cy + 1.1, color="B0B0B0", width_pt=1.2)
    _label(slide, 11.0, cy - 3.0, 32.0,
           "지식은 검색으로 넣는다 — 모델을 바꾸지 않고 문서를 바꿔 답을 바꾼다",
           size=12, color="1F4E79", bold=True)


_IMP = "from pydantic import BaseModel, Field\nfrom typing import List"
_IMP_ANN = "from pydantic import BaseModel, Field\nfrom typing import Annotated, List"


def _amz_extra(page, why, *build_fns, expect=()):
    # 한 페이지에 여러 코드박스(클래스+gen)면 build_fn 을 순서대로 적용한다.
    def _fn(slide):
        for f in build_fns:
            f(slide)
    return SlideExtra(deck="3일차", page=page,
                      why="Amazon 구조화 출력 노트북 동기화 (§9 터짐 버그) — " + why,
                      build_fn=_fn, expect_texts=expect)


def draw_causal_mask(slide):
    """Causal mask — 토큰×토큰 격자. 자기 자신과 앞쪽만 볼 수 있고 뒤(위삼각)는 가린다."""
    toks = ["나", "는", "밥", "을", "먹"]
    n = len(toks)
    cell = 2.0
    gx, gy = 18.6, 17.4          # 격자 좌상단
    # 열 머리표(보는 대상) / 행 머리표(현재 토큰)
    _label(slide, gx, gy - 1.4, n * cell, "→ 이 토큰들을 볼 수 있나", size=11, color="8A8A8A")
    _label(slide, 10.6, gy + n * cell / 2 - 0.4, 7.0, "현재\n토큰", size=11, color="8A8A8A",
           align=PP_ALIGN.LEFT)
    for j, t in enumerate(toks):   # 열 라벨
        _label(slide, gx + j * cell, gy - 0.7, cell, t, size=12, color="3A3A3A")
    for i, t in enumerate(toks):   # 행 라벨
        _label(slide, gx - 2.2, gy + i * cell + 0.5, 2.0, t, size=12, color="3A3A3A",
               align=PP_ALIGN.LEFT)
    for i in range(n):
        for j in range(n):
            allowed = j <= i       # 자기 자신·앞쪽만
            fill, line = ("D9F0DF", "3E9E5A") if allowed else ("3A3A3A", "222222")
            b = _kq_card(slide, gx + j * cell, gy + i * cell, cell - 0.15, cell - 0.15,
                         fill=fill, line=line, radius=0.06)
            b.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = b.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            _kq_run(p, "✓" if allowed else "✕", size=12, bold=True,
                    color=("1A5A2E" if allowed else "FFFFFF"))
    rx = gx + n * cell + 0.6
    _label(slide, rx, gy + 0.4, 12.0, "초록 = 볼 수 있다\n(자기 자신·앞쪽)", size=12,
           color="1A5A2E", align=PP_ALIGN.LEFT, bold=True)
    _label(slide, rx, gy + 3.2, 12.0, "검정 = 가린다\n(아직 안 나온 뒤쪽)", size=12,
           color="3A3A3A", align=PP_ALIGN.LEFT, bold=True)
    _label(slide, 10.6, gy + n * cell + 0.5, 39.0,
           "위쪽 삼각(뒤 단어)을 -inf 로 막으면 softmax 뒤 확률이 0 — 그래서 '다음'을 진짜로 맞히게 된다",
           size=12, color="595959")


def draw_gepa_loop(slide):
    """GEPA 최적화 루프 — 프롬프트를 실행·채점하고, 피드백을 읽어 스스로 고쳐 쓴다 (반복)."""
    cy = 19.4
    steps = [
        (11.0, "프롬프트\n(초안)", "EDEDED", "C9C9C9", "1A1A1A"),
        (19.2, "실행\n결과 생성", "EAF2FB", "AEBECE", "1A1A1A"),
        (27.4, "채점\n(metric)", "FBEED0", "E0B84E", "7A5A10"),
        (35.6, "피드백 읽고\n프롬프트 수정", "D6E8FF", "378ADD", "134A8E"),
    ]
    W = 7.0
    cxs = []
    for l, txt, fill, line, tc in steps:
        _box(slide, l, cy - 1.3, W, 2.6, txt, fill=fill, line=line, size=13, tcolor=tc)
        cxs.append((l, l + W))
    for i in range(3):
        _line(slide, cxs[i][1], cy, cxs[i + 1][0], cy, color="5B6B7B", width_pt=1.6)
    # 되먹임 화살표: 수정 → 다시 프롬프트 (아래로 돌아온다)
    back_y = cy + 2.6
    _line(slide, cxs[3][0] + W / 2, cy + 1.3, cxs[3][0] + W / 2, back_y, color="378ADD", width_pt=1.6, arrow=False)
    _line(slide, cxs[3][0] + W / 2, back_y, cxs[0][0] + W / 2, back_y, color="378ADD", width_pt=1.6, arrow=False)
    _line(slide, cxs[0][0] + W / 2, back_y, cxs[0][0] + W / 2, cy + 1.3, color="378ADD", width_pt=1.6)
    _label(slide, 11.0, back_y + 0.2, 32.0, "점수가 오를 때까지 반복 — '프롬프트를 스스로 고쳐 쓴다'",
           size=12, color="378ADD", bold=True)
    _label(slide, 11.0, cy - 2.6, 38.0,
           "사람이 프롬프트를 손보는 대신 — 채점 결과(피드백)를 모델이 읽고 다음 프롬프트를 쓴다",
           size=13, color="1F4E79", bold=True)
    _label(slide, 11.0, back_y + 1.4, 38.0,
           "학습(가중치 변경) 없이 프롬프트만 바꿔 val 0.80 → 1.00 (L40S 실측)",
           size=12, color="595959")


def draw_gepa_principle(slide):
    """RL vs GEPA — 배우는 신호의 차이. RL 은 숫자 하나, GEPA 는 '왜 틀렸는지' 문장."""
    # --- RL 행 ---
    yR, hR = 16.2, 2.2
    _box(slide, 10.4, yR + 0.1, 3.2, hR, "RL\n(GRPO)", fill="F0EDE6", line="D8CFBE",
         size=12, bold=True, tcolor="7A6A3A")
    r = [(14.4, 5.8, "롤아웃\n(응답 생성)", "EDEDED", "C9C9C9", "1A1A1A"),
         (22.0, 6.8, "보상 = 0.5", "FBEED0", "E0B84E", "7A5A10"),
         (31.2, 6.8, "가중치를\n경사로 수정", "ECECEC", "B8B8B8", "5A5A5A")]
    for l, w, t, f, ln, tc in r:
        _box(slide, l, yR, w, hR, t, fill=f, line=ln, size=13, tcolor=tc)
    for i in range(len(r) - 1):
        _line(slide, r[i][0] + r[i][1], yR + hR / 2, r[i + 1][0], yR + hR / 2,
              color="9A8A5A", width_pt=1.6)
    _label(slide, 14.4, yR + hR + 0.15, 30.0,
           "신호 = 숫자 하나 — '무엇이 왜 틀렸는지'는 알 수 없다",
           size=12, color="7A6A3A", align=PP_ALIGN.LEFT)

    # --- GEPA 행 ---
    yG, hG = 20.4, 2.4
    _box(slide, 10.4, yG + 0.15, 3.2, hG, "GEPA", fill="D6E8FF", line="378ADD",
         size=13, bold=True, tcolor="134A8E")
    g = [(14.4, 5.8, "롤아웃\n(응답 생성)", "EDEDED", "C9C9C9", "1A1A1A"),
         (21.6, 9.0, "feedback\n'category 를 대분류로'", "D6E8FF", "378ADD", "134A8E"),
         (32.4, 6.6, "reflection LM\n지시문 재작성", "EAF2FB", "AEBECE", "1A1A1A"),
         (40.6, 6.6, "Pareto\n후보 선택", "DDEAF6", "8FB4D8", "1F4E79")]
    for l, w, t, f, ln, tc in g:
        _box(slide, l, yG, w, hG, t, fill=f, line=ln, size=13, tcolor=tc)
    for i in range(len(g) - 1):
        _line(slide, g[i][0] + g[i][1], yG + hG / 2, g[i + 1][0], yG + hG / 2,
              color="5B6B7B", width_pt=1.6)
    _label(slide, 14.4, yG + hG + 0.15, 34.0,
           "신호 = 왜 틀렸는지(글) — 가중치는 그대로, 프롬프트만 고친다",
           size=12, color="1F4E79", align=PP_ALIGN.LEFT, bold=True)

    _label(slide, 10.4, 24.9, 39.0,
           "같은 목표에 GEPA 는 최대 ~35× 적은 시도 — 숫자 대신 '이유'를 읽기 때문",
           size=13, color="378ADD", bold=True)


def draw_minhash(slide):
    """MinHash 직관 — 실제 조각·번호로. 각 문서의 '가장 작은 번호'가 공통 조각이면 일치."""
    # 조각에 해시로 번호를 매긴다. 나·다 는 두 문서 공통(초록).
    A = [("가", 7, False), ("나", 2, True), ("다", 5, True), ("라", 9, False)]
    B = [("나", 2, True), ("다", 5, True), ("마", 4, False), ("바", 8, False)]
    cw, gap, lx = 3.8, 0.5, 15.4
    yA, yB = 15.4, 17.9

    def row(y, cells, label):
        # 문서임을 분명히 — 태그 상자로
        _box(slide, 10.2, y + 0.15, 4.4, 1.8, label, fill="1F4E79", line=None,
             size=14, bold=True, tcolor="FFFFFF")
        xs = []
        for i, (sh, num, common) in enumerate(cells):
            x = lx + i * (cw + gap)
            xs.append(x + cw / 2)
            is_min = (num == min(n for _, n, _ in cells))
            fill, line, tc = (("D9F0DF", "3E9E5A", "1A5A2E") if common
                              else ("EFEFEF", "C9C9C9", "6B6B6B"))
            b = _kq_card(slide, x, y, cw, 2.1, fill=fill,
                         line=("378ADD" if is_min else line))
            if is_min:
                b.line.width = Pt(2.5)
            b.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = b.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            _kq_run(p, sh, size=15, bold=True, color=tc)
            p2 = b.text_frame.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            _kq_run(p2, "해시 " + str(num), size=11, color="8A8A8A")
        return xs

    _label(slide, 10.4, yA - 1.2, 39.0,
           "각 조각에 해시로 번호를 매긴다 — 문서마다 '가장 작은 번호'의 조각을 고른 것이 MinHash",
           size=12, color="595959")
    xsA = row(yA, A, "문서 A")
    xsB = row(yB, B, "문서 B")
    # 두 문서의 최소(나·2)를 잇는다 — 같은 조각이라 일치
    _line(slide, xsA[1], yA + 2.1, xsB[0], yB, color="378ADD", width_pt=1.8)
    _label(slide, 10.4, yB + 2.3, 39.0,
           "각 문서의 최소 = 둘 다 '나'(2)  →  MinHash 일치", size=13, color="1A5A2E", bold=True)

    # ── 유사도(자카드)를 눈에 보이게: 합집합 6칸 중 겹침(초록) 2칸 ──
    uy = yB + 4.0
    union = [("가", False), ("나", True), ("다", True), ("라", False), ("마", False), ("바", False)]
    ucw, ux0 = 2.3, 22.6
    _label(slide, 10.4, uy + 0.1, 12.0, "유사도(자카드) =", size=13, color="1F4E79",
           bold=True, align=PP_ALIGN.LEFT)
    for i, (sh, common) in enumerate(union):
        col = ("D9F0DF", "3E9E5A", "1A5A2E") if common else ("EFEFEF", "C9C9C9", "6B6B6B")
        _box(slide, ux0 + i * (ucw + 0.25), uy, ucw, 1.5, sh, fill=col[0], line=col[1],
             size=13, tcolor=col[2])
    _label(slide, ux0 + 6 * (ucw + 0.25) + 0.3, uy + 0.1, 11.0,
           "= 겹침 2 / 전체 6 ≈ 0.33", size=13, color="1F4E79", bold=True, align=PP_ALIGN.LEFT)
    _label(slide, 10.4, uy + 2.0, 39.0,
           "무작위 번호라, 최소가 겹침(초록)에 떨어질 확률이 곧 이 유사도 — MinHash 일치율로 유사도를 잰다",
           size=12, color="595959")


def draw_data_pipeline(slide):
    """데이터 정제 파이프라인 — 원본 → 중복제거 → 품질필터 → 정제본."""
    cy = 21.6
    steps = [
        (10.8, 6.4, "원본 위키\n5,000건", "EDEDED", "C9C9C9", "1A1A1A"),
        (18.4, 7.2, "정확 중복 제거\n(해시)", "EAF2FB", "AEBECE", "1A1A1A"),
        (27.0, 7.8, "근사 중복 제거\n(MinHash·LSH)", "EAF2FB", "AEBECE", "1A1A1A"),
        (36.2, 6.6, "품질 필터\n(휴리스틱)", "FBEED0", "E0B84E", "7A5A10"),
    ]
    right = []
    for l, w, txt, fill, line, tc in steps:
        _box(slide, l, cy - 1.2, w, 2.4, txt, fill=fill, line=line, size=13, tcolor=tc)
        right.append(l + w)
    for i in range(3):
        _line(slide, right[i], cy, steps[i + 1][0], cy, color="5B6B7B", width_pt=1.6)
    _box(slide, 43.6, cy - 1.2, 5.8, 2.4, "정제본\n3,524건", fill="D9F0DF", line="3E9E5A",
         size=13, tcolor="1A5A2E")
    _line(slide, right[3], cy, 43.6, cy, color="5B6B7B", width_pt=1.6)
    _label(slide, 10.8, cy - 3.0, 38.6,
           "숫자만 보지 말고 실물을 열어 본다 — 이 실습의 반복 원칙", size=12, color="1F4E79", bold=True)


def draw_cpt_tradeoff(slide):
    """CPT replay 트레이드오프 — replay↑ 이면 동화 회복(좋음)·위키 악화(나쁨)."""
    cols = [("replay 0%", "37.3", "53.5"), ("replay 5%", "22.5", "54.8"),
            ("replay 25%", "20.5", "59.7")]
    x0, dx, top = 15.0, 8.6, 16.6
    _label(slide, 10.6, top - 0.2, 3.8, "동화 PPL\n(회복 = 좋다)", size=12, color="1A5A2E",
           align=PP_ALIGN.LEFT)
    _label(slide, 10.6, top + 4.6, 3.8, "위키 PPL\n(상승 = 대가)", size=12, color="C65B77",
           align=PP_ALIGN.LEFT)
    prev_story = prev_wiki = None
    for i, (name, story, wiki) in enumerate(cols):
        cx = x0 + i * dx
        _box(slide, cx, top - 0.8, 6.4, 1.4, name, fill="EFEFEF", line="C9C9C9", size=13)
        _box(slide, cx, top + 1.4, 6.4, 1.7, "동화 " + story, fill="D9F0DF", line="3E9E5A",
             size=14, bold=True, tcolor="1A5A2E")
        _box(slide, cx, top + 5.0, 6.4, 1.7, "위키 " + wiki, fill="FAD9E0", line="C65B77",
             size=14, bold=True, tcolor="7A2A3E")
        if prev_story is not None:
            _line(slide, cx - dx + 6.4, top + 2.25, cx, top + 2.25, color="3E9E5A",
                  width_pt=1.6)
            _line(slide, cx - dx + 6.4, top + 5.85, cx, top + 5.85, color="C65B77",
                  width_pt=1.6)
        prev_story, prev_wiki = story, wiki
    _label(slide, 10.6, top + 7.4, 38.6,
           "replay 를 조금 섞으면 동화가 크게 돌아온다(37→22) — 대신 위키가 조금씩 나빠진다. 공짜가 아니다",
           size=12, color="595959")


def draw_amazon_pipeline(slide):
    """Amazon 6단계 LLM 파이프라인 — 여러 호출을 엮어 상품 요약을 만든다 (스네이크)."""
    r1y, r2y = 16.6, 21.4
    top = [("① 요소 유형", "무슨 정보가 있나"), ("② 소분류", "요소를 묶는다"),
           ("③ 값 추출", "각 요소의 값")]
    bot = [("④ 소비자 유형", "누구를 위한 상품"), ("⑤ 소분류 요약", "묶음별 한 줄"),
           ("⑥ 통합 요약", "소비자별 최종")]
    xs = [11.0, 24.0, 37.0]
    W = 11.4
    for i, (t, s) in enumerate(top):
        _box(slide, xs[i], r1y, W, 2.6, t + "\n" + s, fill="EAF2FB", line="378ADD",
             size=13, tcolor="134A8E")
        if i < 2:
            _line(slide, xs[i] + W, r1y + 1.3, xs[i + 1], r1y + 1.3, color="5B6B7B", width_pt=1.6)
    # 오른쪽에서 아래로 꺾는다
    _line(slide, xs[2] + W / 2, r1y + 2.6, xs[2] + W / 2, r2y, color="5B6B7B", width_pt=1.6)
    for i, (t, s) in enumerate(bot):
        j = 2 - i    # 오른쪽→왼쪽
        _box(slide, xs[j], r2y, W, 2.6, t + "\n" + s, fill="D9F0DF", line="3E9E5A",
             size=13, tcolor="1A5A2E")
        if i < 2:
            _line(slide, xs[j], r2y + 1.3, xs[j - 1] + W, r2y + 1.3, color="5B6B7B", width_pt=1.6)
    _label(slide, 11.0, r1y - 1.4, 37.8,
           "한 번의 호출이 아니라 — 작은 LLM 호출 여섯 개를 파이프라인으로 엮는다",
           size=13, color="1F4E79", bold=True)
    _label(slide, 11.0, r2y + 2.9, 37.8,
           "각 단계가 구조화 출력(JSON) — 다음 단계의 입력이 된다", size=12, color="595959")


# ---------------------------------------------------------------------------
# 코드 슬라이드 — 신규 장에 monospace 코드 박스를 그린다 (2026-09-03)
#
# 1일차·Amazon 은 원본 덱의 코드 박스를 물려받지만, 2일차 신규 실습
# (데이터처리·미니GPT·프롬프트최적화)은 원본에 대응 장이 없어 직접 그린다.
# 도너(1일차 p19)에는 코드 박스가 없으므로, 본문(제목·인트로) 아래에
# 코드 카드를 새 도형으로 얹는다. fill_new_slide 가 spec["code"] 를 보고 호출한다.
# ---------------------------------------------------------------------------
CODE_FONT = "Consolas"           # Windows 기본 monospace — 들여쓰기가 유지된다


def _code(src: str) -> str:
    """소스에 편하게 들여써 둔 코드 리터럴에서 공통 들여쓰기를 벗기고 앞뒤 개행 정리."""
    return textwrap.dedent(src).strip("\n")


def draw_code_box(slide, code: str, *, top_cm: float, left_cm: float = 10.6,
                  width_cm: float = 39.2, size: int = 15) -> None:
    lines = code.split("\n")
    line_cm = size * 0.047                     # 대략의 줄 간격 (여유 있게)
    h = line_cm * len(lines) + 0.5
    card = _kq_card(slide, left_cm, top_cm, width_cm, h,
                    fill="F5F7FA", line="D0D7DE", radius=0.02)
    tf = card.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.TOP
    tf.margin_left = Cm(0.45)
    tf.margin_right = Cm(0.2)
    tf.margin_top = Cm(0.18)
    tf.margin_bottom = Cm(0.1)
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.0
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        comment = ln.lstrip().startswith("#")
        r = p.add_run()
        r.text = ln if ln.strip() else " "
        r.font.name = CODE_FONT
        r.font.size = Pt(size)
        r.font.color.rgb = RGBColor.from_string("5A8250" if comment else "1A1A1A")
        rPr = r._r.get_or_add_rPr()
        rPr.set(qn("a:lang"), "en-US")
        ea = etree.SubElement(rPr, qn("a:ea"))
        ea.set("typeface", TABLE_FONT)          # 한글 주석은 맑은 고딕으로
        t_el = r._r.find(qn("a:t"))
        t_el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


_JUDGE_SRC = r'''from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

SCHEMA = {"type": "object",
  "properties": {
    "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
    "relevance":   {"type": "integer", "minimum": 1, "maximum": 5},
    "fluency":     {"type": "integer", "minimum": 1, "maximum": 5},
    "reason":      {"type": "string",  "maxLength": 150}},  # 없으면 끝없이 생성(90초 실측)
  "required": ["correctness", "relevance", "fluency", "reason"]}

def judge(question, reference, prediction):
    r = client.chat.completions.create(
        model="Qwen/Qwen3-4B-Instruct-2507",
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(...)}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "score", "schema": SCHEMA}},
        temperature=0.0,   # 채점은 재현 가능해야 한다
        max_tokens=200)    # 하드캡 — 스키마만 믿으면 안 된다
    return json.loads(r.choices[0].message.content)'''


def draw_judge_code(slide):
    """평가 judge() 코드 — response_format(json_schema)+temperature=0+max_tokens 안전장치."""
    draw_code_box(slide, _JUDGE_SRC, top_cm=11.0, size=12)


# SFT build_messages — 노트북 셀8 은 KULLM(Alpaca) 의 input 컬럼을 합친다.
# 원본 슬라이드(2일차 p33)는 instruction/output 만 써서 폴백 경로에서 대상이 빠진다.
_BUILD_MSG_SRC = r'''def build_messages(example):
    # KULLM 은 Alpaca 계열이라 input 컬럼이 있다 — 합치지 않으면 대상이 빠진다
    user = example["instruction"]
    if example.get("input"):
        user = f'{user}\n\n{example["input"]}'
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": example["output"]}
    ]
    return dict(messages=messages)

raw_datasets = raw_datasets.map(build_messages)'''


def sync_build_messages(slide):
    _set_code_box(slide, "def build_messages", _BUILD_MSG_SRC)


def add_dpo_template_warning(slide):
    """DPO 손조립(2일차 p58) 설명 상자에 '챗 템플릿 어긋남' 경고를 덧붙인다 (노트북 셀11)."""
    box = next((sh for sh in slide.shapes
                if sh.has_text_frame and "SFT 학습과는 다르게" in sh.text_frame.text), None)
    if box is None:
        raise SystemExit("[중단] DPO 챗템플릿 설명 상자 못 찾음")
    tf = box.text_frame
    tmpl = _para_templates(tf)
    src = tmpl[min(tmpl)]
    tf._txBody.append(_make_para(
        src, "주의: 손조립 표기가 위 토크나이저 챗 템플릿과 어긋나면 학습은 끝나는데 답이 "
             "이상해진다 — 두 곳의 <|user|> 표기가 같은지 확인"))
    # 경고 2줄이 들어갈 자리를 만들려 코드박스를 아래로 민다
    code = next((sh for sh in slide.shapes
                 if sh.has_text_frame and "def chatml_format" in sh.text_frame.text), None)
    if code is not None:
        code.top = (code.top or 0) + Cm(2.0)


EXTRAS: list[SlideExtra] = [
    SlideExtra(
        deck="3일차", page=18,
        why="Amazon 개요에 6단계 LLM 파이프라인 도해 — 단계가 장마다 흩어져 전체가 안 보인다 (2026-09-01)",
        build_fn=draw_amazon_pipeline,
        expect_texts=("요소 유형", "통합 요약", "파이프라인으로 엮는다"),
    ),
    # SFT build_messages 코드박스를 노트북 셀8(입력 컬럼 합치기)로 동기화 (2026-09-03 검토)
    SlideExtra(
        deck="2일차", page=33,
        why="build_messages 가 KULLM input 컬럼을 안 합쳐 폴백 시 대상이 빠진다 — 노트북 셀8과 동기화",
        build_fn=sync_build_messages,
        expect_texts=('if example.get("input"):', '"content": user'),
    ),
    # DPO 손조립 챗 템플릿 어긋남 경고 추가 (노트북 셀11, 2026-09-03 검토)
    SlideExtra(
        deck="2일차", page=58,
        why="손조립 템플릿이 토크나이저 템플릿과 어긋나면 조용한 실패 — 노트북 셀11 경고가 슬라이드에 없다",
        build_fn=add_dpo_template_warning,
        expect_texts=("어긋나면 학습은 끝나는데",),
    ),
    # Amazon 코드박스를 노트북 셀로 통째 교체 (완전 동기화, 2026-09-01)
    _amz_extra(25, "FeatureType",
               sync_amazon_pydantic("class FeatureType", "class FeatureType", _IMP),
               expect=("Field(max_length=40)", "Field(max_length=200)")),
    _amz_extra(28, "gen_text_step_1",
               sync_amazon_gen("def gen_text_step_1", "gen_text_step_1"),
               expect=("temperature=0.2", "max_tokens=2048", 'finish_reason == "length"')),
    _amz_extra(32, "Subsection",
               sync_amazon_pydantic("class Subsection", "class Subsection", _IMP_ANN),
               expect=("Annotated[str, Field(max_length=60)]", "Field(max_length=10)")),
    _amz_extra(33, "gen_text_step_2",
               sync_amazon_gen("def gen_text_step_2", "gen_text_step_2"),
               expect=("temperature=0.2", 'finish_reason == "length"')),
    # p36·39·45·49 는 class+gen 이 한 박스라 셀 전체로 교체 (원본 실측)
    _amz_extra(36, "ExtractedFeature + gen_text_step_3",
               sync_amazon_full("class ExtractedFeature", "class ExtractedFeature", _IMP),
               expect=("Field(max_length=30)", 'finish_reason == "length"')),
    _amz_extra(39, "ConsumerCategory + gen_text_step_4",
               sync_amazon_full("class ConsumerCategory", "class ConsumerCategory", _IMP),
               expect=("Field(max_length=6)", 'finish_reason == "length"')),
    _amz_extra(45, "SummaryDescription(400) + gen_text_summary",
               sync_amazon_full("class SummaryDescription",
                                "summary: str = Field(max_length=400)", _IMP),
               expect=("Field(max_length=400)", 'finish_reason == "length"')),
    _amz_extra(49, "SummaryDescription(1000) + gen_text_featured_summary",
               sync_amazon_full("class SummaryDescription",
                                "summary: str = Field(max_length=1000)", _IMP),
               expect=("Field(max_length=1000)", 'finish_reason == "length"')),
    SlideExtra(
        deck="3일차", page=55,
        why="RAG 개요에 파이프라인 도해 — 질문→BM25→문서→LLM→답 (2026-09-01)",
        build_fn=draw_rag_pipeline,
        expect_texts=("BM25 검색", "top-3 문서", "지식은 검색으로 넣는다"),
    ),
    SlideExtra(
        deck="3일차", page=69,
        why="SubQuestionQueryEngine 기본 질문생성기는 OpenAI 전용 — 로컬 vLLM 용 question_gen 을 노트북과 맞춰 추가 (import 포함)",
        build_fn=add_subq_question_gen,
        expect_texts=("from llama_index.core.question_gen import LLMQuestionGenerator",
                      "question_gen=LLMQuestionGenerator.from_defaults(llm=Settings.llm),"),
    ),
    SlideExtra(
        deck="1일차", page=35,
        why="Classification 학습 코드를 노트북과 맞추면서 클리핑 방지",
        build_fn=compact_training_code_slide,
        expect_texts=('report_to="tensorboard"', 'history = trainer.train()'),
    ),
    SlideExtra(
        deck="1일차", page=46,
        why="NER 학습 코드를 노트북과 맞추면서 클리핑 방지",
        build_fn=compact_training_code_slide,
        expect_texts=('report_to="tensorboard"', 'history = trainer.train()'),
    ),
    # p58(KorQuAD)은 리더보드 캡처로만 태스크를 설명한다 — "정답이 지문 속 구간"이라는
    # 핵심이 안 보인다. 원본 4줄은 그대로 두고, 낡은 리더보드 그림을 빼고 그 자리에
    # 지문+정답 하이라이트 예제를 얹는다 (2026-09-01 사용자 지시 — 그림 빼고 예제만).
    SlideExtra(
        deck="1일차", page=58,
        why="KorQuAD 태스크를 순위표 대신 실제 데이터 예제로 — 원본 유지, 그림 제거, 정답 하이라이트",
        build_fn=build_korquad_example,
        expect_texts=("교향곡", "1악장", "베토벤의 교향곡 9번",
                      "무엇을 쓰고자 했는가?", "파우스트를 읽고"),
    ),
]

# ---------------------------------------------------------------------------
# 신규 슬라이드 레지스트리 — key → 장 목록
#   각 장: dict(header=우상단 헤더, title=소제목, bullets=[(level, text), ...])
#   Phase F 에서 원고로 대체한다. 지금은 파이프라인 검증용 스텁.
# ---------------------------------------------------------------------------
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
    "rag_bm25": [
        {"header": '1. Llama Index를 활용한 RAG 실습 – 위키피디아 Q&A 실습',
         "title": 'BM25 — 임베딩 없이 찾는다',
         "bullets": [
            (0, 'BM25 는 단어가 겹치는 정도로 문서를 고른다 — 의미를 이해하지 않는다'),
            (1, '강한 곳: 고유명사 · 전문용어 · 숫자   /   약한 곳: 바꿔 말한 질문'),
            (1, '임베딩 검색과 반대 — 실무에서는 둘을 섞는다 (하이브리드)'),
            (0, '그래서 Settings.embed_model = None — 임베딩을 일부러 끈다'),
            (0, '청킹: 너무 크면 검색이 뭉뚝하고, 너무 잘면 문맥이 끊긴다'),
            (1, 'chunk_size=512 는 타협값 — RAG 품질이 안 나오면 가장 먼저 손보는 곳'),
        ]},
    ],
    "rag_prompt": [
        {"header": '1. Llama Index를 활용한 RAG 실습 – 위키피디아 Q&A 실습',
         "title": '프롬프트 교체의 함정 · 실패 진단',
         "bullets": [
            (0, '기본 프롬프트는 영어 — 한국어로 물어도 영어로 답하기 쉽다'),
            (0, '함정: get_prompts() 는 deepcopy 를 돌려준다'),
            (1, '받은 객체의 template 을 고쳐도 에러 없이 무시된다 → update_prompts() 를 쓴다'),
            (1, '서브질문 생성 프롬프트도 영어 — 영어 서브질문은 한국어 위키에서 아무것도 못 찾는다'),
            (0, '답만 보지 말고 source_nodes(근거 문서)를 본다'),
            (1, '근거가 무관 → 검색 실패  /  근거는 맞는데 답이 틀림 → 생성 실패'),
            (0, './bm25_retriever 인덱스를 지우지 말 것 — 하루의 끝에 다시 씁니다'),
        ]},
    ],
    "evalsec_a": [
        {"header": '2. 태스크 정의와 평가',
         "title": '평가는 태스크를 적는 것에서 시작한다',
         "bullets": [
            (0, '지표를 고르기 전에 네 가지를 적는다:'),
            (1, '입력 · 출력 · 성공 기준 · 실패 사례'),
            (1, '예: 한국어 질문 → 한국어 답 / 사실이 맞고 질문에 답할 것 / 영어로 답함·회피'),
            (0, '성공 기준이 여러 개면 지표도 여러 개 — 하나로 뭉뚱그리면 무엇이 나빠졌는지 모른다'),
            (0, '이 확인 수단을 학습(SFT·DPO·GRPO)보다 먼저 갖춘다 — 오늘 이 순서의 이유'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": '평가 데이터 — 어떻게 만들고, 학습과 어떻게 분리하나',
         "bullets": [
            (0, '평가셋은 학습에 쓰지 않은 데이터여야 한다 — 배운 걸 다시 물으면 점수가 부풀려진다(암기 착시)'),
            (1, "1일차 '데이터 누수'의 평가판 — 같은 출처·같은 예시가 학습과 평가 양쪽에 들어가면 안 된다"),
            (0, '구축: 실제 분포를 대표하도록 표본 · 각 예시에 정답(reference)을 붙임 · 성공·실패 사례를 고루'),
            (0, '분리: train(학습) / validation(튜닝하며 여러 번) / test(마지막에 한 번만)'),
            (1, 'test 를 보며 튜닝하면 test 도 오염된다 — 그래서 마지막 한 번만 연다'),
            (0, '이 실습에서 SFT·DPO·GRPO 가 train 과 test 를 나눈 이유가 바로 이것'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": '평가셋을 실무에서 운영하는 법',
         "bullets": [
            (0, '골든셋을 고정하고 버전을 매긴다 — 시점 간 점수를 비교하려면 기준이 안 움직여야 한다'),
            (0, '평가셋을 키운다: 운영 중 발견한 실패 사례를 되먹인다 (같은 실수의 재발 방지)'),
            (0, '학습 파이프라인과 물리적으로 분리 — 평가셋이 실수로 학습에 흘러들지 않게'),
            (0, '크기보다 대표성 — 수십 건으로 시작해 늘린다 (실습은 소규모로 감을 잡는다)'),
            (0, '분포가 바뀌면(신제품·신주제) 평가셋도 갱신 — 한 지표에 올인하지 말고 조합해서 본다'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": '자동 지표 — BLEU 와 ROUGE',
         "bullets": [
            (0, '둘 다 모델 응답과 정답이 표면적으로 얼마나 겹치는가를 잰다'),
            (1, 'BLEU: 응답의 n-gram 중 정답에도 있는 비율 (정밀도 중심 · 번역)'),
            (1, 'ROUGE: 정답의 n-gram 중 응답에도 있는 비율 (재현율 중심 · 요약)'),
            (1, 'ROUGE-1 단어 / ROUGE-2 두 단어 연속 / ROUGE-L 최장 공통 부분열'),
            (1, '코드: evaluate.load("sacrebleu") · evaluate.load("rouge") 로 계산'),
            (0, '빠르고 싸고 재현 가능 — 그런데 다음 장의 한계가 있다'),
        ]},
    ],
    "evalsec_b": [
        {"header": '2. 태스크 정의와 평가',
         "title": '자동 지표의 한계 — 실측으로 확인',
         "bullets": [
            (0, '표면 일치만 재기 때문에 두 방향으로 속는다 (실습에서 직접 측정):'),
            (1, '뜻이 같아도 표현이 다르면 점수 하락 — "서울이 대한민국의 수도입니다"'),
            (1, '틀린 답이 높은 점수 — "대한민국의 수도는 부산입니다" (한 단어만 다르니까)'),
            (0, '의미와 사실 여부는 못 잰다 — 열린 생성에서 유일한 기준으로 삼으면 위험'),
            (0, '번역·요약처럼 정답 형태가 정해진 작업에는 여전히 유용하다'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": 'LLM-as-judge — 모델에게 채점을 맡긴다',
         "bullets": [
            (0, '사람 채점(정확·비쌈)과 자동 지표(싸고 얕음)의 중간'),
            (0, '핵심은 채점 기준을 명확히 주는 것 — 축을 나눠 지정한다'),
            (1, 'correctness(사실) · relevance(질문에 답했나) · fluency(자연스러운가)'),
            (0, '출력은 json_schema 로 강제 — 점수를 프로그램이 읽어야 하므로'),
            (1, '문자열 필드에 maxLength 상한 필수 — 없으면 끝없이 생성한다 (90초 실측)'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": 'judge 코드 — 채점을 강제하고 길이를 막는다',
         "bullets": [
            (0, 'json_schema 로 점수 형식 강제 · temperature=0 으로 재현 · max_tokens 로 길이 상한'),
            (1, 'maxLength + max_tokens 이중 방어 · 잘리면 finish_reason=="length" 로 잡는다'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": 'judge 의 함정',
         "bullets": [
            (0, '재현성: temperature=0 으로 고정해야 같은 답에 같은 점수'),
            (0, '그래도 점수는 흔들린다 — 거르는 용도는 충분, 최적화 목적함수로는 위험'),
            (0, 'self-preference: 자기가 만든 답을 후하게 주는 편향'),
            (1, '평가 대상 모델과 채점 모델을 다르게 하는 편이 안전'),
            (0, '비용: 자동 지표는 즉시, judge 는 호출마다 시간과 돈'),
        ]},
        {"header": '2. 태스크 정의와 평가',
         "title": '무엇을 언제 쓰나',
         "bullets": [
            (0, '번역·요약 (정답 형태 고정)  →  BLEU · ROUGE'),
            (0, '분류·추출 (정답 하나)  →  정확도 · F1'),
            (0, '열린 생성 (챗봇 · QA)  →  LLM-as-judge + 자동 지표 보조'),
            (0, '최종 출시 판단  →  사람 평가 (표본이라도)'),
            (1, '실무는 섞어 쓴다: 개발 중 자동 지표 → 릴리즈 전 judge → 최종 사람'),
            (0, '이제 이 뒤의 SFT·DPO·GRPO 가 정말 나아졌는지 말할 수단이 생겼다'),
        ]},
    ],
    "fewshot_extra": [
        {"header": '3. 퓨샷러닝에 대해서 – vLLM을 활용한 퓨샷러닝 실습해보기',
         "title": '출력이 제멋대로다 — 형식을 잡는 3단계',
         "bullets": [
            (0, "형식을 말해주지 않으면 '긍정' / '이 리뷰는 긍정적입니다' / 이유 설명이 섞여 나온다"),
            (1, '모델이 틀린 게 아니라 우리가 형식을 안 정해준 것'),
            (0, '강제력의 3단계:'),
            (1, '① 프롬프트로 부탁한다 — 대체로 따르지만 보장 없음'),
            (1, '② seed 로 재현성 확보 — 흔들림만 줄인다'),
            (1, '③ 제약 디코딩(structured_outputs) — 다른 답이 나올 수 없다 (문법으로 차단)'),
        ]},
        {"header": '3. 퓨샷러닝에 대해서 – vLLM을 활용한 퓨샷러닝 실습해보기',
         "title": '답이 흔들리는 폭을 잰다',
         "bullets": [
            (0, '같은 질문을 n=10 으로 여러 번 — 보려는 것은 답이 아니라 흔들림의 폭'),
            (1, 'temperature 낮음 = 보수적 · 높음 = 과감  /  top_p 로 후보 범위 제한'),
            (0, '같은 입력에 답이 매번 다르면 그 프롬프트의 결과는 신뢰하기 어렵다'),
            (1, '분류처럼 답이 정해진 일에는 temperature 를 낮게'),
            (0, 'judge 점수에 노이즈가 있다던 프롬프트 최적화의 그 얘기와 같은 뿌리'),
        ]},
        {"header": '3. 퓨샷러닝에 대해서 – vLLM을 활용한 퓨샷러닝 실습해보기',
         "title": '결론은 숫자 둘의 비교다',
         "bullets": [
            (0, '같은 평가 루프를 zero-shot / few-shot 으로 두 번 돌려 정확도를 비교한다'),
            (1, '10건은 통계적으로 무의미 — 실습이라 작게, 실무는 수백 건'),
            (0, 'few-shot 이 높다 → 예시가 도움됐다'),
            (0, '비슷하다 → 이 작업에는 지시만으로 충분'),
            (0, '낮다 → 예시가 편향됐거나 문제를 헷갈리게 했다 (라벨 균형을 확인했는가?)'),
            (1, 'few-shot 이 항상 낫지는 않다 — 재봐야 안다'),
        ]},
        {"header": '3. 퓨샷러닝에 대해서 – vLLM을 활용한 퓨샷러닝 실습해보기',
         "title": '정보 추출 — 스키마로 강제한다 (KorQuAD)',
         "bullets": [
            (0, '분류는 choice 로 막았지만, 추출은 답의 개수·내용을 미리 모른다'),
            (0, '3단계 비교가 요점:'),
            (1, '① 그냥 묻는다 → 자유 문장, 파싱 불가'),
            (1, "② 프롬프트에 JSON 예시 → 대체로 되지만 가끔 깨진다 — 그 '가끔'이 운영 사고"),
            (1, '③ pydantic 스키마 + response_format → 항상 그 구조'),
            (0, '스키마 문자열 필드에 길이 상한(maxLength) — 없으면 잘려서 JSON 이 깨진다'),
        ]},
    ],
    "sft_data": [
        {"header": '5. 인스트럭션 모델학습 – SFT 모델 학습',
         "title": '어제 만든 데이터로 학습한다 — 877건',
         "bullets": [
            (0, '두 파일을 합쳐 읽는다:'),
            (1, 'assets/amazon_ko_sft.jsonl.gz — 강사 사전생성 (상품 100개 · 약 800건)'),
            (1, 'data/amazon_ko_sft.mine.jsonl — 어제 여러분이 직접 만든 것 (상품 10개)'),
            (0, '(instruction, output) 완전 중복만 제거 — 생성이 비결정적이라 거의 안 겹친다'),
            (0, '파일이 없으면 공개 데이터셋(KULLM)으로 폴백하되 크게 알린다'),
            (1, "폴백인 줄 모르고 '내 데이터로 학습했다' 고 오해하면 안 되니까"),
        ]},
    ],
    "sft_tmpl": [
        {"header": '5. 인스트럭션 모델학습 – SFT 모델 학습',
         "title": '-Base 모델에는 대화 형식이 없다',
         "bullets": [
            (0, '사전학습만 한 모델은 <|user|> 같은 대화 표시를 모른다 — 우리가 정해서 넣는다'),
            (1, '역할을 특수 문자열로 감싸는 커스텀 챗 템플릿 (Jinja 한 줄)'),
            (0, '-Instruct 모델은 이미 템플릿이 있다 — 덮어쓰면 오히려 성능이 떨어진다'),
            (0, '원칙: 학습할 때와 쓸 때의 형식이 같아야 한다'),
            (1, '어긋나면 학습은 정상 종료되는데 답이 이상한, 조용한 실패가 된다'),
            (1, '이 원칙이 SFT → DPO → GRPO 를 관통한다'),
        ]},
    ],
    "sft_len": [
        {"header": '5. 인스트럭션 모델학습 – SFT 모델 학습',
         "title": '학습 전 토큰 길이 확인 — 조용한 실패',
         "bullets": [
            (0, 'max_length 를 넘는 예시는 뒤가 잘린다 — 뒤 = 정답(assistant 답변)'),
            (0, '정답이 잘려나가도 에러가 없고 loss 는 오히려 낮게 나온다'),
            (1, '학습은 완주했는데 아무것도 못 배운 상태 — 써보기 전엔 모른다'),
            (0, '그래서 학습 전에 토큰 길이 분포를 반드시 찍어 본다'),
            (1, '상한 초과 비율이 몇 %인지, 잘리면 무엇이 잘리는지'),
        ]},
    ],
    "dpo_resume": [
        {"header": '6. 선호기반 모델학습 – DPO 모델 학습',
         "title": 'SFT 를 이어받는다 — 그리고 참조 모델이 안 보이는 이유',
         "bullets": [
            (0, 'DPO 는 SFT 완료 모델에서 출발한다 — 베이스에 바로 걸면 잘 안 된다'),
            (1, "PeftModel.from_pretrained(model, 'data/sft_model', is_trainable=True)"),
            (1, 'is_trainable=True 를 빠뜨리면 어댑터가 얼어붙어 학습이 안 되는 함정'),
            (0, '참조 모델이 코드에 없는 이유 — LoRA 라서'),
            (1, '원래 가중치는 얼려 둔 채 어댑터만 학습 → 어댑터를 잠깐 끄면 그게 곧 참조 모델'),
            (1, 'TRL 이 내부에서 처리 — 모델 하나 분량의 메모리를 아낀다'),
        ]},
    ],
    "dpo_logs": [
        {"header": '6. 선호기반 모델학습 – DPO 모델 학습',
         "title": '학습 로그 읽는 법 · ORPO 라는 선택지',
         "bullets": [
            (0, 'rewards/chosen 과 rewards/rejected — 둘의 차이가 벌어지는지 본다'),
            (1, 'rewards/accuracies: chosen 을 더 높게 준 비율 — 올라가야 정상'),
            (0, '학습률은 SFT(5e-4)보다 낮은 3e-4 — 이미 배운 것을 잃지 않으려고'),
            (0, 'ORPO: SFT 없이 선호 쌍만으로 한 단계 (loss = SFT 항 + 승산비 항)'),
            (1, '선호 데이터부터 모인 상황이라면 파이프라인이 짧아진다 — 마무리 가이드에서 비교'),
        ]},
    ],
    "grpo_reward": [
        {"header": '7. 리즈닝 모델학습 – GRPO 모델 학습',
         "title": '채점 함수의 품질이 곧 학습의 품질',
         "bullets": [
            (0, 'GRPO 에는 정답 라벨이 없다 — 답을 채점하는 함수가 그 자리를 차지한다'),
            (1, 'format_reward: <think>…</think><answer>…</answer> 형식을 지켰나'),
            (1, 'accuracy_reward: 답이 수학적으로 같은가 — math_verify 가 1/2 == 0.5 를 같게 본다'),
            (0, '함수가 엉성하면 모델이 그 엉성함을 파고든다 (리워드 해킹)'),
            (1, 'format_reward 하나만 쓰면 형식만 그럴듯하고 답은 틀리는 모델이 된다'),
            (0, "PPO 와의 차이: 보상 '모델'(신경망) 대신 보상 '함수' — 모델 하나를 통째로 아낀다"),
        ]},
    ],
    "grpo_group": [
        {"header": '7. 리즈닝 모델학습 – GRPO 모델 학습',
         "title": 'Group Relative — num_generations 가 정의다',
         "bullets": [
            (0, '문제 하나에 답을 4개 만들고(num_generations=4), 그 4개를 서로 비교한다'),
            (1, '그룹 평균보다 잘한 답은 확률↑, 못한 답은 확률↓ — 그룹 안에서 상대 평가'),
            (1, '절대 점수가 아니라서 별도의 기준선 모델이 필요 없다'),
            (0, '생성이 학습 루프 안에 있다 — GRPO 가 가장 비싼 이유'),
            (0, '로그: format 이 먼저 1.0 에 가고 accuracy 가 천천히 따라온다 (형식이 더 쉬우니까)'),
        ]},
    ],
    "minigpt": [
        {"header": '3. 미니 GPT 만들기 – 실습 준비',
         "title": '밑바닥부터 만든다',
         "bullets": [
            (0, '라이브러리가 주는 완성품 대신 GPT 구조를 직접 쌓고 데이터로 채운다'),
            (0, '약 2백만 파라미터 — 실습 시간 안에 학습이 끝나는 크기'),
            (1, '구조는 대형 모델과 완전히 같다. 크기만 다르다'),
            (0, '전체(사전학습 + CPT 3회)가 약 4분 (L40S 실측 225초)'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 데이터',
         "title": '한국어 동화 (TinyStories)',
         "bullets": [
            (0, '어휘와 문장 구조를 단순하게 설계한 데이터 — 작은 모델도 문법을 익힐 수 있게'),
            (1, '일반 웹 텍스트로 2M 모델을 학습하면 의미 없는 글자만 나온다'),
            (0, '이 선택이 뒤의 이어학습(CPT) 실험의 무대가 된다 — 동화만 아는 모델'),
            (1, '데이터: g0ster/TinyStories-Korean (MIT) — 200자 이상 문서만 15만 개, 5%는 평가용'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 토크나이저',
         "title": '토크나이저를 직접 학습한다 — ByteLevel BPE',
         "bullets": [
            (0, '모델은 글자를 모른다 — 숫자(토큰 ID)만 다룬다. 그 변환표를 데이터에서 만든다'),
            (0, '자주 붙어 나오는 바이트 쌍을 반복해서 합쳐 가는 방식 (BPE)'),
            (0, '왜 ByteLevel 인가: 한국어 음절이 수천 종이라 사전 5,000 으로는 [UNK] 대량 발생'),
            (1, '모든 텍스트를 바이트로 먼저 쪼개므로 모르는 글자가 원천적으로 없다'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 토크나이저',
         "title": '토큰화와 청킹',
         "bullets": [
            (0, '사전학습은 긴 글을 고정 길이(SEQ_LEN=128)로 잘라 학습한다'),
            (1, '문서 경계에 맞추지 않고 이어붙인 뒤 자르는 것이 일반적'),
            (1, '패딩 낭비 없이 GPU 를 꽉 채워 쓰기 위해서'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 모델 구현',
         "title": 'GPT 구조 개관',
         "bullets": [
            (0, '입력 토큰 ID → 토큰 임베딩 + 위치 임베딩'),
            (0, '→ [ Decoder Block ] × N   (LayerNorm → Causal Self-Attention → 잔차 / FFN → 잔차)'),
            (0, '→ LayerNorm → Linear(vocab 크기) → 다음 토큰의 확률'),
            (1, '앞서 이론에서 본 그 구조를 코드로 그대로 옮긴다'),
            (1, '실습값: 임베딩 256차원 · 헤드 4개(256/4=64) · 블록 4층 · FFN 1024 · dropout 0.1 — 합쳐서 약 2M'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 모델 구현',
         "title": 'Causal Self-Attention — 뒤를 가린다',
         "bullets": [
            (0, '학습할 때는 정답 문장을 통째로 넣는다 — 가리지 않으면 뒤를 보고 베낀다'),
            (0, '상삼각 마스크를 -inf 로 — softmax 를 거치면 확률 0'),
            (1, "GPT 가 '다음 토큰 맞히기'가 되는 핵심 장치"),
        ]},
        {"header": '3. 미니 GPT 만들기 – 모델 구현',
         "title": 'FeedForward 와 잔차 연결',
         "bullets": [
            (0, '어텐션 = 토큰끼리 정보를 주고받는 부분'),
            (0, 'FeedForward = 토큰 하나하나를 따로 가공하는 부분'),
            (0, 'Block 은 둘을 잔차(x + f(x))로 묶는다 — 원래 정보를 잃지 않고 더하기만 해서, 깊게 쌓아도 학습이 무너지지 않는다'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 모델 구현',
         "title": 'PreTrainedModel 상속 — 생태계에 얹는다',
         "bullets": [
            (0, '직접 만든 모델을 HuggingFace PreTrainedModel 로 감싼다'),
            (1, 'Trainer 로 학습하고 generate() 로 생성할 수 있게 된다'),
            (0, '파라미터는 임베딩·출력 레이어에 몰려 있다 — 사전 크기를 함부로 못 늘리는 이유'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 사전학습',
         "title": '학습 — 목표는 다음 토큰 맞히기 하나',
         "bullets": [
            (0, '그것만 반복하면 문법과 표현을 스스로 익힌다 — 이것이 사전학습'),
            (0, 'Trainer 에 데이터셋만 넘기면 루프는 표준 — loss 가 내려가는지 본다'),
            (1, '너무 빨리 0 에 가까워지는 것도 좋은 신호가 아니다 (외우는 중일 수 있다)'),
            (1, '실습값: 배치 64 · 3 에폭 · LR 5e-4 (이 값이 뒤 이어학습 LR 의 기준) — 학습 2~4분'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 텍스트 생성',
         "title": '디코딩 전략 — 같은 모델, 다른 글',
         "bullets": [
            (0, '모델은 매 순간 다음 토큰의 확률 분포를 내놓는다'),
            (0, '그 분포에서 어떻게 고를 것인가가 디코딩 전략:'),
            (1, 'Greedy / Beam / Random / Top-K / Top-P — 다섯을 같은 프롬프트로 비교한다'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 텍스트 생성',
         "title": 'Greedy 와 Beam — 반복의 함정',
         "bullets": [
            (0, 'Greedy: 항상 최고 확률 토큰 — 결정적이지만 다양성이 없다'),
            (0, 'Beam: 전체 확률이 높은 문장을 찾는다 — 그런데 오히려 더 반복한다'),
            (1, '확률 높은 문장 = 짧고 안전하고 반복적인 문장이기 쉽다'),
            (1, '번역·요약(정답 있는 작업)엔 맞고, 이야기 생성(열린 작업)엔 단점'),
        ]},
        {"header": '3. 미니 GPT 만들기 – 텍스트 생성',
         "title": 'Top-K 와 Top-P — 자르는 기준이 다르다',
         "bullets": [
            (0, 'Top-K: 후보 개수를 고정 — 상황에 따라 적절한 개수가 다른 게 약점'),
            (0, 'Top-P: 확률 합으로 자른다 — 확신하면 후보가 줄고, 애매하면 늘어난다'),
            (1, '실무에서 Top-P 를 더 많이 쓰는 이유'),
            (0, '글자가 깨져 나오면(예: 깜짝�어) 버그가 아니라 ByteLevel 의 구조'),
            (1, '한글은 한 글자가 3바이트 — 낮은 확률 토큰까지 뽑으면 글자 중간 바이트만 나온다'),
            (1, '실습값: Top-K 10 · Top-P 0.9 · temperature 는 0.5 / 1.0 / 1.5 세 값을 비교'),
        ]},
    ],
    "minigpt_cpt": [
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": '이어서 학습시키기 — Continuous Pre-training',
         "bullets": [
            (0, '이 모델은 동화만 안다 — 새 도메인(위키)을 가르치고 싶다면?'),
            (0, '처음부터 다시는 너무 비싸다 → 이미 학습된 모델에 이어서 학습한다'),
            (1, 'llama-2-ko, EEVE-Korean 이 이 방식 — 한국어 LLM 대부분의 출생 경로'),
            (0, '재료는 오늘 첫 실습에서 정제한 ko_wiki_clean.jsonl (3,524건)'),
            (1, '그중 2,000건 × 앞 2,000자만 쓴다(CPT_DOCS·CPT_CHARS) — 전부 넣으면 토큰화·청킹에 시간이 다 간다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": '첫 번째 문제 — 토크나이저가 안 맞는다',
         "bullets": [
            (0, '동화로 학습한 사전 5,000 — 위키의 한자·용어·연도가 잘게 쪼개진다'),
            (1, '실측: 동화 0.417 vs 위키 0.849 토큰/글자 (높을수록 잘게 쪼갬 = 비효율) — 2배 차이'),
            (0, '실무의 답은 어휘 확장 (llama-2-ko: 32,000→46,336)'),
            (1, '단 확장 직후엔 성능이 떨어졌다가 수십억 토큰을 학습해야 회복 — 실습에선 안 한다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": '두 번째 문제 — 배운 것을 잊는다',
         "bullets": [
            (0, '새 데이터만 계속 넣으면 원래 알던 것을 잊는다 (치명적 망각)'),
            (0, '대응: 원래 데이터를 조금 섞어 같이 학습 — replay'),
            (1, '문헌(Ibrahim 2024): 1%만으로도 유의미, 도메인이 크게 다르면 25%, 50%는 과함'),
            (0, '동화 → 백과사전은 강한 이동 — 0 / 5 / 25% 를 직접 비교한다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": 'learning rate 는 1/10 로',
         "bullets": [
            (0, '본학습 5e-4 → 이어학습 5e-5 — 임의가 아니라 공개 레시피의 관례'),
            (1, 'EEVE-Korean 4e-5 · llama-2-ko 1e-5 (원 LR 의 1/10~1/30)'),
            (0, 'LR 을 낮추면 망각이 줄고, 높이면 새 도메인에 빨리 적응 — 트레이드오프'),
            (1, 'warmup 길이는 망각·적응 어디에도 영향 없음(같은 논문) — 튜닝 대상이 아니다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": 'Perplexity — 다음 단어를 얼마나 못 맞히나',
         "bullets": [
            (0, '모델에게 문장을 주고 다음 단어를 맞히게 했을 때, 얼마나 헷갈렸는지의 지표'),
            (1, '낮을수록 좋다 — 확신하면 작고, 갈팡질팡하면 크다'),
            (0, '값 자체보다 변화를 본다: 같은 평가셋에서 학습 전후로 오르내리는 방향'),
            (0, '이어학습이 동화를 잊었나 = 동화 문장에 대한 perplexity 가 얼마나 올랐나'),
            (1, '다음 장에서 replay 를 바꿔 가며 이 값이 어떻게 움직이는지 본다'),
            (1, '이 모델: 무작위면 사전 크기 5,000 근처 → 사전학습 후 동화 18.9 (L40S 실측)'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습',
         "title": '결과 — replay 트레이드오프 (L40S 실측)',
         "bullets": [
            (0, "replay 를 조금만 섞어도 동화가 상당히 돌아온다 — 문헌의 '1%만으로도 유의미' 그대로"),
            (0, '생성 결과에도 문체 오염이 보인다 — 동화를 쓰다 백과사전 말투가 튀어나온다'),
        ]},
        {"header": '4. 도메인 최적화 프리트레이닝에 대해서 – 미니GPT 마무리',
         "title": '여기서 만든 것',
         "bullets": [
            (0, '토크나이저를 직접 학습 → GPT 를 직접 구현 → 다음 토큰 맞히기로 사전학습'),
            (0, '디코딩 전략에 따라 같은 모델이 다른 글을 쓰는 것을 확인'),
            (0, '이어학습으로 새 도메인을 가르쳤고 그 대가(망각)와 막는 법(replay)을 봤다'),
            (0, '실제 LLM 은 같은 구조를 수천 배 키운 것 — 구조 자체는 방금 만든 것과 같다'),
            (1, '이 모델을 쓸 만하게 만드는 것(포스트트레이닝)은 내일 — 오늘은 그 재료가 될 데이터를 만드는 일이 아직 남았습니다'),
        ]},
    ],
    "tok_demo": [
        {"header": '3. 자연어처리 머신러닝 소개 – 챗봇을 위한 자연어 처리 실습',
         "title": '토크나이저 확인 — 문장이 몇 개로 쪼개지나',
         "bullets": [
            (0, 'tokenizer = AutoTokenizer.from_pretrained("jhu-clsp/mmBERT-base")'),
            (0, 'test_sentence = "안녕하세요, 반갑습니다."'),
            (1, 'encode = tokenizer.encode(test_sentence)'),
            (1, 'token_print = [tokenizer.decode(token) for token in encode]'),
            (1, 'print(encode)'),
            (1, 'print(token_print)'),
            (0, '출력(토큰 ID·조각)은 실행해서 직접 확인하세요 — 모델마다 다릅니다'),
            (0, '볼 것: 몇 개로 쪼개지나 · [CLS]/[SEP]가 붙나 · 모델과 토크나이저는 항상 짝'),
         ]},
    ],
    "ner_data": [
        {"header": '3. 자연어처리 머신러닝 소개 – 챗봇을 위한 자연어 처리 실습',
         "title": 'NER 데이터 — klue/klue (ner)',
         "bullets": [
            (0, 'dataset = load_dataset("klue/klue", "ner")'),
            (0, '학습 21,008 문장 · 컬럼은 tokens와 ner_tags (어절 단위 라벨)'),
            (1, '예전의 kor_ner는 datasets 5.x에서 로딩되지 않습니다 (스크립트형)'),
            (0, 'KLUE는 test 정답이 비공개 — 평가는 validation으로 합니다'),
        ]},
    ],
    "rag_run": [
        {"header": '1. Llama Index를 활용한 RAG 실습 – 위키피디아 Q&A 실습',
         "title": 'Q&A 실행 — 결과를 읽는 법',
         "bullets": [
            (0, 'response = query_engine.query("백남준은 누구인가요?")'),
            (0, '답을 읽을 때 함께 볼 것:'),
            (1, '한국어로 답하는가 (프롬프트 교체가 실제로 반영됐는가)'),
            (1, '근거 문서(source_nodes)와 답이 맞는가 — 다음 장에서 근거를 직접 확인'),
            (0, '4B 모델이라 문장이 어색한 부분도 있습니다 — 그것도 관찰 대상입니다'),
        ]},
    ],
    "rag_subq": [
        {"header": '1. Llama Index를 활용한 RAG 실습 – 위키피디아 Q&A 실습',
         "title": '서브질문 — 왜 쓰고, 어디서 안 통하나',
         "bullets": [
            (0, '복합 질문("백남준과 맥스웰은 각각…")은 한 번의 검색으로 안 된다 — 둘이 함께 나오는 문서가 없다'),
            (1, 'LLM 이 하위 질문으로 쪼개 각각 검색 → 답을 합친다 (이게 서브질문의 동기)'),
            (0, '서브질문 생성 프롬프트도 기본이 영어 — 영어 하위 질문은 한국어 위키에서 0건'),
            (1, '한국어 생성 프롬프트로 update_prompts() 교체 (refine 때와 같은 방식)'),
            (0, '노트북 실행은 단일 개체 질문: sub_query_engine.query("백남준은 누구인가요?")'),
            (1, '쪼갤 필요 없는 질문에 쓰면 LLM 호출만 늘고 낫지 않다 — 항상 좋은 도구는 없다'),
        ]},
    ],
    "ml2dl": [
        {"header": '2.\t트랜스포머와 ChatGPT – 머신러닝에서 딥러닝으로',
         "title": '머신러닝 — 사람이 피처를 설계하던 시대',
         "bullets": [
            (0, '텍스트를 숫자로 바꾸는 일을 사람이 했다 — 단어 빈도(BoW), TF-IDF'),
            (0, '그 위에 통계 모델(로지스틱 회귀 · SVM)을 얹는 구조'),
            (0, '성능은 모델보다 피처를 어떻게 설계하느냐가 좌우했다'),
            (1, '새 도메인마다 피처를 다시 설계 — 옮겨 쓸 수 있는 것이 적었다'),
        ]},
        {"header": '2.\t트랜스포머와 ChatGPT – 머신러닝에서 딥러닝으로',
         "title": '딥러닝 — 표현을 데이터에서 배운다',
         "bullets": [
            (0, '단어를 사람이 설계한 규칙이 아니라 학습된 벡터(임베딩)로 표현'),
            (0, '다층 신경망이 피처를 스스로 만들어 낸다 — 표현 학습'),
            (0, '사람의 일이 피처 설계에서 구조(아키텍처) 설계로 옮겨 갔다'),
            (1, '데이터가 많을수록 좋아지는 성질이 여기서 시작된다'),
        ]},
        {"header": '2.\t트랜스포머와 ChatGPT – 머신러닝에서 딥러닝으로',
         "title": '순차 데이터와 RNN의 한계',
         "bullets": [
            (0, '문장은 순서가 있는 데이터 — RNN은 앞에서부터 차례로 읽는다'),
            (0, '한계 ① 멀리 떨어진 단어의 관계가 흐려진다 (장거리 의존)'),
            (0, '한계 ② 순서대로만 계산할 수 있어 병렬화가 안 된다 — 크게 못 키운다'),
        ]},
        {"header": '2.\t트랜스포머와 ChatGPT – 머신러닝에서 딥러닝으로',
         "title": '어텐션 — 전체를 두고, 중요한 것에 스스로 주목한다',
         "bullets": [
            (0, 'RNN처럼 앞에서부터 한 칸씩 전달하는 대신, 문장 전체를 한눈에 두고 시작한다'),
            (0, '그 전체에서 지금 필요한 단어가 무엇인지 스스로 골라 주목한다 — 멀어도 흐려지지 않는다'),
            (1, "사과가 떨어지는 장면에서, 배경 전체가 눈에 들어와도 '떨어지는 사과'에 저절로 주목하는 것과 같다"),
            (0, '단어를 순서대로 처리할 필요가 없어 병렬 계산이 가능해졌다 — 크게 키울 수 있게 됐다'),
        ]},
    ],
    # p51(Self-Attention 컨셉·행렬) 바로 앞. 어텐션 vs self-attention 을 구분하고,
    # 한 단어가 같은 문장을 보는 예를 점수 막대로 먼저 보여준다 (p51 이 행렬로 일반화).
    "selfattn_ex": [
        {"header": '2.\t트랜스포머와 ChatGPT – Transformer 소개',
         "title": 'Self-Attention — 한 문장이 자기 자신을 본다',
         "bullets": [
            (0, '앞의 어텐션은 두 문장 사이였다 — 예: 번역할 단어가 입력 문장을 본다'),
            (0, 'Self-Attention은 한 문장 안에서, 각 단어가 같은 문장의 다른 단어를 본다'),
            (1, '아래는 \'먹었다\'가 같은 문장의 단어들을 보는 정도 — 다음 장은 이걸 모든 단어 쌍으로 넓힌다'),
        ]},
    ],
    # 원본 p58(KorQuAD)의 리더보드 캡처가 2019~2022년 것이라 낡았다. 순위표 대신
    # "정답이 지문 속 구간"이라는 태스크 자체를 실제 데이터로 보여준다.
    # 값은 datasets-server API 실측 (2026-09-01, train[0:3] — 세 문항이 지문을 공유).
    "pretrain_intro": [
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '과제마다 모델을 새로 만들던 시대',
         "bullets": [
            (0, '분류 모델 하나, 번역 모델 하나 — 과제마다 데이터를 모으고 처음부터 학습'),
            (0, '과제별 라벨 데이터 수천~수만 건이 늘 발목을 잡았다'),
            (1, '어제 분류·NER 실습을 사전학습 없이 했다면 15만 건으로도 부족했을 것'),
        ]},
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '전이학습 — 한 번 배운 언어를 옮겨 쓴다',
         "bullets": [
            (0, '언어 자체를 크게 한 번 배워 두고(사전학습), 과제에는 살짝 맞춘다(파인튜닝)'),
            (0, 'BERT · GPT 가 이 2단계 구조를 표준으로 만들었다'),
            (1, '어제 0.1 에폭 만으로 분류가 됐던 이유 — 모델이 이미 언어를 알고 있었다'),
            (0, '과제별 데이터 요구량이 수만 건에서 수백 건 수준으로 내려왔다'),
        ]},
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '자기지도학습 — 라벨 없이 배운다',
         "bullets": [
            (0, '사전학습의 라벨은 텍스트 자체다 — 다음 토큰 맞히기, 가린 토큰 맞히기'),
            (1, '가린 토큰 맞히기 = MLM(BERT·양방향) · 다음 토큰 맞히기 = NTP(GPT·왼→오) — 어제 배운 그 둘'),
            (0, '사람이 라벨을 달지 않으므로 웹 전체가 학습 데이터가 된다'),
            (1, '라벨 병목이 사라지자 남은 병목은 데이터의 양과 질 — 그래서 오늘 데이터 처리부터 한다'),
        ]},
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '스케일링 — 크기가 능력을 산다',
         "bullets": [
            (0, '모델 · 데이터 · 연산을 함께 키우면 성능이 예측 가능하게 좋아진다 (스케일링 법칙)'),
            (0, '일정 규모를 넘으면 가르치지 않은 능력이 나타나기 시작했다'),
            (1, '지시 따르기 · 몇 개의 예시로 배우기(few-shot) — 내일 3일차의 주제들'),
            (0, '그래서 경쟁은 "누가 더 좋은 데이터를 더 많이 태우는가" 가 됐다'),
        ]},
        {"header": '1.\tPre-training 등장 배경과 목적',
         "title": '그래서 오늘 하는 것',
         "bullets": [
            (0, '사전학습의 재료 — 데이터 처리 (바로 다음 절 + 실습)'),
            (0, '사전학습의 원리 — 미니 GPT 를 밑바닥부터 만들어 학습'),
            (0, '사전학습의 연장 — 이어학습(CPT)으로 새 도메인 가르치기'),
            (1, '구조와 코드는 공개돼 있습니다 — 오늘의 결론은 "데이터가 성능을 가른다" 입니다'),
        ]},
    ],
    "datacleaning": [
        {"header": '2.\tPre-training 데이터 처리',
         "title": '왜 데이터인가',
         "bullets": [
            (0, '모델 구조와 학습 코드는 공개돼 있다 — 같은 구조로 학습해도 성능이 갈린다'),
            (0, '가장 큰 이유는 데이터. 실무에서 시간이 가장 많이 드는 곳도 여기다'),
            (1, 'GPU 시간은 비싸다 — 쓰레기를 학습시킬 여유가 없다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '원본을 눈으로 본다',
         "bullets": [
            (0, '실습: 한국어 위키백과 5,000건 — 비교적 깨끗한 코퍼스조차 지저분하다'),
            (0, '중복 문서 · 목록뿐인 문서 · 다른 언어 · 마크업 잔해'),
            (1, '숫자(통계)만 보지 말고 실물을 열어 보는 것 — 이 실습에서 반복되는 원칙'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '정확 중복 제거 — 해시',
         "bullets": [
            (0, '완전히 같은 문서부터 걷어낸다'),
            (0, '문자열끼리 직접 비교하면 문서 수의 제곱 — 해시로 바꾸면 한 번씩만 훑는다'),
            (0, '그 전에 유니코드 정규화(NFKC) — 눈에 같아 보여도 표현이 다르면 다른 문자열'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '진짜 문제는 근사 중복',
         "bullets": [
            (0, '글자 하나만 달라도 해시는 완전히 달라진다 — 정확 중복은 사실 별로 없다'),
            (0, '웹 코퍼스의 실제 중복: 같은 기사의 재게시 · 같은 템플릿 · 문단 하나 차이'),
            (1, '이걸 잡아야 모델이 특정 문장을 통째로 외우는 것을 막는다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '자카드 유사도와 shingle',
         "bullets": [
            (0, '문서를 연속된 n글자 조각(shingle)의 집합으로 만든다'),
            (0, '겹침 정도 = 자카드 유사도  J(A,B) = |교집합| / |합집합|'),
            (0, '문제: 문서 2만 개면 비교 쌍이 2억 개 — 전부 비교할 수 없다'),
            (1, '실습값 SHINGLE=5 — 한국어는 글자당 정보량이 커서 영어보다 짧게 잡는다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": 'MinHash — 시그니처로 압축',
         "bullets": [
            (0, 'shingle 집합을 여러 해시 함수로 돌려 각 함수의 최솟값만 남긴다'),
            (0, '두 시그니처가 일치하는 비율 ≈ 자카드 유사도 (아래 그림의 직관)'),
            (0, '긴 문서도 고정 길이 64개(NUM_HASH=64) 숫자로 — 이제 비교가 싸졌다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": 'LSH 밴딩 — 비교 자체를 줄인다',
         "bullets": [
            (0, '시그니처 64개를 16개 밴드 × 4개로 쪼갠다 (BANDS=16)'),
            (0, '한 밴드라도 완전히 같은 문서끼리만 후보로 본다 — 나머지 쌍은 보지도 않는다'),
            (1, '실습에서는 numpy 벡터화로 수천 문서를 수 초에 처리한다'),
            (0, '후보 쌍만 시그니처 일치율을 재서 0.8 이상(THRESHOLD)이면 중복 — 뒤에 나온 쪽을 버린다'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '품질 필터 — 정답 없는 휴리스틱',
         "bullets": [
            (0, '최소 길이 · 한국어 비율 · 반복도 · 특수문자 비율 · 문장 부호'),
            (0, '정답이 있는 작업이 아니다 — 코퍼스마다, 목적마다 기준이 달라진다'),
            (0, '중요한 것은 걸러진 것을 직접 보는 것'),
            (1, '정상 문서가 잘리면 기준이 너무 센 것 · 쓰레기가 남으면 조여야 하는 것'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '실무 도구 — 원리는 방금 만든 것과 같다',
         "bullets": [
            (0, 'datatrove (HuggingFace) — 파이프라인 방식, 로컬·Slurm·S3 같은 코드'),
            (0, 'dolma (AI2) · text-dedup (커뮤니티)'),
            (0, 'GopherQualityFilter = 우리가 손으로 만든 휴리스틱들의 표준 구현'),
            (1, '도구를 쓰더라도 무엇이 걸러지는지는 직접 확인해야 한다 — 기본값을 믿지 말 것'),
        ]},
        {"header": '2.\tPre-training 데이터 처리',
         "title": '산출물 — 오늘 만든 데이터로 오늘 학습한다',
         "bullets": [
            (0, 'ko_wiki_clean.jsonl — 5,000건 → 정제 후 3,524건 (실습 29초, L40S 실측)'),
            (0, '이 파일을 잠시 뒤 미니 GPT 의 이어학습(CPT)이 그대로 읽는다'),
            (0, '데이터셋 카드를 함께 남긴다 — 몇 달 뒤엔 출처도 기준도 기억나지 않는다'),
        ]},
    ],
    # Amazon 구조화 출력 코드 직전 — 생성 파라미터 + §9 함정을 개념으로 먼저 설명
    "amz_trap": [
        {"header": '5.\tvLLM을 활용한 데이터 생성 실습 – LLM을 활용한 데이터처리 실습',
         "title": '생성 파라미터 — 무엇을 걸고 생성하나',
         "bullets": [
            (0, 'temperature: 다음 단어를 얼마나 과감하게 고르나 (0에 가까울수록 보수적·일관)'),
            (1, '추출·분류처럼 정답이 정해진 일에는 낮게 (이 실습은 0.2)'),
            (0, 'top_p: 확률 상위 몇 %의 후보에서만 고른다 (나머지는 잘라 버린다)'),
            (0, 'seed: 같은 입력에 같은 출력이 나오도록 — 재현성'),
            (1, 'temperature·top_p·seed 는 3일차 퓨샷에서 직접 바꿔 가며 확인한다'),
        ]},
        {"header": '5.\tvLLM을 활용한 데이터 생성 실습 – LLM을 활용한 데이터처리 실습',
         "title": '구조화 출력의 함정 — 스키마에 길이 상한이 없으면 잘린다',
         "bullets": [
            (0, 'json_schema 로 형식은 강제되지만, 문자열 필드에 상한이 없으면 제약 디코딩이 끝없이 늘린다'),
            (1, 'max_tokens 에 걸려 JSON 이 중간에서 잘리고 → 몇 셀 뒤 json.loads 가 "Unterminated string" 으로 죽는다'),
            (1, '원인에서 가장 먼 곳에서 드러난다 — 가장 잡기 나쁜 형태'),
            (0, '두 가지 안전장치를 반드시 함께:'),
            (1, '① pydantic Field(max_length=N) — 각 문자열에 상한을 못박아 모델이 그 안에 들어오게 한다'),
            (1, '② finish_reason == "length" 검사 — 잘린 응답은 예외가 아니라 정상 응답이라 try/except 에 안 걸린다'),
            (0, 'max_tokens 만 거는 것으로는 부족 — 상한을 거는 것과 그 안에 들어오게 만드는 것은 다르다'),
            (1, '영어에선 우연히 넘어갔다가 한국어화로 출력이 길어지며 드러났다 (실측)'),
        ]},
    ],
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
         "title": "GEPA — 점수가 아니라 '왜 틀렸는지'로 배운다",
         "bullets": [
            (0, 'GEPA = Genetic-Pareto. 가중치는 한 번도 바꾸지 않고 프롬프트 텍스트만 진화시킨다'),
            (0, '① 신호가 숫자가 아니라 글 — 점수와 함께 오는 feedback 문장이 그래디언트 역할'),
            (0, '② reflection LM 이 그 글을 읽고 지시문을 다시 쓴다 = 돌연변이(mutation)'),
            (0, '③ Pareto frontier — 한 샘플이라도 최고인 후보를 보존해 다양성을 유지(선택)'),
            (1, '함정: feedback 없이 점수만 주면 reflection 이 "score 0.5" 만 봐 무력해진다'),
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
            (1, '검색→프롬프트 조립을 손으로 재현(build_prompt) — 아침엔 LlamaIndex 가 대신했던 그 몇 줄'),
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

# ml2dl 의 p24(3방식 비교)·p25(어텐션 상세)에 도해를 붙인다. 딕셔너리 리터럴에
# 직접 못 넣는 건 도해 함수가 위(EXTRAS 앞)에 정의돼 있고 NEW_SLIDES 가 그보다
# 뒤라, 리터럴 안에서 이름을 참조하면 순서상 걸리기 때문이다. 여기서 사후 주입한다.
NEW_SLIDES["ml2dl"][2]["diagram"] = draw_rnn_limit
NEW_SLIDES["ml2dl"][2]["body_h_cm"] = 7.6
# p25(어텐션 개념)에 사과 장면 이미지를 넣는다 — work/assets/p25_apple.png 가 있으면.
# self-attention 점수 막대는 p51 앞 신규 장(selfattn_ex)으로 옮겼다.
NEW_SLIDES["ml2dl"][3]["diagram"] = add_apple_image
NEW_SLIDES["ml2dl"][3]["body_h_cm"] = 8.8
NEW_SLIDES["selfattn_ex"][0]["diagram"] = draw_attention
NEW_SLIDES["selfattn_ex"][0]["body_h_cm"] = 7.0
# 3일차 메커니즘 도해 (2026-09-01)
NEW_SLIDES["lora"][0]["diagram"] = draw_lora
NEW_SLIDES["lora"][0]["body_h_cm"] = 9.0
NEW_SLIDES["grpo_group"][0]["diagram"] = draw_grpo_group
NEW_SLIDES["grpo_group"][0]["body_h_cm"] = 9.0

# 2일차 도해 (2026-09-01) — CPT 결과 = minigpt_cpt[5] (LR 뒤 Perplexity 삽입으로 +1),
# 데이터처리 산출물 = datacleaning[-1]
def _idx_by_title(key, title_sub):
    for i, spec in enumerate(NEW_SLIDES[key]):
        if title_sub in spec.get("title", ""):
            return i
    raise SystemExit(f"[중단] {key} 에서 제목 '{title_sub}' 못 찾음")


NEW_SLIDES["minigpt_cpt"][_idx_by_title("minigpt_cpt", "replay 트레이드오프")]["diagram"] = draw_cpt_tradeoff
NEW_SLIDES["minigpt_cpt"][_idx_by_title("minigpt_cpt", "replay 트레이드오프")]["body_h_cm"] = 5.0
NEW_SLIDES["datacleaning"][_idx_by_title("datacleaning", "산출물")]["diagram"] = draw_data_pipeline
NEW_SLIDES["datacleaning"][_idx_by_title("datacleaning", "산출물")]["body_h_cm"] = 8.5
NEW_SLIDES["datacleaning"][_idx_by_title("datacleaning", "MinHash — 시그니처로 압축")]["diagram"] = draw_minhash
NEW_SLIDES["datacleaning"][_idx_by_title("datacleaning", "MinHash — 시그니처로 압축")]["body_h_cm"] = 6.5
NEW_SLIDES["minigpt"][_idx_by_title("minigpt", "Causal Self-Attention")]["diagram"] = draw_causal_mask
NEW_SLIDES["minigpt"][_idx_by_title("minigpt", "Causal Self-Attention")]["body_h_cm"] = 5.5
NEW_SLIDES["promptopt"][_idx_by_title("promptopt", "학습 전에 최선")]["diagram"] = draw_gepa_loop
NEW_SLIDES["promptopt"][_idx_by_title("promptopt", "학습 전에 최선")]["body_h_cm"] = 6.5
NEW_SLIDES["promptopt"][_idx_by_title("promptopt", "왜 틀렸는지")]["diagram"] = draw_gepa_principle
NEW_SLIDES["promptopt"][_idx_by_title("promptopt", "왜 틀렸는지")]["body_h_cm"] = 8.0
NEW_SLIDES["evalsec_b"][_idx_by_title("evalsec_b", "judge 코드")]["diagram"] = draw_judge_code
NEW_SLIDES["evalsec_b"][_idx_by_title("evalsec_b", "judge 코드")]["body_h_cm"] = 3.4


# ---------------------------------------------------------------------------
# 코드 슬라이드 등록 (2026-09-03)
#   2일차 신규 실습(데이터처리·미니GPT·프롬프트최적화)은 원리·도해만 있고 코드가
#   없었다. 1일차·Amazon 처럼 핵심 코드를 보여 주기 위해 각 실습에 코드 장을 끼운다.
#   큰 리터럴을 건드리지 않고 개념 장 제목을 기준으로 뒤에 삽입한다.
#   코드 문자열은 r""" 로 두어 \\s \\n 등이 그대로 남게 한다 (셀에서 대조 완료).
#   코드 장 제목은 위 _idx_by_title 이 쓰는 예약 부분문자열을 피한다.
# ---------------------------------------------------------------------------
SCALING_ASSET = ROOT / "work" / "assets" / "scaling_law.png"


def draw_scaling(slide):
    """스케일링 법칙 그래프 — 이미지를 그대로 삽입하고 레퍼런스를 단다(2026-09-03 지시).

    강사가 work/assets/scaling_law.png 로 저장하면 add_picture 로 넣는다(add_apple_image
    패턴). 파일이 없으면 자리표시 박스를 두어 빌드는 깨지지 않는다. 손으로 다시 그리지
    않는다. 출처 캡션은 두 경우 모두 슬라이드 하단 고정 위치에 남긴다."""
    if SCALING_ASSET.exists():
        slide.shapes.add_picture(str(SCALING_ASSET), Cm(13.0), Cm(14.2), width=Cm(24.8))
    else:
        print(f"  [안내] {SCALING_ASSET.name} 없음 — 스케일링 그래프 자리표시만 둠")
        top, h, left, w = 13.5, 8.2, 9.0, 33.0
        box = _kq_card(slide, left, top, w, h, fill="FAFBFC", line="C6CCD2", radius=0.02)
        box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        box.text_frame.word_wrap = True
        p = box.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        _kq_run(p, "［ 스케일링 법칙 그래프 삽입 위치 — work/assets/scaling_law.png 저장 ］",
                size=15, color="9AA0A6")
    _label(slide, 9.0, 25.4, 33.0,
           "출처: Kaplan et al. (2020), “Scaling Laws for Neural Language Models” · arXiv:2001.08361",
           size=12, color="7A7A7A", align=PP_ALIGN.LEFT)


NEW_SLIDES["pretrain_intro"][_idx_by_title("pretrain_intro", "스케일링")]["diagram"] = draw_scaling
NEW_SLIDES["pretrain_intro"][_idx_by_title("pretrain_intro", "스케일링")]["body_h_cm"] = 5.8


def _codeslide(header, title, intro, code, *, body_h_cm=4.6, code_size=15):
    # intro 항목은 문자열(레벨 0) 또는 (레벨, 문자열) 튜플을 받는다
    return {"header": header, "title": title,
            "bullets": [ln if isinstance(ln, tuple) else (0, ln) for ln in intro],
            "code": _code(code), "body_h_cm": body_h_cm, "code_size": code_size}


def _insert_after(key, title_sub, specs):
    i = _idx_by_title(key, title_sub)
    NEW_SLIDES[key][i + 1:i + 1] = specs


_DC = '2.\tPre-training 데이터 처리'
_insert_after("datacleaning", "정확 중복 제거 — 해시", [_codeslide(
    _DC, '정확 중복 제거 — 정규화 후 해시 한 번',
    ['가장 쉬운 것부터. 문서 전체를 해시로 바꿔 한 번씩만 훑으면 완전히 같은 문서가 걸러집니다.'],
    r'''
def normalize(text):
    text = unicodedata.normalize("NFKC", text)  # 조합형→완성형 통일
    text = re.sub(r"\s+", " ", text)            # 연속 공백을 하나로
    return text.strip()

def doc_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()

seen, exact_dedup = set(), []
for d in docs:
    h = doc_hash(d["text"])
    if h in seen:            # 이미 본 문서면 버린다
        continue
    seen.add(h); exact_dedup.append(d)
''', body_h_cm=3.6)])

_insert_after("datacleaning", "MinHash — 시그니처로 압축", [_codeslide(
    _DC, 'MinHash 구현 — numpy 벡터화 한 줄',
    ['진짜 어려운 근사 중복입니다. shingle 집합을 해시 64개의 최솟값만 남겨 고정 길이로 압축합니다.'],
    r'''
SHINGLE, NUM_HASH = 5, 64
MOD = (1 << 31) - 1                     # 2^31-1, 소수
HASH_A = rng.integers(1, MOD, NUM_HASH, dtype=np.int64)   # 해시함수 64개의
HASH_B = rng.integers(0, MOD, NUM_HASH, dtype=np.int64)   # a, b 계수

def shingles(text):
    t = normalize(text)[:MAX_CHARS]     # 앞 1,500자만 본다
    return np.fromiter(
        {zlib.crc32(t[i:i+SHINGLE].encode()) & 0x7FFFFFFF
         for i in range(len(t) - SHINGLE + 1)}, dtype=np.int64)

def minhash(sh):
    # 이중 루프면 수천 문서에 수십 분. numpy 로 한 번에.
    return ((HASH_A[:, None] * sh[None, :] + HASH_B[:, None]) % MOD).min(axis=1)
''', body_h_cm=3.6, code_size=14)])

_insert_after("datacleaning", "LSH 밴딩", [_codeslide(
    _DC, 'LSH 밴딩 — 후보만 추린다',
    ['시그니처를 밴드로 쪼갭니다. 한 밴드라도 완전히 같은 문서끼리만 후보로 봐서 비교 자체를 줄입니다.'],
    r'''
ROWS = NUM_HASH // BANDS                # 밴드당 4개 (64/16)

buckets = defaultdict(list)
for idx, sig in enumerate(sigs):
    for b in range(BANDS):
        band = tuple(sig[b*ROWS:(b+1)*ROWS].tolist())
        buckets[(b, band)].append(idx)  # 같은 밴드값끼리 한 통에

candidates = set()
for members in buckets.values():
    if len(members) > 1:                # 한 밴드라도 겹친 문서끼리만
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                candidates.add((members[i], members[j]))
''', body_h_cm=3.6, code_size=14)])

_insert_after("datacleaning", "품질 필터", [_codeslide(
    _DC, '품질 필터 — 코드로 기준을 못박는다',
    ['정답이 없는 휴리스틱입니다. 무엇을 어떤 숫자로 자를지 코드에 적어 두고, 걸러진 걸 눈으로 봅니다.'],
    r'''
def quality_report(text):
    t = normalize(text); n = max(1, len(t))
    lines = [l for l in text.split("\n") if l.strip()]
    return {
        "length":         len(t),
        "hangul_ratio":   len(HANGUL.findall(t)) / n,
        "special_ratio":  len(SPECIAL.findall(t)) / n,
        "dup_line_ratio": Counter(lines).most_common(1)[0][1]/len(lines) if lines else 1.0,
        "sentence_end":   len(re.findall(r"[.!?]", t)),
    }

RULES = {
    "너무 짧음":     lambda r: r["length"] < 300,
    "한국어 부족":   lambda r: r["hangul_ratio"] < 0.3,
    "특수문자 과다": lambda r: r["special_ratio"] > 0.15,
    "같은 줄 반복":  lambda r: r["dup_line_ratio"] > 0.3,
    "문장 구조 없음": lambda r: r["sentence_end"] < 3,
}
''', body_h_cm=3.6, code_size=13)])


_MG = '3. 미니 GPT 만들기 – 실습 코드'
_insert_after("minigpt", "토크나이저를 직접 학습", [_codeslide(
    _MG, '토크나이저를 데이터로 학습 — train_from_iterator',
    ['사전은 주어지는 게 아니라 데이터에서 자랍니다. ByteLevel BPE 트레이너에 동화 코퍼스를 흘려보냅니다.'],
    r'''
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,                # 5,000
    special_tokens=[PAD, BOS, EOS],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

def corpus_iter(batch=1000):
    for i in range(0, len(train_docs), batch):
        yield train_docs[i : i + batch][COL]

tok.train_from_iterator(corpus_iter(), trainer=trainer, length=len(train_docs))
''', body_h_cm=3.6, code_size=14)])

_insert_after("minigpt", "토큰화와 청킹", [_codeslide(
    _MG, '청킹 — 이어붙여 SEQ_LEN 으로 자른다',
    ['문서 경계를 신경 쓰지 않습니다. 전부 이어붙인 뒤 SEQ_LEN 단위로 자릅니다.',
     'labels 는 input_ids 의 복사본 — 한 칸 밀기는 손실에서 합니다.'],
    r'''
def chunk(batch):
    flat = [i for seq in batch["ids"] for i in seq]      # 전부 이어붙이고
    total = (len(flat) // SEQ_LEN) * SEQ_LEN             # SEQ_LEN 배수로 자른다
    blocks = [flat[i : i + SEQ_LEN] for i in range(0, total, SEQ_LEN)]
    return {"input_ids": blocks, "labels": [b[:] for b in blocks]}
''', body_h_cm=4.6, code_size=15)])

_insert_after("minigpt", "Causal Self-Attention", [_codeslide(
    _MG, '어텐션 구현 — 뒤를 -inf 로 가린다',
    ["마스크 그림의 '가림'이 코드에선 한 줄입니다. masked_fill 뒤에 softmax 를 태우면 뒤쪽 확률이 0이 됩니다."],
    r'''
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)   # Q·K·V 한 번에 투영
        mask = torch.tril(torch.ones(cfg.n_positions, cfg.n_positions))
        self.register_buffer("mask", mask.view(1, 1, *mask.shape))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)               # 셋으로 쪼갬 (k·v 동일)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # Q·Kᵀ / √d
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)                        # 가린 뒤는 확률 0
        y = (att @ v).transpose(1, 2).reshape(B, T, C)      # 가중합·헤드 합치기
        return self.proj(y)
''', body_h_cm=3.6, code_size=13)])

_insert_after("minigpt", "다음 토큰 맞히기 하나", [_codeslide(
    _MG, 'MiniGPT forward — 한 칸 밀어 맞힌다',
    ['PreTrainedModel 을 상속하면 Trainer·generate() 가 공짜로 붙습니다.',
     "'다음 토큰 맞히기'는 로짓과 라벨을 한 칸 어긋나게 비교하는 두 줄입니다."],
    r'''
class MiniGPT(PreTrainedModel, GenerationMixin):
    config_class = MiniGPTConfig            # 직접 만든 모델을 HF 생태계에 얹는다

    def forward(self, input_ids, labels=None, **kw):
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)   # 토큰+위치 임베딩
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if labels is not None:              # i번째 출력으로 i+1번째 정답을
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1))  # ← 한 칸 밀어서 비교
        return CausalLMOutput(loss=loss, logits=logits)
''', body_h_cm=4.6, code_size=13)])


_CPT = '4. 도메인 최적화 프리트레이닝에 대해서 – 이어학습 실습'
_insert_after("minigpt_cpt", "learning rate 는 1/10 로", [
    _codeslide(
        _CPT, 'replay 구성 — 총량은 고정, 구성만 바꾼다',
        ["'동화를 조금 섞는다'가 코드로는 이렇습니다. 비율만 바꾸고 전체 블록 수는 고정합니다.",
         '학습량이 같은 조건이라야 replay 효과만 떼어 볼 수 있습니다.'],
        r'''
def make_mixed(replay_ratio, total_blocks):
    # 위키 + 동화(replay) 를 섞는다. 총량은 고정, 구성만 바꾼다.
    n_replay = int(total_blocks * replay_ratio)
    n_wiki   = total_blocks - n_replay
    parts = [wiki_train.shuffle(seed=0).select(range(min(n_wiki, len(wiki_train))))]
    if n_replay:
        parts.append(lm_train.shuffle(seed=0).select(range(min(n_replay, len(lm_train)))))
    return concatenate_datasets(parts).shuffle(seed=0)
''', body_h_cm=4.6, code_size=14),
    _codeslide(
        _CPT, '이어학습 설정 — LR 1/10 과 스케줄러',
        ["이론의 '본학습 LR의 1/10'이 learning_rate=CPT_LR 한 줄입니다.",
         'epoch 대신 max_steps 로 고정해 데이터 크기와 무관하게 시간을 잡습니다.'],
        r'''
m = copy.deepcopy(model_before)            # 매번 학습 전 사본에서 시작
ds = make_mixed(replay_ratio, NEED)

cpt_args = TrainingArguments(
    output_dir=f"minigpt_cpt_{int(replay_ratio*100)}",
    per_device_train_batch_size=BATCH_SIZE,
    max_steps=CPT_STEPS,                   # epoch 대신 스텝으로 고정
    learning_rate=CPT_LR,                  # 5e-5 = 본학습(5e-4)의 1/10
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr_rate": 0.1},   # 최저 LR = 최고의 10%
    warmup_steps=10,
    bf16=torch.cuda.is_bf16_supported(),
)
Trainer(model=m, args=cpt_args, train_dataset=ds,
        processing_class=tokenizer).train()
''', body_h_cm=4.6, code_size=13),
])


_PO = '6.\t프롬프트 자동 최적화'
_insert_after("promptopt", "3막 구조", [_codeslide(
    _PO, '구조화 출력 — 상한을 두 겹으로 건다',
    ['JSON 으로 답하라고만 하면 모델이 문자열을 끝없이 씁니다(한 요청이 90초 초과).',
     '스키마의 maxLength 와 API 의 max_tokens, 두 겹을 겁니다.'],
    r'''
TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        # maxLength 없으면 모델이 끝없이 쓴다
        "translation": {"type": "string", "maxLength": 3000},
    },
    "required": ["translation"],
}

def translate(text):
    r = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(text=text)}],
        response_format={"type": "json_schema",
            "json_schema": {"name": "translation", "schema": TRANSLATE_SCHEMA}},
        max_tokens=1500,   # ★ 스키마의 maxLength 만 믿으면 안 된다
    )
    return json.loads(r.choices[0].message.content)["translation"]
''', body_h_cm=4.6, code_size=13)])

_insert_after("promptopt", "왜 틀렸는지", [
    _codeslide(
        _PO, 'DSPy Signature — 형식은 고정, 지시문만 최적화',
        ['출력 형식은 우리 요구사항입니다 — OutputField 로 못박습니다.',
         'GEPA 가 고쳐 나가는 것은 오직 지시문 한 줄(docstring)입니다.'],
        r'''
class ExtractProduct(dspy.Signature):
    '''
    + "'''정보를 추출한다.'''"
    + r'''          # ← 시작 프롬프트. GEPA 가 이 문장을 고쳐 나간다

    text: str = dspy.InputField()
    # 출력 형식은 여기서 못박는다. 최적화 대상이 아니다.
    category: str = dspy.OutputField()
    features: list[str] = dspy.OutputField()
    target_users: list[str] = dspy.OutputField()

program = dspy.Predict(ExtractProduct)
''', body_h_cm=4.6, code_size=15),
    _codeslide(
        _PO, 'metric — 판정은 코드로, feedback 이 핵심 채널',
        ["무엇이 '잘한 것'인지 전부 코드로 판정합니다 — 채점 LLM 이 없으니 노이즈가 0입니다.",
         "GEPA 엔 점수만이 아니라 '왜 틀렸는지'를 문장으로 함께 넘깁니다."],
        r'''
def extract_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    # GEPA 는 5인자 시그니처 · Prediction(score=, feedback=) 을 받는다
    src   = gold.text
    feats = [str(x).strip() for x in (getattr(pred, "features", None) or [])]
    problems = []

    if len(feats) < MIN_FEATURES:
        problems.append(f"features 가 {len(feats)}개뿐이다. {MIN_FEATURES}개 이상 뽑아야 한다")
    ungrounded = [f for f in feats if not any(t in src for t in _tokens(f))]
    if feats and len(ungrounded) > len(feats) // 2:
        problems.append(f"원문에 없는 내용이다(예: {ungrounded[0][:20]})")
    # … category 길이 · target_users 개수 · 한국어 여부도 같은 방식으로

    score = max(0.0, 1.0 - 0.2 * len(problems))
    return dspy.Prediction(score=score, feedback="; ".join(problems) or "정상")
''', body_h_cm=4.6, code_size=13),
])

_insert_after("promptopt", "목적함수에 노이즈가 없어야", [_codeslide(
    _PO, 'reflection LM 분리 + GEPA 실행',
    ['프롬프트를 통째로 새로 쓰는 reflection 은 출력이 task 보다 훨씬 깁니다.',
     'task LM 을 그대로 넘기면 제안이 잘려 버려집니다 — 별도 LM 으로 분리합니다.'],
    r'''
reflection_lm = dspy.LM(
    f"openai/{MODEL}", api_base=BASE, api_key="EMPTY", model_type="chat",
    temperature=1.0,   # 공식 권장. 다양한 제안이 나와야 한다
    max_tokens=4096,   # ★ 여기가 핵심. task LM(작은 값)을 쓰면 제안이 잘린다
    cache=False,
)

optimizer = dspy.GEPA(
    metric=extract_metric,
    max_metric_calls=MAX_CALLS,   # 강의용 하드캡. 120회 ≈ 1.7분
    reflection_lm=reflection_lm,
    num_threads=8, track_stats=True,
)
optimized = optimizer.compile(program, trainset=trainset, valset=valset)
''', body_h_cm=4.6, code_size=14)])


# ---------------------------------------------------------------------------
# 3일차 평가 실습 코드 슬라이드 (2026-09-03)
#   평가 구간은 개념 장 위주라 코드가 judge() 하나뿐이었다 — 노트북 코드를 반영해
#   다른 실습 구간과 밀도를 맞춘다. 개념 장 뒤에 대응 코드 장을 끼운다.
#   제목은 위 _idx_by_title 앵커("자동 지표의 한계"·"judge 코드" 등)와 겹치지 않게 쓴다.
# ---------------------------------------------------------------------------
_EV = '2. 태스크 정의와 평가'

_insert_after("evalsec_a", "평가셋을 실무에서 운영", [_codeslide(
    _EV, '평가 데이터 · 응답 생성 (코드)',
    [(0, '평가셋은 (질문, reference) 쌍 — 개수보다 대표성'),
     (1, 'SFT 모델로 응답 생성, 없으면 예시 응답으로 폴백해 흐름은 동일하게')],
    r'''
eval_set = [
    {"question": "대한민국의 수도는 어디인가요?",
     "reference": "대한민국의 수도는 서울입니다."},
    {"question": "광합성이 무엇인지 설명해주세요.",
     "reference": "광합성은 식물이 빛으로 이산화탄소와 물에서 포도당을 만드는 과정입니다."},
    # 김치 · 리스트/튜플 … 총 4건 (개수보다 reference 가 태스크를 대표하는가가 중요)
]
SFT_DIR = "data/sft_model"          # SFT 실습을 완주하면 잡힌다
try:
    tok = AutoTokenizer.from_pretrained(SFT_DIR)
    model = AutoModelForCausalLM.from_pretrained(SFT_DIR, device_map="auto")
    predictions = [generate(model, tok, ex["question"]) for ex in eval_set]
except Exception:
    predictions = [...]             # 모델 없으면 준비된 예시 응답으로 (흐름 동일)
''', body_h_cm=4.6, code_size=12)])

_insert_after("evalsec_a", "자동 지표 — BLEU", [_codeslide(
    _EV, 'BLEU · ROUGE 계산 (코드)',
    [(0, 'evaluate.load 로 불러와 .compute 로 계산'),
     (1, '전체 평균과 건별을 함께 — 평균만 보면 무엇이 나빠졌는지 모른다')],
    r'''
bleu  = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
refs  = [ex["reference"] for ex in eval_set]

bleu_score  = bleu.compute(predictions=predictions,
                           references=[[r] for r in refs])
rouge_score = rouge.compute(predictions=predictions, references=refs)
print(f"BLEU {bleu_score['score']:.2f}")
for k in ["rouge1", "rouge2", "rougeL"]:
    print(f"{k} {rouge_score[k]:.4f}")
# 건별로도 본다 — 평균만 보면 무엇이 나빠졌는지 모른다
''', body_h_cm=4.6, code_size=12)])

_insert_after("evalsec_b", "자동 지표의 한계", [_codeslide(
    _EV, '자동지표의 한계 — 5케이스 실측 (코드)',
    [(0, '표현이 다르면 정답도 점수 하락, 틀려도 표현 비슷하면 점수 상승'),
     (1, '★ 부산 케이스: 틀렸는데 BLEU 는 높다 — 표면 일치의 함정')],
    r'''
question  = "대한민국의 수도는 어디인가요?"
reference = "대한민국의 수도는 서울입니다."
cases = [
    ("정답과 완전히 동일",    "대한민국의 수도는 서울입니다."),
    ("뜻이 같고 표현만 다름",  "서울이 대한민국의 수도입니다."),
    ("맞지만 더 짧음",        "서울입니다."),
    ("★ 틀렸는데 표현이 비슷",  "대한민국의 수도는 부산입니다."),
]
for name, pred in cases:
    b = bleu.compute(predictions=[pred], references=[[reference]])["score"]
    r = rouge.compute(predictions=[pred], references=[reference])["rougeL"]
    print(f"{name:<24}{b:>8.1f}{r:>10.3f}")
''', body_h_cm=4.6, code_size=12)])

_insert_after("evalsec_b", "judge 코드", [
    _codeslide(
        _EV, 'judge 를 평가셋에 적용 (코드)',
        [(0, 'judge() 를 평가셋 전체에 돌려 축별 점수와 평균을 낸다')],
        r'''
results = []
for ex, p in zip(eval_set, predictions):
    s = judge(ex["question"], ex["reference"], p)
    results.append(s)
    print(ex["question"])
    print(f"  정확성 {s['correctness']}  관련성 {s['relevance']}  자연스러움 {s['fluency']}")
    print(f"  사유: {s['reason']}")

print("── 평균 ──")
for k in ["correctness", "relevance", "fluency"]:
    print(f"  {k:<14}{mean(r[k] for r in results):.2f}")
''', body_h_cm=3.6, code_size=12),
    _codeslide(
        _EV, '재채점 — judge 가 놓친 걸 잡는다 (코드)',
        [(0, '같은 5케이스를 BLEU 와 judge 로 나란히 — 부산 케이스에서 갈린다'),
         (1, '자동지표로 거르고 judge 로 확인하는 실무 순서의 근거')],
        r'''
# 자동지표가 틀리게 매긴 5케이스를 judge 로 다시 채점
print(f"{'경우':<24}{'BLEU':>7}{'정확성':>8}")
for name, pred in cases:
    b = bleu.compute(predictions=[pred], references=[[reference]])["score"]
    s = judge(question, reference, pred)
    print(f"{name:<24}{b:>7.1f}{s['correctness']:>8}")
# '부산' 케이스: BLEU 는 높아도 judge 정확성은 낮다 — 자동지표가 놓친 걸 잡는다
''', body_h_cm=4.6, code_size=12),
])


# ---------------------------------------------------------------------------
# 텍스트 채우기 (도너 복제 후)
# ---------------------------------------------------------------------------
def _para_templates(tf) -> dict:
    """상자의 레벨별 <a:p> 를 서식 템플릿으로 수집한다 (run 이 있는 첫 문단)."""
    tmpl = {}
    for para in tf.paragraphs:
        if para.level not in tmpl and para.runs:
            tmpl[para.level] = para._p
    if not tmpl:
        raise SystemExit("[중단] 도너 상자에 run 있는 문단이 없습니다 — 도너를 바꾸세요")
    return tmpl


def _make_para(tmpl_p, text: str, size_scale: float | None = None):
    """템플릿 문단을 복제해 텍스트만 갈아 끼운다. 서식(pPr·rPr)은 그대로."""
    p_el = copy.deepcopy(tmpl_p)
    runs = p_el.findall(qn("a:r"))
    for r in runs[1:]:
        p_el.remove(r)
    for br in p_el.findall(qn("a:br")):
        p_el.remove(br)
    t = runs[0].find(qn("a:t"))
    t.text = text
    if size_scale is not None:
        rPr = runs[0].find(qn("a:rPr"))
        if rPr is not None and rPr.get("sz"):
            rPr.set("sz", str(int(int(rPr.get("sz")) * size_scale)))
    return p_el


def _fill_box(tf, lines: list[tuple[int, str]], size_scale: float | None = None) -> None:
    """상자를 (level, text) 목록으로 다시 채운다 — 도너의 레벨별 서식을 유지한 채."""
    tmpl = _para_templates(tf)
    levels = sorted(tmpl)

    def pick(lv):
        return tmpl[lv] if lv in tmpl else tmpl[min(levels, key=lambda a: abs(a - lv))]

    new_ps = [_make_para(pick(lv), tx, size_scale) for lv, tx in lines]
    txBody = tf._txBody
    for old in txBody.findall(qn("a:p")):
        txBody.remove(old)
    for np_ in new_ps:
        txBody.append(np_)


# 표 서식 — 도너 본문과 같은 글꼴을 쓴다 (1일차 p19 실측: 맑은 고딕 24pt).
# 슬라이드가 50.8×28.575cm(표준 16:9 의 1.5배)이라 pt 값도 1.5배로 잡아야 눈에 맞는다.
TABLE_FONT = "맑은 고딕"
TABLE_GRID_STYLE = "{5940675A-B579-460E-94D1-54222C63F5DA}"   # No Style, Table Grid


def _set_ea_font(run, name: str) -> None:
    """한글은 latin 만 지정하면 안 먹는다 — <a:ea> 를 따로 넣어야 한다."""
    rPr = run.font._rPr
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
        latin = rPr.find(qn("a:latin"))
        if latin is not None:
            latin.addnext(ea)
    ea.set("typeface", name)


def _shade(cell, hex_rgb: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    for old in tcPr.findall(qn("a:solidFill")):
        tcPr.remove(old)
    fill = etree.SubElement(tcPr, qn("a:solidFill"))
    etree.SubElement(fill, qn("a:srgbClr")).set("val", hex_rgb)


def _add_table(slide, t: dict, body) -> None:
    """신규 장에 표를 얹는다. 가로 위치·폭은 도너 본문 상자를 따른다.

    python-pptx 의 기본 표 스타일은 파란 줄무늬라 이 덱과 어긋난다 —
    "No Style, Table Grid" 로 바꾸고 머리행만 직접 칠한다 (2026-09-01).
    """
    rows = t["rows"]
    n_r, n_c = len(rows), len(rows[0])
    if any(len(r) != n_c for r in rows):
        raise SystemExit("[중단] table.rows 의 열 수가 행마다 다릅니다")
    widths = t["col_cm"]
    if len(widths) != n_c:
        raise SystemExit(f"[중단] table.col_cm {len(widths)}개 ≠ 열 {n_c}개")

    row_cm = t.get("row_cm", 1.6)
    left = Cm(t["left_cm"]) if "left_cm" in t else body.left
    gf = slide.shapes.add_table(n_r, n_c, left, Cm(t["top_cm"]),
                                Cm(sum(widths)), Cm(row_cm * n_r))
    tbl = gf.table
    tbl._tbl.tblPr.set("firstRow", "1")
    st = tbl._tbl.tblPr.find(qn("a:tableStyleId"))
    if st is None:
        st = etree.SubElement(tbl._tbl.tblPr, qn("a:tableStyleId"))
    st.text = TABLE_GRID_STYLE

    for i, w in enumerate(widths):
        tbl.columns[i].width = Cm(w)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Cm(row_cm)
        for c, text in enumerate(row):
            cell = tbl.cell(r, c)
            cell.margin_left = cell.margin_right = Cm(0.3)
            cell.text = text
            if r == 0:
                _shade(cell, t.get("head_fill", "DEEEF3"))
            para = cell.text_frame.paragraphs[0]
            for run in para.runs:
                run.font.name = TABLE_FONT
                run.font.size = Pt(t.get("font_pt", 18))
                run.font.bold = (r == 0)
                _set_ea_font(run, TABLE_FONT)


def _body_box(slide):
    """장식·보일러플레이트를 뺀 가장 큰 텍스트 상자."""
    boxes = [sh for sh in slide.shapes
             if sh.has_text_frame and BOILER not in sh.text_frame.text
             and sh.text_frame.text.strip()]
    if not boxes:
        raise SystemExit("[중단] 본문 텍스트 상자를 못 찾았습니다")
    return max(boxes, key=lambda b: (b.width or 0) * (b.height or 0))


def apply_extra(part, ex) -> None:
    """원본 장에 도형을 덧붙인다 (SlideExtra)."""
    slide = Slide(part._element, part)

    if ex.add_bullets:
        body = _body_box(slide)
        tf = body.text_frame
        tmpl = _para_templates(tf)
        levels = sorted(tmpl)
        for lv, tx in ex.add_bullets:
            src = tmpl[lv] if lv in tmpl else tmpl[min(levels, key=lambda a: abs(a - lv))]
            tf._txBody.append(_make_para(src, tx))
        if ex.body_h_cm is not None:
            body.height = Cm(ex.body_h_cm)

    if ex.move_picture is not None:
        pics = [sh for sh in slide.shapes if sh.shape_type == MSO_SHAPE_TYPE.PICTURE]
        if not pics:
            raise SystemExit(f"[중단] {ex.deck} p{ex.page}: 옮길 그림이 없습니다")
        pic = max(pics, key=lambda s: (s.width or 0) * (s.height or 0))
        l, t, w, h = ex.move_picture
        pic.left, pic.top, pic.width, pic.height = Cm(l), Cm(t), Cm(w), Cm(h)

    if ex.table is not None:
        _add_table(slide, ex.table, _body_box(slide))

    if ex.build_fn is not None:
        ex.build_fn(slide)


def fill_new_slide(part, spec: dict) -> None:
    slide = Slide(part._element, part)
    boxes = [sh for sh in slide.shapes
             if sh.has_text_frame and BOILER not in sh.text_frame.text
             and sh.text_frame.text.strip()]

    if "toc" in spec:
        # 목차 도너: "목 차" 제목은 그대로, 본문 상자만 교체.
        # 항목이 많으면 원본(4항목) 대비 비율로 글자를 줄여 상자를 넘치지 않게 한다.
        body = max(boxes, key=lambda b: len(b.text_frame.paragraphs))
        lines = list(spec["toc"])
        # 2·3일차 원본 목차는 번호를 자동 번호(buAutoNum)로 그린다 — 텍스트에도
        # 번호를 넣으면 "1. 1. ..." 로 두 번 찍힌다 (사용자 스크린샷으로 확인).
        tmpl0 = _para_templates(body.text_frame)[min(_para_templates(body.text_frame))]
        pPr = tmpl0.find(qn("a:pPr"))
        autonum = pPr is not None and pPr.find(qn("a:buAutoNum")) is not None
        if autonum:
            tab = chr(9)
            lines = [ln.split(tab, 1)[1] if tab in ln and ln.split(tab, 1)[0].rstrip(".").isdigit()
                     else ln for ln in lines]
        n = len(lines)
        scale = None if n <= 5 else round(5.5 / n, 3)
        _fill_box(body.text_frame, [(0, line) for line in lines], scale)
        return

    if len(boxes) < 3:
        raise SystemExit(f"[중단] 도너 장의 텍스트 상자가 예상과 다릅니다: {len(boxes)}개")
    boxes.sort(key=lambda b: (b.top or 0))
    header = boxes[0]
    body = max(boxes[1:], key=lambda b: (b.width or 0) * (b.height or 0))
    tagline = next((b for b in boxes[1:] if b is not body), None)

    _fill_box(header.text_frame, [(0, spec["header"])])
    # 도너 본문의 레벨 구조: lvl0 = 소제목 스타일(불릿·24pt), lvl1/2 = 하위 불릿.
    # 우리의 title → lvl0, bullets 의 level+1 → lvl1/2 로 자연 대응된다.
    lines = [(0, spec["title"])] + [(lv + 1, tx) for lv, tx in spec["bullets"]]
    _fill_box(body.text_frame, lines)
    if tagline is not None:
        _fill_box(tagline.text_frame, [(0, spec.get("tagline", ""))])
    if "table" in spec:
        _add_table(slide, spec["table"], body)
    if "diagram" in spec:
        # 본문 상자가 도해 자리를 침범하지 않게 높이를 줄이고, 그 아래에 도형을 그린다
        if "body_h_cm" in spec:
            body.height = Cm(spec["body_h_cm"])
        spec["diagram"](slide)
    if "code" in spec:
        # 본문(제목·인트로)을 줄이고 그 아래에 monospace 코드 카드를 얹는다
        bh = spec.get("body_h_cm", 3.2)
        body.height = Cm(bh)
        top = (body.top or 0) / 360000 + bh + 0.35
        draw_code_box(slide, spec["code"], top_cm=top, size=spec.get("code_size", 15))


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
    extras_by_page: dict[tuple[str, int], list[SlideExtra]] = {}
    for ex in EXTRAS:
        extras_by_page.setdefault((ex.deck, ex.page), []).append(ex)
    extra_hits = {id(ex): 0 for ex in EXTRAS}
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
                for ex in extras_by_page.get((it.deck, it.page), []):
                    apply_extra(part, ex)
                    extra_hits[id(ex)] += 1
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

    for ex in EXTRAS:
        if extra_hits[id(ex)] != 1:
            raise SystemExit(
                f"[중단] SlideExtra {ex.deck} p{ex.page}: {extra_hits[id(ex)]}회 적용 "
                f"(기대 1 — 그 장이 배치에 없거나 두 번 들어간다)\n  사유: {ex.why}")
    if verbose and EXTRAS:
        print(f"덧붙임 규칙 {len(EXTRAS)}건 — 전부 적용")
    return outputs


if __name__ == "__main__":
    build()
    print("\n검증: uv run --project tools python tools/validate_slides.py")
