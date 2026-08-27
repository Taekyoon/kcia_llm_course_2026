"""3일차 마무리 "RAG 개선 확인" 실습 노트북을 만든다.

왜 이 노트북인가 (2026-08-27 사용자 확정)
----------------------------------------
3일차를 RAG 로 열고 RAG 로 닫는 수미상관 구조로 재편했다.
아침에 만든 BM25 인덱스에 **같은 0.6B 모델의 학습 전(Base)/후(SFT)** 를 꽂아
포스트트레이닝이 시스템에서 무엇을 바꾸는지 눈으로 확인하는 것이 목적이다.

비교 설계의 근거:
- 아침 RAG 는 4B Instruct 로 돌았다. 학습한 0.6B 를 4B 와 비교하면
  "학습 효과"가 아니라 "크기 차이"가 보인다 → 같은 0.6B 의 전/후를 비교한다.
- Base 는 지시를 못 따라 이어쓰기만 하고, SFT 후에는 답하려 한다.
  개선의 실체가 '지식'이 아니라 '행동'이라는 것이 이 실습의 결론이다.
- vLLM 서버가 필요 없다 (0.6B 는 transformers 로 직접 돌린다).
  LlamaIndex 의 LLM 래퍼도 쓰지 않는다 — 검색→프롬프트→생성을 손으로 이어
  RAG 가 결국 무엇인지 탈신비화하는 쪽이 교육적이다.

의존 산출물 (같은 3일차 폴더의 상대경로):
- ./bm25_retriever  ← 0_HPC_BM25_RAG실습 이 저장
- data/sft_model    ← 3_HPC_SFT실습 이 저장 (어댑터 + 챗 템플릿 포함 토크나이저)
"""

from __future__ import annotations

import json
from pathlib import Path

from layout import work_path
from nbcommon import save_notebook

ROOT = Path(__file__).resolve().parent.parent
OUT = work_path("HPC_RAG개선실습.ipynb")   # 일자 배치는 tools/layout.py 가 정한다

MD, CODE = "markdown", "code"
CELLS: list[tuple[str, str]] = []


def md(t: str) -> None:
    CELLS.append((MD, t.strip("\n")))


def code(t: str) -> None:
    CELLS.append((CODE, t.strip("\n")))


# =====================================================================
md("""
# 학습이 시스템을 바꿨는가 — RAG 전후 비교

오늘 아침 RAG 시스템을 만들면서 이렇게 예고했습니다 —
**"하루의 끝에, 학습한 모델을 이 시스템에 다시 꽂아 봅니다."**
지금이 그때입니다.

## 무엇을 하게 되나

1. 아침에 저장해 둔 **BM25 검색 인덱스**를 다시 엽니다
2. 검색 → 프롬프트 조립 → 생성을 **손으로** 이어 붙입니다
3. 그 자리에 **학습 전 모델**(Qwen3-0.6B-Base)을 꽂아 봅니다
4. 같은 자리에 **오늘 학습한 모델**(SFT)을 꽂아 봅니다
5. 같은 질문, 같은 검색 결과, 같은 프롬프트 — **모델만 바꿔서** 비교합니다

## 왜 4B 와 비교하지 않는가

아침 실습은 기성품 **4B Instruct** 로 돌았습니다. 오늘 학습한 것은 **0.6B** 입니다.
그 둘을 비교하면 학습 효과가 아니라 **크기 차이**가 보입니다.

공정한 비교는 **같은 모델의 학습 전과 후**입니다. Base 0.6B 와 SFT 0.6B —
크기도 같고 사전학습도 같고, 다른 것은 오늘 오후에 한 학습뿐입니다.
달라진 것이 있다면 그것은 전부 학습이 만든 것입니다.

> vLLM 서버는 필요 없습니다. 0.6B 는 노트북 안에서 직접 돌립니다.
> 서버를 이미 내렸어도 됩니다. GPU 학습(SFT·DPO·GRPO)이 끝난 직후라도
> 0.6B 두 개를 올리는 정도는 메모리에 전혀 부담이 없습니다.
""")

code("""
%pip install -q llama-index-core llama-index-retrievers-bm25
""")

# =====================================================================
md("""
## 1. 아침의 검색기를 다시 연다

아침 실습이 `./bm25_retriever` 에 인덱스를 저장해 두었습니다.
색인을 다시 만들 필요 없이 그대로 불러옵니다 — **RAG 의 장점 그대로**입니다.
문서와 색인은 그대로 두고 모델만 갈아 끼울 수 있습니다.

없다는 에러가 나면 첫 실습(`0_HPC_BM25_RAG실습`)의 저장 셀을 먼저 실행하고 오세요.

여기에 운영상의 요점이 하나 숨어 있습니다. **인덱스와 모델은 서로 독립**입니다.
문서가 바뀌면 인덱스만 다시 만들고, 모델이 좋아지면 모델만 갈아 끼웁니다.
지금 우리가 하려는 것이 정확히 후자입니다 — 아침의 인덱스는 그대로 두고
모델 자리만 세 번 바꿔 봅니다. 실제 서비스에서 모델 업그레이드가
전체 재구축 없이 가능한 이유가 이 분리에 있습니다.
""")

code("""
from pathlib import Path
from llama_index.retrievers.bm25 import BM25Retriever

persist_dir = Path("./bm25_retriever")
if not persist_dir.exists():
    print("★" * 30)
    print("★ ./bm25_retriever 가 없습니다.")
    print("★ 오늘 첫 실습(0_HPC_BM25_RAG실습)에서 인덱스를 저장한 뒤 다시 실행하세요.")
    print("★" * 30)
    raise RuntimeError("검색 인덱스 없음")

retriever = BM25Retriever.from_persist_dir(str(persist_dir))

# 잘 열렸는지 확인 — 아침에 했던 질문 그대로
for node in retriever.retrieve("백남준은 누구인가요?")[:2]:
    print(f"[{node.score:.2f}]", node.text[:80].replace(chr(10), " "), "…")
""")

# =====================================================================
md("""
## 2. 조건을 고정한다 — 바꾸는 것은 모델 하나

비교 실험의 기본은 **바꾸는 것 하나만 남기고 전부 고정**하는 것입니다.

| 고정 | 값 |
|---|---|
| 질문 | 아침 실습과 같은 2개 |
| 검색기 | 아침의 BM25 인덱스, top-3 |
| 프롬프트 | 아래 한국어 템플릿 하나 |
| 생성 방식 | `do_sample=False` — 운이 끼어들지 않게 |

이 중 하나라도 함께 바뀌면, 결과 차이가 모델 때문인지 알 수 없게 됩니다.
2일차 CPT 비교에서 "같은 프롬프트, 같은 난수" 를 고집했던 것과 같은 이유입니다.

프롬프트 조립을 라이브러리에 맡기지 않고 직접 하는 것도 눈여겨보세요.
RAG 는 결국 **검색 결과를 프롬프트에 붙여 넣는 것**입니다. 아침에는 LlamaIndex 가
이 일을 대신해 줬지만, 속은 지금 보는 이 몇 줄입니다.
""")

code("""
QUESTIONS = [
    "백남준은 누구인가요?",
    "맥스웰 방정식이 무엇인가요?",
]

PROMPT_TEMPLATE = \"\"\"다음 문서를 참고해 질문에 한국어로 답하세요.

[문서]
{context}

[질문]
{question}

[답변]\"\"\"


def build_prompt(question: str) -> str:
    chunks = [node.text.strip() for node in retriever.retrieve(question)]
    context = (chr(10) + chr(10)).join(chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


print(build_prompt(QUESTIONS[0])[:400], "…")
""")

# =====================================================================
md("""
## 3. 학습 전 — Base 모델을 꽂는다

`Qwen3-0.6B-Base` 는 사전학습만 된 모델입니다. 다음 토큰을 이어 쓸 줄만 알고,
"질문에 답하라"는 **지시라는 개념 자체가 없습니다.**

그래서 예고합니다 — 아래 출력은 **답이 아니라 이어쓰기**일 겁니다.
문서 스타일의 문장을 계속 잇거나, 질문을 하나 더 만들어 내거나, 같은 말을
반복할 겁니다. 검색이 완벽한 문서를 찾아다 줘도 소용없습니다.

**시스템의 나머지 전부가 멀쩡해도 모델이 지시를 못 따르면 시스템은 동작하지
않습니다.** 이걸 눈으로 확인하는 것이 이 절의 목적입니다.

읽을 때 아침에 배운 구분을 다시 쓰세요 — 근거 문서가 엉뚱하면 **검색**의 실패,
근거는 맞는데 답이 이상하면 **생성**의 실패입니다. 아래에서 벌어지는 일은
전부 후자입니다. 검색은 아침과 똑같이 잘 찾아 줍니다.
""")

code("""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_ID = "Qwen/Qwen3-0.6B-Base"

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
base_model.eval()


def generate(model, prompt_text: str, max_new_tokens: int = 150) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
""")

code("""
base_answers = {}
for q in QUESTIONS:
    base_answers[q] = generate(base_model, build_prompt(q))
    print("=" * 60)
    print("질문:", q)
    print("Base 출력:", base_answers[q][:300])
""")

# =====================================================================
md("""
## 4. 학습 후 — 오늘 만든 SFT 모델을 꽂는다

오후에 SFT 실습이 저장한 어댑터(`data/sft_model`)를 불러옵니다.
같은 0.6B 에 **LoRA 어댑터 하나**가 얹힌 것이 전부입니다 — 파일로 수십 MB 입니다.

토크나이저도 저장본에서 불러옵니다. SFT 때 정한 **챗 템플릿이 함께 저장**되어
있어서, 학습할 때와 같은 형식으로 물을 수 있습니다.
"학습할 때와 쓸 때의 형식이 같아야 한다" — SFT 실습의 그 원칙이 여기서도 지켜집니다.

Base 에는 템플릿 없이 프롬프트를 그대로 넣었는데 SFT 에는 템플릿을 씌우니
불공정하다고 느낄 수 있습니다. 반대입니다 — **각 모델이 가장 잘 받아들이는
형식으로 주는 것**이 공정한 비교입니다. Base 에는 애초에 대화 형식이 없습니다.
""")

code("""
from peft import PeftModel
from transformers import AutoTokenizer as _AT

sft_dir = Path("data/sft_model")
sft_model, sft_tokenizer = None, None

if (sft_dir / "adapter_config.json").exists():
    print(f"SFT 산출물을 불러옵니다 — {sft_dir}")
    sft_tokenizer = _AT.from_pretrained(str(sft_dir))
    _base = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    sft_model = PeftModel.from_pretrained(_base, str(sft_dir))
    sft_model.eval()
else:
    print("★" * 30)
    print("★ data/sft_model 이 없습니다 — SFT 실습(3_HPC_SFT실습)을 먼저 완주하세요.")
    print("★ 아래 비교 셀은 Base 출력만 보여주게 됩니다.")
    print("★" * 30)
""")

code("""
sft_answers = {}
if sft_model is not None:
    for q in QUESTIONS:
        messages = [{"role": "user", "content": build_prompt(q)}]
        chat_text = sft_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = sft_tokenizer(chat_text, return_tensors="pt").to(sft_model.device)
        with torch.no_grad():
            out = sft_model.generate(**inputs, max_new_tokens=150,
                                     do_sample=False,
                                     pad_token_id=sft_tokenizer.eos_token_id)
        sft_answers[q] = sft_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("=" * 60)
        print("질문:", q)
        print("SFT 출력:", sft_answers[q][:300])
""")

# =====================================================================
md("""
## 5. 나란히 놓고 읽는 법

아래 셀이 두 출력을 붙여서 보여줍니다. 이렇게 읽으세요.

**Base 쪽에서 볼 것** — 답의 형태가 아예 없다는 것.
문서 문체를 이어 쓰거나, 질문을 새로 만들거나, 반복에 빠집니다.
내용이 틀렸다/맞았다 이전의 문제입니다.

**SFT 쪽에서 볼 것** — 일단 **답하려 한다**는 것.
질문을 받았다는 걸 알고, 문서를 참고해 답의 형태를 만듭니다.

**그리고 정직하게 볼 것** — SFT 출력도 완벽하지 않을 겁니다.
오늘 학습 데이터는 **상품 요약 877건**이었습니다. 위키 질문 답변용 데이터가
아닙니다. 그런데도 "질문에 답한다"는 행동은 옮겨 왔습니다 —
SFT 가 가르친 것이 **특정 지식이 아니라 행동 양식**이라는 증거입니다.
아침의 4B 답변보다 못한 것도 당연합니다. 그건 크기와 학습량의 차이입니다.
""")

code("""
for q in QUESTIONS:
    print("#" * 70)
    print("질문:", q)
    print()
    print("── 학습 전 (Base) " + "─" * 40)
    print(base_answers[q][:400])
    print()
    if sft_answers:
        print("── 학습 후 (SFT) " + "─" * 40)
        print(sft_answers[q][:400])
        print()
""")

# =====================================================================
md("""
## 마무리 — 3일의 결론

같은 검색기, 같은 질문, 같은 프롬프트에 모델만 바꿔 끼웠습니다.

- **Base**: 시스템의 나머지가 완벽해도 동작하지 않는다
- **SFT 후**: 답하려 한다 — 오늘 오후의 학습이 만든 차이의 전부

여기서 3일 과정의 조각들이 하나로 맞춰집니다.

| 배운 것 | 시스템에서의 역할 |
|---|---|
| 데이터 정제·생성 (2일차) | 학습의 재료 — 오늘 쓴 877건이 그것 |
| 사전학습·CPT (2일차) | 모델이 언어를 아는 이유 |
| 평가 (오늘) | "좋아졌다"를 말할 수 있는 근거 |
| 프롬프트·퓨샷 (오늘) | 학습 없이 먼저 시도하는 것 |
| SFT·DPO·GRPO (오늘) | **행동을 바꾸는** 도구 |
| RAG (오늘) | **지식을 넣는** 도구 |

마지막 두 줄이 이 실습의 결론입니다. **지식은 검색으로 넣고, 행동은 학습으로
바꿉니다.** 방금 그 둘이 한 시스템 안에서 각자의 일을 하는 것을 봤습니다.

실무로 돌아가면 순서는 늘 같습니다 —
**프롬프트로 되는지 먼저, 안 되면 그때 학습. 모르는 것은 검색으로.**
그리고 무엇을 하든 **바꾸기 전에 잴 수단부터** 갖춥니다. 오늘 아침 평가 실습을
학습보다 먼저 한 이유이고, 방금의 전후 비교가 성립한 이유입니다.
그 판단을 스스로 내릴 수 있게 된 것이 이 3일의 목표였습니다.
""")


# =====================================================================
def build() -> dict:
    cells = []
    for kind, src in CELLS:
        cell = {"cell_type": kind, "metadata": {},
                "source": src.splitlines(keepends=True)}
        if kind == CODE:
            cell["outputs"] = []
            cell["execution_count"] = None
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_notebook(OUT, build())
    n_code = sum(1 for k, _ in CELLS if k == CODE)
    print(f"생성: {OUT}")
    print(f"  셀 {len(CELLS)}개 (코드 {n_code} · 마크다운 {len(CELLS) - n_code})")


if __name__ == "__main__":
    main()
