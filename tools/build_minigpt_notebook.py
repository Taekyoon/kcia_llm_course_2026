"""MiniGPT 실습 노트북을 PyTorch/HuggingFace 스택으로 새로 만든다.

    uv run python tools/build_minigpt_notebook.py

왜 새로 만드는가
----------------
기존 노트북은 9종 중 **혼자만 Keras/TensorFlow** 스택이었다. 그래서
  - keras-hub → tensorflow-text → tensorflow 2.20 → nvidia-*-cu12 를 끌어와
    기본 커널의 torch cu130 과 충돌하므로 **별도 venv 가 필요**했고
  - 수강생이 나머지 8종에서 배운 도구(transformers·datasets·Trainer)를
    여기서만 다시 배워야 했다.
슬라이드도 어차피 재작성 대상이었다(C-2: 1일차 p63·p69-90 이 구 keras_nlp API).

교육 목표는 그대로 유지한다
  ① 토크나이저를 직접 학습시켜 동작을 이해한다
  ② GPT 구조를 직접 구성하고 사전학습한다
  ③ 디코딩 전략 5종을 비교한다

원본 대비 의도적으로 바꾼 것
  - Keras 레이어 조립 → **nn.Module 로 직접 작성** 후 PreTrainedModel 로 감싼다.
    1일차 p49-58 에서 배운 Self-Attention·Multi-head 가 코드로 바로 연결된다.
  - keras_hub.samplers 5종 → `model.generate()` 플래그 5종
  - SimpleBooks(영어, S3 직링크) → **TinyStories 한국어본**(MIT, HF)
  - NUM_HEADS 3 → 4. PyTorch 표준 구현은 n_embd 가 n_head 로 나누어떨어져야 한다
    (256/3 은 나누어떨어지지 않는다). Keras MultiHeadAttention 은 key_dim 이
    별도라 3이 가능했다.
  - WordPiece → **ByteLevel BPE**. 한국어는 음절이 많아 vocab 5000 WordPiece 로는
    [UNK] 가 다수 발생한다. ByteLevel 은 UNK 가 원천적으로 없다.
"""

from __future__ import annotations

import json
from pathlib import Path

from layout import work_path
from nbcommon import DATA_DIR_CODE, DATA_DIR_MD

ROOT = Path(__file__).resolve().parent.parent
OUT = work_path("HPC_MiniGPT실습.ipynb")   # 일자 배치는 tools/layout.py 가 정한다

MD = "markdown"
CODE = "code"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append((MD, text.strip("\n")))


def code(text: str) -> None:
    CELLS.append((CODE, text.strip("\n")))


# =====================================================================
md("""
# 미니 GPT 만들기 — 밑바닥부터 사전학습하기

이번 실습에서는 **GPT 를 직접 만들어서 처음부터 학습**시켜 봅니다.
남이 만든 모델을 가져다 쓰는 것이 아니라, 구조를 손으로 쌓고 데이터로 채웁니다.

## 무엇을 하게 되나

1. **토크나이저를 직접 학습**시킵니다 — 사전이 어떻게 만들어지는지 봅니다
2. **GPT 구조를 직접 구현**합니다 — Self-Attention, Multi-head, Decoder Block
3. **사전학습(Pre-training)** 을 돌립니다 — 다음 토큰 맞히기
4. **디코딩 전략 5가지**를 비교합니다 — Greedy / Beam / Random / Top-K / Top-P

앞서 트랜스포머 구조에서 배운 **Self-Attention 과 Multi-head** 가
여기서 실제 코드로 어떻게 생겼는지 확인하게 됩니다.

> GPU 런타임이 필요합니다. 학습에 수 분이 걸립니다.
""")

code("""
%pip install -q -U transformers datasets tokenizers accelerate
""")

code("""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

print("torch:", torch.__version__, "/ CUDA 사용가능:", torch.cuda.is_available())
""")

# ---------------------------------------------------------------------
md("""
## 하이퍼파라미터

일부러 작게 잡았습니다. 실습 시간 안에 학습이 끝나야 하니까요.
그래도 GPT 의 구조는 대형 모델과 **완전히 같습니다.** 크기만 다릅니다.

> 학습에 **2~4분** 정도 걸립니다. 더 줄이고 싶으면 `N_DOCS` 나 `EPOCHS` 를 낮추면 되는데,
> 너무 줄이면 생성 결과가 같은 말만 반복하게 됩니다. 실제로 그렇게 만들었다가
> 학습이 5초 만에 끝나서 `"엄마, "엄마, "엄마,` 만 나온 적이 있습니다.
""")

code("""
# 데이터
SEQ_LEN    = 128      # 한 번에 보는 토큰 수 (컨텍스트 길이)
VOCAB_SIZE = 5000     # 토크나이저 사전 크기
MIN_CHARS  = 200      # 너무 짧은 글은 학습에서 뺀다
N_DOCS     = 150000   # 사용할 문서 수

# 모델
N_EMBD  = 256         # 임베딩 차원
N_HEAD  = 4           # 어텐션 헤드 수 — N_EMBD 를 나누어떨어뜨려야 한다 (256/4=64)
N_LAYER = 4           # 디코더 블록 개수
FF_DIM  = 4 * N_EMBD  # 피드포워드 은닉 차원 (관례상 임베딩의 4배)
DROPOUT = 0.1

# 학습
BATCH_SIZE = 64
EPOCHS     = 3
LR         = 5e-4
""")

# ---------------------------------------------------------------------
md("""
## 1. 데이터 준비

**TinyStories** 를 씁니다. 아주 작은 언어 모델도 문법에 맞는 문장을 만들 수 있도록
어휘와 문장 구조를 단순하게 설계한 데이터입니다.

일반 웹 텍스트로 2백만 파라미터 모델을 학습시키면 의미 없는 글자만 나옵니다.
TinyStories 는 그 문제를 피하려고 만들어졌습니다.
""")

code("""
from datasets import load_dataset

# 한국어 번역본 (MIT). 원본 파일이 .txt 라 text 빌더로 읽힌다.
raw = load_dataset("g0ster/TinyStories-Korean", split="train")
print(raw)
print("컬럼:", raw.column_names)
""")

code("""
# 실제로 어떻게 들어있는지 눈으로 확인한다.
# 줄 단위로 쪼개져 있을 수도, 이야기 하나가 한 줄일 수도 있다.
for i in range(5):
    s = raw[i][raw.column_names[0]]
    print(f"[{i}] ({len(s)}자) {s[:120]}")
""")

code("""
COL = raw.column_names[0]

# 줄 단위로 들어있으면 이야기가 조각나므로, 짧은 줄은 버리고
# 충분히 긴 것만 하나의 학습 문서로 쓴다.
docs = raw.filter(lambda x: len(x[COL]) >= MIN_CHARS)
print(f"{len(raw):,}줄 → {len(docs):,}개 (>= {MIN_CHARS}자)")

n = min(N_DOCS, len(docs))
docs = docs.select(range(n))
split = docs.train_test_split(test_size=0.05, seed=42)
train_docs, eval_docs = split["train"], split["test"]
print(f"학습 {len(train_docs):,} / 평가 {len(eval_docs):,}")
print()
print(train_docs[0][COL][:300])
""")

# ---------------------------------------------------------------------
md("""
## 2. 토크나이저 학습

모델은 글자를 모릅니다. **숫자(토큰 ID)** 만 다룹니다.
그 변환표를 만드는 것이 토크나이저이고, 여기서는 **데이터로부터 직접 학습**시킵니다.

**ByteLevel BPE** 를 씁니다. 자주 붙어 나오는 바이트 쌍을 반복해서 합쳐가는 방식입니다.

한국어에 ByteLevel 을 쓰는 이유가 있습니다. 한국어는 음절 종류가 수천 개라
사전 크기 5,000 짜리 WordPiece 로는 **모르는 글자(`[UNK]`)가 대량 발생**합니다.
ByteLevel 은 모든 텍스트를 바이트로 먼저 쪼개기 때문에 UNK 가 원천적으로 없습니다.
""")

code("""
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

PAD, BOS, EOS = "[PAD]", "[BOS]", "[EOS]"

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=[PAD, BOS, EOS],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

def corpus_iter(batch=1000):
    for i in range(0, len(train_docs), batch):
        yield train_docs[i : i + batch][COL]

tok.train_from_iterator(corpus_iter(), trainer=trainer, length=len(train_docs))
print("학습된 사전 크기:", tok.get_vocab_size())
""")

code("""
# 문장 앞뒤에 [BOS], [EOS] 를 자동으로 붙이도록 한다.
# 모델이 "어디서 시작하고 어디서 끝나는지" 를 배우려면 이 표시가 필요하다.
bos_id, eos_id = tok.token_to_id(BOS), tok.token_to_id(EOS)
tok.post_processor = processors.TemplateProcessing(
    single=f"{BOS} $A {EOS}",
    special_tokens=[(BOS, bos_id), (EOS, eos_id)],
)
print(f"{BOS}={bos_id}, {EOS}={eos_id}, {PAD}={tok.token_to_id(PAD)}")
""")

code("""
from transformers import PreTrainedTokenizerFast

# transformers 생태계(Trainer, generate 등)에서 쓰려면 이 클래스로 감싼다.
# model_max_length 를 SEQ_LEN 으로 두지 않는다.
# 우리는 문서를 통째로 토큰화한 뒤 직접 SEQ_LEN 으로 자르기 때문에,
# 여기서 길이를 제한하면 "170 > 128" 같은 경고만 잔뜩 뜬다.
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    pad_token=PAD, bos_token=BOS, eos_token=EOS,
    model_max_length=int(1e9),
)
print(tokenizer)
""")

code("""
# 직접 학습시킨 토크나이저가 어떻게 쪼개는지 확인한다.
for s in ["옛날에 작은 고양이가 살았습니다.", "톰은 공원에 갔어요."]:
    ids = tokenizer.encode(s)
    print(f"{s}")
    print(f"  → {len(ids)}토큰 {tokenizer.convert_ids_to_tokens(ids)}")
    print(f"  → 복원: {tokenizer.decode(ids)}")
    print()
""")

# ---------------------------------------------------------------------
md("""
### 토큰화와 청킹

사전학습은 **긴 글을 고정 길이로 잘라서** 학습합니다.
문서 경계에 맞추지 않고 이어붙인 뒤 `SEQ_LEN` 단위로 자르는 것이 일반적입니다.
그래야 패딩 낭비 없이 GPU 를 꽉 채워 쓸 수 있습니다.
""")

code("""
def tokenize(batch):
    return {"ids": [tokenizer.encode(t) for t in batch[COL]]}

tok_train = train_docs.map(tokenize, batched=True, remove_columns=train_docs.column_names,
                           desc="토큰화(train)")
tok_eval  = eval_docs.map(tokenize, batched=True, remove_columns=eval_docs.column_names,
                          desc="토큰화(eval)")

def chunk(batch):
    flat = [i for seq in batch["ids"] for i in seq]      # 전부 이어붙이고
    total = (len(flat) // SEQ_LEN) * SEQ_LEN             # SEQ_LEN 배수로 자른다
    blocks = [flat[i : i + SEQ_LEN] for i in range(0, total, SEQ_LEN)]
    return {"input_ids": blocks, "labels": [b[:] for b in blocks]}

lm_train = tok_train.map(chunk, batched=True, batch_size=1000,
                         remove_columns=["ids"], desc="청킹(train)")
lm_eval  = tok_eval.map(chunk, batched=True, batch_size=1000,
                        remove_columns=["ids"], desc="청킹(eval)")

print(f"학습 블록 {len(lm_train):,}개 / 평가 블록 {len(lm_eval):,}개")
print("블록 하나:", len(lm_train[0]["input_ids"]), "토큰")
print(tokenizer.decode(lm_train[0]["input_ids"])[:200])
""")

# ---------------------------------------------------------------------
md("""
## 3. GPT 직접 만들기

이제 모델을 만듭니다. 라이브러리가 주는 완성품을 부르는 대신 **직접 씁니다.**
앞에서 배운 Self-Attention 과 Multi-head 가 코드로 어떻게 생겼는지 보기 위해서입니다.

구조는 이렇게 생겼습니다.

```
입력 토큰 ID
  → 토큰 임베딩 + 위치 임베딩
  → [ Decoder Block ] × N_LAYER
        ├─ LayerNorm → Causal Self-Attention → 잔차 연결
        └─ LayerNorm → FeedForward         → 잔차 연결
  → LayerNorm
  → Linear (vocab 크기로 투영)
  → 다음 토큰의 확률
```
""")

md("""
### 3-1. Causal Self-Attention

핵심은 **causal(인과) 마스킹** 입니다.

GPT 는 다음 토큰을 맞히는 모델입니다. 그런데 학습할 때는 정답 문장을 통째로 넣습니다.
아무 처리도 안 하면 3번째 토큰을 예측할 때 4번째, 5번째를 그냥 보고 베낍니다.
그래서 **자기보다 뒤에 있는 토큰을 못 보도록 가려야** 합니다.

아래에서 `mask` 로 상삼각 부분을 `-inf` 로 만드는 것이 그 처리입니다.
softmax 를 거치면 `-inf` 는 확률 0 이 됩니다.
""")

code("""
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd 는 n_head 로 나누어떨어져야 합니다"
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        # Query, Key, Value 를 한 번에 만든다 (합쳐서 3배 크기로 투영 후 쪼갬)
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        # 뒤를 못 보게 가리는 마스크. 학습 대상이 아니라 buffer 로 둔다.
        mask = torch.tril(torch.ones(cfg.n_positions, cfg.n_positions)).view(
            1, 1, cfg.n_positions, cfg.n_positions)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape

        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, n_head, T, head_dim) : 헤드별로 쪼개서 병렬 계산
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention Score = Q·Kᵀ / √d   (스케일링을 해줘야 학습이 안정적이다)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)

        y = att @ v                                   # 확률로 Value 를 가중합
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # 헤드를 다시 합친다
        return self.drop(self.proj(y))
""")

md("""
### 3-2. FeedForward 와 Decoder Block

어텐션이 "토큰끼리 정보를 주고받는" 부분이라면,
FeedForward 는 **토큰 하나하나를 따로 가공하는** 부분입니다.

Block 은 이 둘을 잔차 연결(residual)로 묶습니다.
`x + f(x)` 형태라 층을 깊게 쌓아도 그래디언트가 잘 흐릅니다.
""")

code("""
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc   = nn.Linear(cfg.n_embd, cfg.ff_dim)
        self.proj = nn.Linear(cfg.ff_dim, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.att = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.att(self.ln1(x))   # 잔차 연결
        x = x + self.mlp(self.ln2(x))
        return x
""")

md("""
### 3-3. 모델 전체

`PreTrainedModel` 을 상속하면 `Trainer` 로 학습하고 `generate()` 로 생성할 수 있습니다.
직접 만든 모델을 HuggingFace 생태계에 그대로 얹는 것입니다.
""")

code("""
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput


class MiniGPTConfig(PretrainedConfig):
    model_type = "minigpt"

    # transformers 내부는 표준 이름(num_hidden_layers, hidden_size ...)으로 config 를
    # 조회한다. 우리는 GPT-2 식 짧은 이름을 쓰므로 둘을 이어준다.
    # 이게 없으면 generate() 가 캐시를 준비하다가 이렇게 죽는다:
    #   AttributeError: 'MiniGPTConfig' object has no attribute 'num_hidden_layers'
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(self, vocab_size=5000, n_positions=128, n_embd=256,
                 n_layer=2, n_head=4, ff_dim=1024, dropout=0.1,
                 use_cache=False, **kw):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.ff_dim = ff_dim
        self.dropout = dropout
        # KV 캐시를 구현하지 않았다(매번 전체 시퀀스를 다시 계산한다).
        # 실습용 소형 모델이라 속도 차이가 크지 않고, 코드가 훨씬 단순해진다.
        self.use_cache = use_cache
        super().__init__(**kw)


class MiniGPT(PreTrainedModel, GenerationMixin):
    config_class = MiniGPTConfig
    main_input_name = "input_ids"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # 토큰 임베딩
        self.pos_emb = nn.Embedding(cfg.n_positions, cfg.n_embd)  # 위치 임베딩
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        # 밑바닥부터 학습하는 모델이라 초기화가 실제로 중요하다.
        # 너무 크면 발산하고 너무 작으면 학습이 안 된다. GPT-2 가 쓴 표준편차 0.02 를 따른다.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device)

        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if labels is not None:
            # 다음 토큰 맞히기: i번째 출력으로 i+1번째 정답을 맞힌다.
            # 그래서 한 칸씩 밀어서 비교한다.
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return CausalLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kw):
        # 컨텍스트 길이를 넘으면 뒤쪽만 남긴다 (캐시를 쓰지 않는 단순 구현)
        return {"input_ids": input_ids[:, -self.config.n_positions:]}
""")

code("""
config = MiniGPTConfig(
    vocab_size=tokenizer.vocab_size,
    n_positions=SEQ_LEN, n_embd=N_EMBD, n_layer=N_LAYER,
    n_head=N_HEAD, ff_dim=FF_DIM, dropout=DROPOUT,
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model = MiniGPT(config)
print(model)
""")

md("""
### 파라미터가 어디에 몰려 있나

이 모델에서 **임베딩과 출력 레이어가 파라미터의 대부분**을 차지합니다.
정작 "생각하는" 부분인 Decoder Block 은 상대적으로 작습니다.

사전 크기(`VOCAB_SIZE`)가 커지면 이 두 레이어가 같이 커집니다.
대형 모델이 사전 크기를 함부로 늘리지 못하는 이유가 여기 있습니다.
""")

code("""
groups = {"토큰 임베딩": 0, "위치 임베딩": 0, "Decoder Blocks": 0, "출력(head)": 0, "기타": 0}
for name, p in model.named_parameters():
    n = p.numel()
    if   name.startswith("tok_emb"): groups["토큰 임베딩"] += n
    elif name.startswith("pos_emb"): groups["위치 임베딩"] += n
    elif name.startswith("blocks"):  groups["Decoder Blocks"] += n
    elif name.startswith("head"):    groups["출력(head)"] += n
    else:                            groups["기타"] += n

total = sum(groups.values())
for k, v in groups.items():
    print(f"  {k:<16} {v:>10,}  ({v/total:5.1%})")
print(f"  {'합계':<16} {total:>10,}")
""")

# ---------------------------------------------------------------------
md("""
## 4. 사전학습

이제 학습시킵니다. 목표는 하나입니다 — **다음 토큰 맞히기**.
그것만 반복하면 모델이 문법과 표현을 스스로 익힙니다. 이것이 사전학습입니다.
""")

code("""
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="minigpt",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=100,
    bf16=torch.cuda.is_bf16_supported(),
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=lm_train,
    eval_dataset=lm_eval,
    processing_class=tokenizer,
)
trainer.train()
""")

code("""
# Perplexity: 모델이 다음 토큰을 고를 때 몇 개 중에서 헷갈리는지를 나타낸다.
# 낮을수록 좋다. 학습 전에는 사전 크기(5000)에 가깝고, 학습되면 크게 떨어진다.
metrics = trainer.evaluate()
print(metrics)
ppl = math.exp(metrics["eval_loss"])
print(f"\\nPerplexity: {ppl:.2f}")
print(f"  학습 전(무작위)이라면 약 {VOCAB_SIZE} 근처였을 값이다.")
print(f"  → 5000개 중에서 고르던 것이 지금은 {ppl:.0f}개 수준으로 좁혀졌다는 뜻이다.")
""")

# ---------------------------------------------------------------------
md("""
## 5. 텍스트 생성 — 디코딩 전략 5가지

학습된 모델은 매 순간 **다음 토큰의 확률 분포**를 내놓습니다.
그 분포에서 실제로 토큰 하나를 **어떻게 고를 것인가** 가 디코딩 전략입니다.

같은 모델이라도 고르는 방법에 따라 결과가 완전히 달라집니다.
""")

code("""
device = next(model.parameters()).device
prompt = torch.tensor([[tokenizer.bos_token_id]], device=device)

def show(title, explain, **kw):
    # use_cache=False: KV 캐시를 구현하지 않았으므로 매번 전체를 다시 계산한다.
    out = model.generate(prompt, max_new_tokens=80,
                         pad_token_id=tokenizer.pad_token_id,
                         use_cache=False, **kw)
    print("=" * 68)
    print(f"[{title}] {explain}")
    print("=" * 68)
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    print()
""")

code("""
show("Greedy Search", "매번 확률이 가장 높은 토큰 하나만 고른다",
     do_sample=False, num_beams=1)
""")

md("""
그리디는 **결정적(deterministic)** 입니다. 같은 입력이면 항상 같은 출력이 나옵니다.
학습이 충분하면 꽤 그럴듯한 글을 씁니다. 대신 다양성이 없습니다.

학습이 부족하면 한 번 빠진 패턴에서 못 나와 같은 말만 반복하게 됩니다.
""")

code("""
show("Beam Search", "여러 후보 경로를 동시에 추적해 전체 확률이 가장 높은 것을 고른다",
     do_sample=False, num_beams=5, early_stopping=True)
""")

md("""
### 빔서치가 오히려 더 반복하는 것을 보셨나요?

직관과 어긋나 보입니다. 후보를 여러 개 놓고 고르는데 왜 더 나쁠까요.

빔서치는 **전체 확률이 가장 높은 문장**을 찾습니다. 그런데 언어 모델에서
확률이 높은 문장은 대개 **짧고 안전하고 반복적인** 문장입니다.
`그들은 함께 놀았어요` 를 계속 이어붙이는 것이, 새로운 전개를 만드는 것보다
확률상으로는 "더 좋은" 선택이거든요.

빔서치는 번역이나 요약처럼 **정답이 정해진** 작업에는 잘 맞습니다.
반면 이야기 생성처럼 **열린 작업**에서는 이 성질이 단점이 됩니다.

그래서 열린 생성에는 **확률에서 무작위로 뽑는** 방식을 씁니다.
""")

code("""
show("Random Sampling", "확률 분포 그대로 무작위로 뽑는다",
     do_sample=True, top_k=0, top_p=1.0, temperature=1.0)
""")

md("""
무작위 추출은 반복이 사라지고 이야기가 살아납니다.
대신 **확률이 아주 낮은 엉뚱한 토큰**도 가끔 뽑혀서 문맥이 튈 수 있습니다.
그래서 후보를 적당히 잘라내는 방법이 나왔습니다.

### 글자가 깨져 나온다면

`깜짝�어` 처럼 글자가 깨질 수 있습니다. 버그가 아니라 **ByteLevel BPE 의 구조** 때문입니다.

ByteLevel 은 텍스트를 **바이트 단위**로 쪼갭니다. 영어는 한 글자가 1바이트라 문제가 없지만
**한글은 한 글자가 3바이트**입니다. 확률이 낮은 토큰까지 뽑다 보면
글자 중간의 바이트만 골라서 완성되지 않은 글자가 나옵니다.

두 가지 조건에서 잘 나타납니다.

- **학습이 부족할 때** — 바이트 조합 규칙을 아직 덜 익혔습니다
- **temperature 가 높을 때** — 낮은 확률 토큰까지 뽑기 때문입니다

위의 `temperature=1.5` 출력과 `temperature=0.5` 출력을 비교해보면 차이가 보입니다.
""")

code("""
show("Top-K Sampling", "확률 상위 K개 중에서만 무작위로 뽑는다",
     do_sample=True, top_k=10)
""")

code("""
show("Top-P (Nucleus)", "확률을 더해 P가 될 때까지의 후보 중에서만 뽑는다",
     do_sample=True, top_k=0, top_p=0.9)
""")

md("""
### Top-K 와 Top-P 의 차이

Top-K 는 **개수**를 고정합니다. 그런데 상황에 따라 적절한 개수가 다릅니다.
확신이 강할 때는 1~2개면 충분하고, 애매할 때는 20개도 부족합니다.

Top-P 는 **확률 합**을 기준으로 후보를 자릅니다.
모델이 확신하면 후보가 저절로 줄고, 애매하면 늘어납니다.
그래서 실무에서는 Top-P 를 더 많이 씁니다.
""")

code("""
# temperature 로 분포를 날카롭게/부드럽게 조절할 수도 있다.
for t in [0.5, 1.0, 1.5]:
    show(f"Top-P (temperature={t})",
         "낮으면 보수적, 높으면 과감해진다",
         do_sample=True, top_k=0, top_p=0.9, temperature=t)
""")

# =====================================================================
# 6. Continuous Pre-training (커리큘럼 3단원 ⑩)
# =====================================================================
md("""
## 6. 이어서 학습시키기 — Continuous Pre-training

지금 이 모델은 **동화만** 압니다. 동화 데이터로만 학습했으니 당연합니다.

여기에 다른 도메인을 가르치고 싶다면 어떻게 할까요. 처음부터 다시 학습시키는 것은
너무 비쌉니다. 그래서 **이미 학습된 모델에 새 데이터를 이어서 학습**시킵니다.
이것을 Continuous Pre-training(CPT) 이라고 합니다.

한국어 LLM 은 대부분 이 방식으로 만들어졌습니다. `llama-2-ko` 는 Llama-2 에,
`EEVE-Korean` 은 SOLAR 에 한국어를 이어 학습시킨 것입니다.

**말은 간단한데 실제로는 까다롭습니다.** 바로 앞 노트북(`HPC_데이터처리실습`)에서
정제한 위키 코퍼스를 넣어보면 무슨 일이 벌어지는지 보입니다.
""")

md(DATA_DIR_MD)
code(DATA_DIR_CODE)

code("""
import json

wiki_path = DATA_DIR / "ko_wiki_clean.jsonl"
if not wiki_path.exists():
    raise FileNotFoundError(
        f"{wiki_path} 가 없습니다.\\n"
        "  이 파일은 앞 노트북 HPC_데이터처리실습 이 만듭니다.\\n"
        "  2일차 순서대로 그 노트북을 먼저 실행해 주세요."
    )

wiki_docs = [json.loads(l) for l in wiki_path.open(encoding="utf-8")]
print(f"위키 정제 코퍼스 {len(wiki_docs):,}건")
print(f"  예시: [{wiki_docs[0]['title']}] {wiki_docs[0]['text'][:80]!r}")
""")

# ---------------------------------------------------------------------
md("""
### 6-1. 첫 번째 문제 — 토크나이저가 안 맞는다

우리 토크나이저는 **동화로 학습시킨 것**입니다. 사전 크기도 5,000 뿐입니다.
동화에 안 나오는 말(한자, 학술 용어, 연도, 괄호 안 원어)은 사전에 없으니
잘게 쪼개집니다.

같은 한 글자를 표현하는 데 토큰이 몇 개나 드는지 재보면 바로 보입니다.
""")

code("""
def tokens_per_char(texts):
    t = sum(len(tokenizer.encode(x)) for x in texts)
    c = sum(len(x) for x in texts)
    return t / c

story_sample = [train_docs[i][COL] for i in range(200)]
wiki_sample  = [d["text"][:2000] for d in wiki_docs[:200]]

tpc_story = tokens_per_char(story_sample)
tpc_wiki  = tokens_per_char(wiki_sample)

print(f"동화  {tpc_story:.3f} 토큰/글자")
print(f"위키  {tpc_wiki:.3f} 토큰/글자   ← {tpc_wiki/tpc_story:.1f}배")
print()
print("같은 분량의 글을 넣어도 위키 쪽이 토큰을 훨씬 많이 먹습니다.")
print("학습 비용이 그만큼 늘고, 컨텍스트에 담기는 내용은 줄어듭니다.")
print()
print("── 실제로 어떻게 쪼개지는지 ──")
for s in ["옛날 옛적에 작은 고양이가 살았어요.",
          "1919년 3월 1일 경성부에서 만세운동이 일어났다."]:
    ids = tokenizer.encode(s)
    print(f"  {s}")
    print(f"    {len(ids)}토큰: {[tokenizer.decode([i]) for i in ids][:18]}")
""")

md("""
> **그래서 실무에서는 어휘 확장(vocab expansion)을 합니다.** `llama-2-ko` 는 사전을
> 32,000 → 46,336 으로, `EEVE-Korean` 은 32,000 → 40,960 으로 늘렸습니다.
>
> 다만 어휘 확장은 **처음에 성능이 오히려 떨어졌다가 수십억 토큰을 학습해야 회복**됩니다.
> 이 실습은 몇 분짜리라 회복 구간에 도달할 수 없어서, 여기서는 확장하지 않고
> **불일치가 어떤 문제인지 보는 데까지만** 갑니다.
> 최신 사례인 `Llama-3-Open-Ko` 도 확장 없이 갔습니다 — 원 토크나이저가 이미 충분히
> 컸기 때문입니다. 늘리는 것만이 답은 아닙니다.
""")

# ---------------------------------------------------------------------
md("""
### 6-2. 학습 전 상태를 붙잡아 둔다

이어학습은 **모델을 제자리에서 바꿉니다.** 한 번 돌리면 지금의 모델은 사라집니다.
비교하려면 미리 복사해 두어야 합니다.

그리고 비교는 **같은 프롬프트, 같은 난수**로 해야 합니다. 안 그러면 달라진 것이
학습 때문인지 샘플링 운 때문인지 알 수 없습니다.
""")

code("""
import copy

# 2백만 파라미터라 통째로 복사해도 부담이 없다. 되돌릴 수 있게 해둔다.
model_before = copy.deepcopy(model)

CPT_PROMPTS = ["옛날 옛적에 작은 고양이가", "톰과 지미는 공원에서"]

def sample_stories(m):
    \"\"\"같은 프롬프트·같은 시드로 생성한다. 전후 비교용이라 재현성이 핵심이다.\"\"\"
    dev = next(m.parameters()).device
    out = {}
    for p in CPT_PROMPTS:
        torch.manual_seed(0)          # ★ 없으면 비교가 무의미해진다
        ids = tokenizer(p, return_tensors="pt").input_ids.to(dev)
        g = m.generate(ids, max_new_tokens=60, do_sample=True, top_p=0.9,
                       pad_token_id=tokenizer.pad_token_id, use_cache=False)
        out[p] = tokenizer.decode(g[0], skip_special_tokens=True)
    return out

def story_ppl(m):
    \"\"\"동화 평가셋에 대한 perplexity. '동화를 얼마나 잊었는가' 의 지표다.\"\"\"
    t = Trainer(model=m, args=TrainingArguments(
        output_dir="tmp_eval", per_device_eval_batch_size=BATCH_SIZE,
        report_to="none", bf16=torch.cuda.is_bf16_supported()))
    return math.exp(t.evaluate(eval_dataset=lm_eval)["eval_loss"])

before_txt = sample_stories(model_before)
before_ppl = story_ppl(model_before)

print(f"이어학습 전 동화 Perplexity: {before_ppl:.2f}\\n")
for p, t in before_txt.items():
    print(f"[{p}] {t}\\n")
""")

# ---------------------------------------------------------------------
md("""
### 6-3. 위키 데이터 준비

위키 문서는 수만 자짜리가 흔합니다. 그대로 넣으면 토큰화·청킹만으로 시간이 다 갑니다.
**앞부분만 잘라서** 씁니다.
""")

code("""
CPT_DOCS  = 2000     # 쓸 문서 수
CPT_CHARS = 2000     # 문서당 앞부분만. 자르지 않으면 블록이 수십만 개가 된다
CPT_STEPS = 300      # max_steps 로 고정. epoch 로 하면 데이터 크기에 따라 시간이 요동친다
CPT_LR    = 5e-5     # 본학습 LR(5e-4) 의 1/10. 근거는 아래에서 설명한다
""")

code("""
from datasets import Dataset, concatenate_datasets

wiki_raw = Dataset.from_dict(
    {"text": [d["text"][:CPT_CHARS] for d in wiki_docs[:CPT_DOCS]]}
)

def wiki_tokenize(batch):
    return {"ids": [tokenizer.encode(t) for t in batch["text"]]}

# remove_columns 를 반드시 준다. 위키에는 id/url/title 이 함께 오는데,
# 남겨두면 chunk 가 만드는 블록 수와 길이가 안 맞아 pyarrow 에러가 난다.
wiki_tok = wiki_raw.map(wiki_tokenize, batched=True,
                        remove_columns=wiki_raw.column_names, desc="토큰화(위키)")
wiki_blocks = wiki_tok.map(chunk, batched=True, batch_size=1000,
                           remove_columns=["ids"], desc="청킹(위키)")

# 위키 평가셋은 학습에 안 쓴 뒤쪽에서 뗀다
split = wiki_blocks.train_test_split(test_size=0.05, seed=0)
wiki_train, wiki_eval = split["train"], split["test"]

print(f"위키 학습 블록 {len(wiki_train):,}개 / 평가 블록 {len(wiki_eval):,}개")
print(f"동화 학습 블록 {len(lm_train):,}개 (replay 용으로 재사용)")
""")

# ---------------------------------------------------------------------
md("""
### 6-4. 두 번째 문제 — 배운 것을 잊는다

새 데이터만 계속 넣으면 모델은 **원래 알던 것을 잊습니다.**
이것을 치명적 망각(catastrophic forgetting) 이라고 합니다.

대응은 의외로 단순합니다. **원래 데이터를 조금 섞어서 같이 학습**시키면 됩니다.
이걸 replay 라고 합니다.

얼마나 섞어야 할까요. 문헌에 실측이 있습니다
(Ibrahim et al. 2024, *Simple and Scalable Strategies to Continually Pre-train LLMs*).

| replay 비율 | 결과 |
|---:|---|
| 1% | 이것만으로도 망각이 **유의미하게** 줄어든다 |
| 5% | 도메인이 비슷할 때 권장 |
| **25%** | 도메인이 크게 다를 때 권장 (영어→독일어 실험 기준) |
| 50% | 너무 많다. **새 도메인 적응이 나빠진다** |

우리 경우는 동화 → 백과사전입니다. 주제도 문체도 완전히 다르니 **강한 이동**입니다.
0% / 5% / 25% 를 직접 돌려서 비교해 봅시다.
""")

code("""
def make_mixed(replay_ratio, total_blocks):
    \"\"\"위키 + 동화(replay) 를 섞어 학습셋을 만든다.\"\"\"
    n_replay = int(total_blocks * replay_ratio)
    n_wiki   = total_blocks - n_replay
    parts = [wiki_train.shuffle(seed=0).select(range(min(n_wiki, len(wiki_train))))]
    if n_replay:
        parts.append(lm_train.shuffle(seed=0).select(range(min(n_replay, len(lm_train)))))
    return concatenate_datasets(parts).shuffle(seed=0)

# max_steps 만큼 돌 수 있는 분량이면 충분하다
NEED = CPT_STEPS * BATCH_SIZE
print(f"필요 블록 {NEED:,}개 (= {CPT_STEPS}스텝 x 배치 {BATCH_SIZE})")
for r in [0.0, 0.05, 0.25]:
    ds = make_mixed(r, NEED)
    print(f"  replay {r:>5.0%} → 총 {len(ds):,}블록")
""")

md("""
#### learning rate 를 왜 낮추는가

본학습은 `5e-4` 였습니다. 이어학습은 `5e-5` — **1/10** 로 낮춥니다.

임의로 정한 값이 아닙니다. 공개된 한국어 CPT 레시피가 그렇습니다.

| 모델 | CPT learning rate |
|---|---|
| EEVE-Korean-10.8B | `4e-5` |
| llama-2-ko-7b | `1e-5` |

원 사전학습 LR 의 **1/10 ~ 1/30** 수준입니다. 위 논문도 같은 방향을 확인했습니다 —
**LR 을 낮추면 망각이 줄고, 높이면 새 도메인에 빨리 적응한다.** 트레이드오프입니다.

> 참고로 같은 논문이 warmup 길이(0 / 0.5 / 1 / 2%)도 비교했는데 **망각에도 적응에도
> 영향이 없었습니다.** warmup 은 튜닝할 대상이 아닙니다. 여기서는 아예 주지 않습니다 —
> transformers 5 에서 `warmup_ratio` 인자 자체가 사라지기도 했습니다.
""")

code("""
def run_cpt(replay_ratio):
    \"\"\"학습 전 모델 사본에서 시작해 이어학습하고, 동화/위키 perplexity 를 잰다.

    ★ 매번 model_before 를 복사해서 시작한다. 같은 모델을 이어서 쓰면
      앞 실험의 영향이 누적되어 replay 비율 비교가 무의미해진다.
    \"\"\"
    m = copy.deepcopy(model_before)
    ds = make_mixed(replay_ratio, NEED)

    cpt_args = TrainingArguments(
        output_dir=f"minigpt_cpt_{int(replay_ratio*100)}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        max_steps=CPT_STEPS,          # epoch 대신 스텝으로 고정
        learning_rate=CPT_LR,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},   # 최저 LR = 최고의 10%
        # warmup 은 주지 않는다. transformers 5 에서 warmup_ratio 인자가 제거됐고,
        # 애초에 문헌상 warmup 길이는 망각에도 적응에도 영향이 없다(아래 설명).
        logging_steps=10,             # 기본 100 이면 300스텝에 점이 3개뿐이다
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    )
    t = Trainer(model=m, args=cpt_args, train_dataset=ds,
                processing_class=tokenizer)
    t.train()

    return {
        "model": m,
        "replay": replay_ratio,
        "동화_ppl": math.exp(t.evaluate(eval_dataset=lm_eval)["eval_loss"]),
        "위키_ppl": math.exp(t.evaluate(eval_dataset=wiki_eval)["eval_loss"]),
    }
""")

code("""
results = [run_cpt(r) for r in [0.0, 0.05, 0.25]]
""")

# ---------------------------------------------------------------------
md("""
### 6-5. 결과

**숫자를 먼저 봅니다.** 생성 결과만 보고 판단하려 하면 안 됩니다 —
이 모델은 2백만 파라미터짜리라 원래도 출력이 그리 좋지 않아서,
"망가졌다" 를 눈으로 구분하기 어렵습니다. Perplexity 는 그렇지 않습니다.
""")

code("""
print(f"{'replay':>8}{'동화 PPL':>12}{'(전 대비)':>12}{'위키 PPL':>12}")
print("-" * 46)
print(f"{'학습 전':>8}{before_ppl:>12.2f}{'—':>12}{'—':>12}")
for r in results:
    ratio = r["동화_ppl"] / before_ppl
    print(f"{r['replay']:>7.0%}{r['동화_ppl']:>12.2f}{ratio:>11.1f}x{r['위키_ppl']:>12.2f}")

print()
print("읽는 법")
print("  동화 PPL 이 올라간 정도 = 잊어버린 정도")
print("  위키 PPL 이 낮을수록   = 새 도메인을 잘 배운 것")
print("  replay 를 늘리면 앞은 좋아지고 뒤는 나빠진다. 그 사이를 고르는 것이다.")
""")

code("""
# replay 0% 모델이 동화를 어떻게 쓰는지 — 학습 전과 같은 프롬프트, 같은 시드
after_txt = sample_stories(results[0]["model"])

for p in CPT_PROMPTS:
    print("=" * 70)
    print(f"프롬프트: {p}")
    print("=" * 70)
    print(f"  [이어학습 전]   {before_txt[p]}")
    print()
    print(f"  [replay 0% 후]  {after_txt[p]}")
    print()
""")

md("""
#### 무엇을 봤나

- **동화 Perplexity 가 뛰었습니다.** replay 없이 위키만 학습시키면 동화를 잊습니다.
  숫자로 몇 배가 올랐는지 확인하세요.
- **replay 를 조금만 섞어도 상당히 돌아옵니다.** 5% 와 0% 의 차이를 보세요.
  논문이 "1% 만으로도 유의미하다" 고 한 이유입니다.
- **25% 는 동화를 거의 지킵니다.** 대신 위키 Perplexity 는 0% 보다 높습니다.
  공짜가 아닙니다.
- 생성 결과에도 **문체가 섞이는 것**이 보입니다. 동화를 쓰다가 백과사전 말투가
  튀어나옵니다.

> 실제 한국어 LLM 을 만들 때도 정확히 이 문제를 다룹니다. 다만 규모가 달라서
> 위키 몇천 건이 아니라 **수십 GB** 를 넣고, 몇 분이 아니라 **몇 주**를 돌립니다.
> 원리와 손잡이(LR, replay 비율, 어휘 확장)는 방금 만진 것과 같습니다.
""")

code("""
# 원래 모델로 되돌린다. 위쪽 디코딩 전략 셀들을 다시 돌려보려면 이게 필요하다.
model = model_before
print("model 을 이어학습 전 상태로 되돌렸습니다.")
""")

# ---------------------------------------------------------------------
md("""
## 마무리

밑바닥부터 GPT 를 만들어 학습시키고 텍스트를 생성해봤습니다.

- **토크나이저**를 데이터로부터 직접 학습시켰습니다
- **Causal Self-Attention** 을 직접 구현하며 뒤를 가리는 마스킹을 봤습니다
- **다음 토큰 맞히기** 하나로 사전학습이 이뤄지는 것을 확인했습니다
- **디코딩 전략**에 따라 같은 모델이 다른 글을 쓰는 것을 봤습니다
- **이어학습(CPT)** 으로 새 도메인을 가르쳤고, 그 대가로 치른 것 —
  토크나이저 불일치와 치명적 망각 — 그리고 **replay 로 막는 법**을 봤습니다

여기서 만든 모델은 2백만 파라미터 남짓입니다.
실제 LLM 은 같은 구조를 **수천 배 키우고** 훨씬 많은 데이터로 학습시킨 것입니다.
구조 자체는 방금 만든 것과 다르지 않습니다.

내일은 이 모델을 **쓸 만하게 만드는** 단계로 갑니다.
사전학습된 모델은 다음 토큰을 이어붙일 뿐 지시를 따르지는 못합니다.
그것을 가르치는 것이 Post-training 입니다.
""")


# =====================================================================
def build() -> dict:
    cells = []
    for kind, src in CELLS:
        lines = src.splitlines(keepends=True)
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
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
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    n_code = sum(1 for k, _ in CELLS if k == CODE)
    n_md = len(CELLS) - n_code
    print(f"생성: {OUT}")
    print(f"  셀 {len(CELLS)}개 (코드 {n_code} · 마크다운 {n_md})")
    print()
    print("주의: migrate_notebooks.py 는 이 파일을 건드리지 않습니다.")
    print("      MiniGPT 는 원본에서 변환하는 것이 아니라 새로 작성한 것이라,")
    print("      이 스크립트가 유일한 생성원입니다.")


if __name__ == "__main__":
    main()
