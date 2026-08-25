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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "notebook" / "1일차" / "HPC_MiniGPT실습.ipynb"

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
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    pad_token=PAD, bos_token=BOS, eos_token=EOS,
    model_max_length=SEQ_LEN,
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
그리디는 안전하지만 **같은 말을 반복하는** 경향이 있습니다.
매번 최선만 고르다 보니 한 번 빠진 패턴에서 못 나옵니다.
""")

code("""
show("Beam Search", "여러 후보 경로를 동시에 추적해 전체 확률이 가장 높은 것을 고른다",
     do_sample=False, num_beams=5, early_stopping=True)
""")

code("""
show("Random Sampling", "확률 분포 그대로 무작위로 뽑는다",
     do_sample=True, top_k=0, top_p=1.0, temperature=1.0)
""")

md("""
무작위 추출은 반복이 사라지지만, **확률이 아주 낮은 엉뚱한 토큰**도 가끔 뽑힙니다.
그래서 후보를 적당히 잘라내는 방법이 나왔습니다.

### 깨진 글자가 보이나요?

`노��죠` 처럼 글자가 깨져 나올 수 있습니다. 버그가 아니라 **ByteLevel BPE 의 구조** 때문입니다.

ByteLevel 은 텍스트를 **바이트 단위**로 먼저 쪼갭니다. 영어는 한 글자가 1바이트라 문제가 없지만,
**한글은 한 글자가 3바이트**입니다. 무작위로 뽑다 보면 글자 중간의 바이트만 골라서
완성되지 않은 글자가 나옵니다.

Top-K 나 Top-P 처럼 **후보를 좁히면 이 현상이 크게 줄어듭니다.**
모델이 "이 바이트 다음엔 저 바이트" 라는 규칙을 이미 배웠기 때문에,
확률이 높은 쪽만 고르면 대개 온전한 글자가 됩니다.

학습을 더 오래 시켜도 줄어듭니다. 바이트 조합 규칙을 더 확실하게 익히기 때문입니다.
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

# ---------------------------------------------------------------------
md("""
## 마무리

밑바닥부터 GPT 를 만들어 학습시키고 텍스트를 생성해봤습니다.

- **토크나이저**를 데이터로부터 직접 학습시켰습니다
- **Causal Self-Attention** 을 직접 구현하며 뒤를 가리는 마스킹을 봤습니다
- **다음 토큰 맞히기** 하나로 사전학습이 이뤄지는 것을 확인했습니다
- **디코딩 전략**에 따라 같은 모델이 다른 글을 쓰는 것을 봤습니다

여기서 만든 모델은 2백만 파라미터 남짓입니다.
실제 LLM 은 같은 구조를 **수천 배 키우고** 훨씬 많은 데이터로 학습시킨 것입니다.
구조 자체는 방금 만든 것과 다르지 않습니다.
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
