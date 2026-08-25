# =====================================================================
# [VESSL 점검 4] 인코더 모델 비교 — NER F1 + NSMC 정확도
#
#   붙여넣어 한 번에 실행합니다. L40S 기준 6~10분.
#   출력 전체를 그대로 회신해 주세요.
#
# 왜 필요한가
# -----------
# 03_수정본검증.py 에서 mmBERT-base 의 NSMC 정확도가 0.8545 로 나왔습니다.
# 기준선은 넘었지만 여유가 없고, 무엇보다 **한국어 토큰화가 거의 음절 단위**입니다.
#     '안녕하세요' → ' 안','녕','하세요'      '홍길동' → ' 홍','길','동'
# NER 은 토큰 경계가 성능에 직결되므로 분류보다 타격이 클 수 있습니다.
# 숫자 하나만으로는 판단할 수 없어 **동일 조건 비교**로 확정합니다.
#
# 비교 대상 — 둘 다 상업 이용 가능하고 gated 가 아닙니다
#   A. jhu-clsp/mmBERT-base                       MIT          현재 선택, 다국어
#   B. monologg/koelectra-base-v3-discriminator   Apache-2.0   한국어 네이티브
#
# 판단 기준
#   NER F1 차이가 크면(예: 0.03 이상) 한국어 네이티브로 교체합니다.
#   비슷하면 mmBERT 를 유지합니다(다국어라 영어 예시도 다룰 수 있는 이점).
# =====================================================================
import json

import numpy as np
import torch

CANDIDATES = {
    "mmBERT-base (MIT, 다국어)": "jhu-clsp/mmBERT-base",
    "KoELECTRA-v3 (Apache-2.0, 한국어)": "monologg/koelectra-base-v3-discriminator",
}

# 두 모델에 완전히 동일하게 적용한다. 비교의 공정성이 이 스크립트의 핵심이다.
SEED = 42
NER_STEPS = 600
CLS_STEPS = 500
LR = 3e-5
RESULT = {}


def hr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


# ---------------------------------------------------------------------
hr("0. 한국어 토큰화 효율 비교")
# ---------------------------------------------------------------------
from transformers import AutoTokenizer  # noqa: E402

SAMPLES = [
    "안녕하세요, 반갑습니다.",
    "홍길동은 서울에 산다",
    "이 영화는 정말 재미있었고 배우들의 연기도 훌륭했다",
    "2026년 8월 25일 한국은행이 기준금리를 발표했다",
]
toks = {}
for label, mid in CANDIDATES.items():
    tk = AutoTokenizer.from_pretrained(mid)
    toks[mid] = tk
    total = sum(len(tk.encode(s)) for s in SAMPLES)
    RESULT[f"tok_{mid}"] = total
    print(f"\n[{label}]  vocab={tk.vocab_size:,}  총 {total}토큰")
    for s in SAMPLES[:2]:
        ids = tk.encode(s)
        print(f"    {s}")
        print(f"      → {len(ids):>2}토큰 {[tk.decode([i]) for i in ids]}")

base = RESULT[f"tok_{CANDIDATES['mmBERT-base (MIT, 다국어)']}"]
ko = RESULT[f"tok_{CANDIDATES['KoELECTRA-v3 (Apache-2.0, 한국어)']}"]
print(f"\n  → mmBERT 가 KoELECTRA 대비 {base/ko:.2f}배 토큰을 씁니다"
      f" (같은 문장에 {base} vs {ko})")


# ---------------------------------------------------------------------
def run_ner(mid: str):
    """klue/klue ner 로 학습 후 seqeval F1 측정."""
    import evaluate
    from datasets import load_dataset
    from transformers import (AutoModelForTokenClassification,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments)

    tk = toks[mid]
    ds = load_dataset("klue/klue", "ner")
    names = ds["train"].features["ner_tags"].feature.names

    ds["train"] = ds["train"].shuffle(seed=SEED).select(range(8000))
    ds["validation"] = ds["validation"].shuffle(seed=SEED).select(range(2000))

    def align(labels, word_ids):
        out, cur = [], None
        for w in word_ids:
            if w != cur:
                cur = w
                out.append(-100 if w is None else labels[w])
            elif w is None:
                out.append(-100)
            else:
                lab = labels[w]
                if lab % 2 == 1:      # B-XXX → I-XXX
                    lab += 1
                out.append(lab)
        return out

    def prep(ex):
        enc = tk(ex["tokens"], truncation=True, is_split_into_words=True, max_length=256)
        enc["labels"] = [align(l, enc.word_ids(i)) for i, l in enumerate(ex["ner_tags"])]
        return enc

    tds = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)
    metric = evaluate.load("seqeval")

    def metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        tp, tl = [], []
        for pr, la in zip(preds, labels):
            tp.append([names[a] for a, b in zip(pr, la) if b != -100])
            tl.append([names[b] for b in la if b != -100])
        r = metric.compute(predictions=tp, references=tl)
        return {"f1": r["overall_f1"], "precision": r["overall_precision"],
                "recall": r["overall_recall"], "accuracy": r["overall_accuracy"]}

    m = AutoModelForTokenClassification.from_pretrained(mid, num_labels=len(names))
    args = TrainingArguments(
        output_dir=f"/tmp/ner_{mid.split('/')[-1]}", learning_rate=LR,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        max_steps=NER_STEPS, eval_strategy="no", save_strategy="no",
        report_to="none", bf16=torch.cuda.is_bf16_supported(),
        logging_steps=200, seed=SEED,
    )
    tr = Trainer(model=m, args=args, train_dataset=tds["train"],
                 eval_dataset=tds["validation"], processing_class=tk,
                 data_collator=DataCollatorForTokenClassification(tokenizer=tk),
                 compute_metrics=metrics)
    tr.train()
    return tr.evaluate()


def run_cls(mid: str):
    """NSMC 감정분류 정확도 측정 (03 과 동일 조건)."""
    import evaluate
    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification,
                              DataCollatorWithPadding, Trainer, TrainingArguments)

    tk = toks[mid]
    ds = load_dataset("e9t/nsmc", revision="refs/convert/parquet")
    ds["train"] = ds["train"].shuffle(seed=SEED).select(range(8000))
    ds["test"] = ds["test"].shuffle(seed=SEED).select(range(2000))
    tds = ds.map(lambda b: tk(b["document"], truncation=True, max_length=128),
                 batched=True)
    acc = evaluate.load("accuracy")

    m = AutoModelForSequenceClassification.from_pretrained(mid, num_labels=2)
    args = TrainingArguments(
        output_dir=f"/tmp/cls_{mid.split('/')[-1]}", learning_rate=LR,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        max_steps=CLS_STEPS, eval_strategy="no", save_strategy="no",
        report_to="none", bf16=torch.cuda.is_bf16_supported(),
        logging_steps=200, seed=SEED,
    )
    tr = Trainer(model=m, args=args, train_dataset=tds["train"],
                 eval_dataset=tds["test"], processing_class=tk,
                 data_collator=DataCollatorWithPadding(tokenizer=tk),
                 compute_metrics=lambda p: acc.compute(
                     predictions=np.argmax(p[0], axis=1), references=p[1]))
    tr.train()
    return tr.evaluate()


# ---------------------------------------------------------------------
for label, mid in CANDIDATES.items():
    hr(f"NER (KLUE-NER, {NER_STEPS}스텝) — {label}")
    try:
        r = run_ner(mid)
        RESULT[f"ner_{mid}"] = r
        print(f"  ★ F1={r['eval_f1']:.4f}  P={r['eval_precision']:.4f}  "
              f"R={r['eval_recall']:.4f}  acc={r['eval_accuracy']:.4f}")
    except Exception as e:  # noqa: BLE001
        RESULT[f"ner_{mid}"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        print(f"  [FAIL] {type(e).__name__}: {e}")

    hr(f"NSMC 분류 ({CLS_STEPS}스텝) — {label}")
    try:
        r = run_cls(mid)
        RESULT[f"cls_{mid}"] = r
        print(f"  ★ accuracy={r['eval_accuracy']:.4f}")
    except Exception as e:  # noqa: BLE001
        RESULT[f"cls_{mid}"] = {"error": f"{type(e).__name__}: {e}"[:500]}
        print(f"  [FAIL] {type(e).__name__}: {e}")

# ---------------------------------------------------------------------
hr("최종 비교")
# ---------------------------------------------------------------------
print(f"{'모델':<40} {'NER F1':>9} {'NSMC acc':>10} {'토큰수':>8}")
print("-" * 72)
rows = {}
for label, mid in CANDIDATES.items():
    n = RESULT.get(f"ner_{mid}", {})
    c = RESULT.get(f"cls_{mid}", {})
    f1 = n.get("eval_f1")
    ac = c.get("eval_accuracy")
    rows[label] = (f1, ac)
    print(f"{label:<40} {f1 if f1 is None else f'{f1:.4f}':>9} "
          f"{ac if ac is None else f'{ac:.4f}':>10} {RESULT.get(f'tok_{mid}'):>8}")

vals = [v[0] for v in rows.values() if v[0] is not None]
if len(vals) == 2:
    d = vals[1] - vals[0]
    print(f"\n  NER F1 차이 (KoELECTRA - mmBERT) = {d:+.4f}")
    if d >= 0.03:
        print("  → 한국어 네이티브가 뚜렷하게 낫습니다. KoELECTRA 로 교체를 권합니다.")
    elif d <= -0.03:
        print("  → mmBERT 가 뚜렷하게 낫습니다. 현행 유지.")
    else:
        print("  → 차이가 작습니다. mmBERT 유지해도 무방합니다.")

print("\n--- 아래 JSON 도 함께 회신해 주세요 ---")
print(json.dumps(RESULT, ensure_ascii=False, indent=1, default=str)[:9000])
