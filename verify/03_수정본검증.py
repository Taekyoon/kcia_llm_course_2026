# =====================================================================
# [VESSL 점검 3] 마이그레이션 수정본 검증
#
#   붙여넣어 한 번에 실행합니다. 학습은 돌리지 않아 몇 분이면 끝납니다.
#   출력 전체를 그대로 회신해 주세요.
#
#   전제: 01_스택설치.py 를 이미 실행했을 것 (transformers 5.15 / datasets 5.0 등)
#
#   확인 대상 — 전부 work/notebook/ 수정본에 적용한 변경입니다.
#     A. 인코더 전환 (mmBERT-base) 이 분류·NER 에서 실제로 동작하는가
#     B. mmBERT-base 의 **한국어 성능** — 유일하게 남은 모델 선정 리스크
#     C. klue/klue ner 라벨 접근이 datasets 5.x 에서 되는가
#     D. TRL/transformers 신 API 로 학습이 1스텝 도는가
#     E. kullm-v2 스키마가 SFT 전처리와 맞는가
#     F. kowikitext parquet 직접 지정이 되는가
# =====================================================================
import json
import traceback

R = {}


def step(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def probe(key, fn):
    try:
        out = fn()
        R[key] = {"ok": True, "detail": str(out)[:400]}
        print(f"  [OK]   {key}: {str(out)[:260]}")
        return out
    except Exception as e:  # noqa: BLE001
        R[key] = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:700]}
        print(f"  [FAIL] {key}")
        print("         " + f"{type(e).__name__}: {e}"[:700].replace("\n", "\n         "))
        return None


ENCODER = "jhu-clsp/mmBERT-base"
TRAIN_LM = "Qwen/Qwen3-0.6B-Base"

# ---------------------------------------------------------------------
step("A. 인코더 모델 로드 — 분류·NER 전환의 전제")
# ---------------------------------------------------------------------
import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

tok = probe("enc_tokenizer", lambda: AutoTokenizer.from_pretrained(ENCODER))
probe("enc_is_fast", lambda: tok.is_fast)  # word_ids() 에 필요
probe("enc_seq_cls", lambda: AutoModelForSequenceClassification.from_pretrained(
    ENCODER, num_labels=2).config.architectures)
probe("enc_tok_cls", lambda: AutoModelForTokenClassification.from_pretrained(
    ENCODER, num_labels=13).config.architectures)

print("\n--- 한국어 토큰화 확인 (다국어 모델이라 한국어 분절이 관건) ---")
if tok:
    for s in ["안녕하세요, 반갑습니다.", "저 영화 짱이다.", "홍길동은 서울에 산다"]:
        ids = tok.encode(s)
        print(f"  {s!r}\n    → {len(ids)}토큰 {[tok.decode([i]) for i in ids][:16]}")

# ---------------------------------------------------------------------
step("B. mmBERT-base 한국어 성능 실측 ★ 남은 최대 리스크")
# ---------------------------------------------------------------------
print("mmBERT 는 1800+ 언어 다국어 모델이라 한국어 성능이 한국어 네이티브 모델보다")
print("낮을 수 있습니다. NSMC 로 소규모 학습을 돌려 실제 정확도를 확인합니다.")
print("(500 스텝, L40S 기준 2~4분)\n")


def bench_nsmc():
    import numpy as np
    from datasets import load_dataset
    from transformers import (DataCollatorWithPadding, Trainer, TrainingArguments)
    import evaluate

    ds = load_dataset("e9t/nsmc", revision="refs/convert/parquet")
    ds["train"] = ds["train"].shuffle(seed=42).select(range(8000))
    ds["test"] = ds["test"].shuffle(seed=42).select(range(2000))

    def prep(b):
        return tok(b["document"], truncation=True, max_length=128)

    tds = ds.map(prep, batched=True)
    acc = evaluate.load("accuracy")

    def metrics(p):
        return acc.compute(predictions=np.argmax(p[0], axis=1), references=p[1])

    m = AutoModelForSequenceClassification.from_pretrained(ENCODER, num_labels=2)
    args = TrainingArguments(
        output_dir="/tmp/nsmc_bench", learning_rate=3e-5,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        max_steps=500, eval_strategy="no", save_strategy="no",
        report_to="none", bf16=torch.cuda.is_bf16_supported(), logging_steps=100,
    )
    tr = Trainer(model=m, args=args, train_dataset=tds["train"],
                 eval_dataset=tds["test"], processing_class=tok,
                 data_collator=DataCollatorWithPadding(tokenizer=tok),
                 compute_metrics=metrics)
    tr.train()
    return tr.evaluate()


res = probe("nsmc_accuracy_mmbert", bench_nsmc)
if res and isinstance(res, dict):
    a = res.get("eval_accuracy", 0)
    print(f"\n  ★ NSMC 정확도: {a:.4f}")
    print("    참고: 한국어 네이티브 인코더는 보통 0.88~0.90 대입니다.")
    print("    0.85 미만이면 한국어 인코더 재탐색이 필요합니다.")

# ---------------------------------------------------------------------
step("C. klue/klue ner — 라벨 접근이 datasets 5.x 에서 되는가")
# ---------------------------------------------------------------------
from datasets import load_dataset  # noqa: E402

nd = probe("klue_ner_load", lambda: load_dataset("klue/klue", "ner"))
if nd:
    # 수정본 NER 노트북 셀 11-12 가 쓰는 접근 경로
    probe("klue_ner_feature_names",
          lambda: nd["train"].features["ner_tags"].feature.names)
    probe("klue_ner_splits", lambda: list(nd.keys()))

    if tok:
        def align_check():
            ex = nd["train"][0]
            enc = tok(ex["tokens"], truncation=True, is_split_into_words=True)
            wid = enc.word_ids(0) if hasattr(enc, "word_ids") else None
            return {"tokens": len(ex["tokens"]), "word_ids_사용가능": wid is not None}
        probe("klue_ner_word_ids", align_check)

# ---------------------------------------------------------------------
step("D. TRL 신 API — peft_config 경로로 SFT 1스텝")
# ---------------------------------------------------------------------
def sft_one_step():
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    data = Dataset.from_list([
        {"messages": [{"role": "user", "content": "안녕?"},
                      {"role": "assistant", "content": "안녕하세요!"}]}
    ] * 32)
    cfg = SFTConfig(output_dir="/tmp/sft_smoke", max_steps=1,
                    per_device_train_batch_size=2, report_to="none",
                    max_length=256, logging_steps=1)
    tr = SFTTrainer(model=TRAIN_LM, args=cfg, train_dataset=data,
                    peft_config=LoraConfig(r=8, lora_alpha=16,
                                           target_modules=["q_proj", "v_proj"]))
    tr.train()
    # 수정본이 쓰는 새 경로
    return {"processing_class": type(tr.processing_class).__name__,
            "tokenizer속성_존재": hasattr(tr, "tokenizer")}


probe("sft_1step_peft_config", sft_one_step)

# ---------------------------------------------------------------------
step("E. kullm-v2 스키마 — KoAlpaca 대체본이 SFT 전처리와 맞는가")
# ---------------------------------------------------------------------
k = probe("kullm_load", lambda: load_dataset("nlpai-lab/kullm-v2", split="train[:5]"))
if k:
    probe("kullm_columns", lambda: k.column_names)
    print("\n  첫 행 샘플:")
    row = k[0]
    for kk, vv in row.items():
        print(f"    {kk}: {str(vv)[:150]!r}")
    print("\n  ※ SFT 노트북 셀 7 build_messages() 는 KoAlpaca 의")
    print("     'instruction'/'output' 컬럼을 전제합니다. 위 컬럼명과 대조하세요.")

# ---------------------------------------------------------------------
step("F. kowikitext parquet 직접 지정 (BM25 RAG 수정본)")
# ---------------------------------------------------------------------
probe("kowikitext_parquet", lambda: load_dataset(
    "heegyu/kowikitext", data_files="kowikitext-20221001.parquet",
    split="train[:10]").column_names)
print("  실패하면 대체: load_dataset('wikimedia/wikipedia', '20231101.ko', split='train[:1000]')")
probe("wikipedia_ko_대체", lambda: load_dataset(
    "wikimedia/wikipedia", "20231101.ko", split="train[:10]").column_names)

# ---------------------------------------------------------------------
step("결과 요약")
# ---------------------------------------------------------------------
fails = [k for k, v in R.items() if isinstance(v, dict) and not v.get("ok")]
print(f"실패 {len(fails)}건:")
for f in fails:
    print(f"  - {f}")
acc = R.get("nsmc_accuracy_mmbert", {}).get("detail", "")
print(f"\n★ mmBERT 한국어 NSMC 결과: {acc[:200]}")
print("\n--- 아래 JSON 도 함께 회신해 주세요 ---")
print(json.dumps(R, ensure_ascii=False, indent=1)[:9000])
