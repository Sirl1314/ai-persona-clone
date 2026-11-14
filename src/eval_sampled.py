#!/usr/bin/env python3
"""Sampled evaluation with the three statistics that actually matter.

Greedy decoding is deliberately NOT used here. On this model greedy decoding
collapses into n-gram loops on long replies (17/312 rows degenerated, one reply
repeated a filler 47 times). That looks like a broken model but is a decoding
artifact. Sampling removes it.

Reported metrics:
  1. degeneracy  - rows containing a 4+ times repeated fragment
  2. over-length - rows longer than 80 Chinese chars (real median is 11)
  3. verbatim    - fraction of rows whose 8-grams all appear in the training set
                   (this, not "same output twice", is the real memorisation test)

Usage:
    python src/eval_sampled.py --adapter saves/qwen25-7b-lora \
        --input data/eval.jsonl --train data/train_sharegpt.json \
        --output out/eval_report.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "你是微信里的我，用短句、口语回复，想说的话可以分成多条，用换行分隔。"


def degeneracy(reply: str) -> bool:
    """True if the same 2-8 char chunk repeats 4+ times."""
    for size in (2, 3, 4):
        chunks = [reply[i:i + size] for i in range(0, len(reply) - size + 1, size)]
        if not chunks:
            continue
        most = Counter(chunks).most_common(1)[0]
        if most[1] >= 4 and len(most[0].strip()) >= 2:
            return True
    return False


def ngrams(text: str, n: int = 8) -> set[str]:
    clean = re.sub(r"\s+", "", text)
    return {clean[i:i + n] for i in range(len(clean) - n + 1)}


def build_train_ngrams(train_path: str) -> set[str]:
    """Collect 8-grams from every assistant turn in the training set."""
    grams = set()
    data = json.loads(Path(train_path).read_text(encoding="utf-8"))
    for conv in data:
        for turn in conv.get("conversations", []):
            if turn.get("from") == "gpt":
                grams |= ngrams(turn.get("value", ""))
    return grams


@torch.inference_mode()
def generate(model, tokenizer, prompt: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        max_new_tokens=64,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--input", required=True, help="jsonl with a prompt field")
    ap.add_argument("--train", help="train_sharegpt.json, needed for the verbatim metric")
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    train_grams = build_train_ngrams(args.train) if args.train else set()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out = []
    for i, row in enumerate(rows, 1):
        reply = generate(model, tokenizer, row["prompt"])
        out.append({"prompt": row["prompt"], "reply": reply, "dim": row.get("dim", "")})
        print(f"[{i}/{len(rows)}] {row['prompt'][:18]} => {reply[:40]}")

    lengths = sorted(len(r["reply"]) for r in out)
    n = len(lengths)
    stats = {
        "rows": n,
        "len_avg": round(sum(lengths) / max(n, 1), 1),
        "len_median": lengths[n // 2] if n else 0,
        "len_p90": lengths[min(int(n * 0.9), n - 1)] if n else 0,
        "degenerate": sum(degeneracy(r["reply"]) for r in out),
        "over_80_chars": sum(len(r["reply"]) > 80 for r in out),
    }
    if train_grams:
        hits = 0
        for r in out:
            g = ngrams(r["reply"])
            if g and g <= train_grams:
                hits += 1
        stats["verbatim_8gram"] = hits
        stats["verbatim_ratio"] = round(hits / max(n, 1), 3)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps({"stats": stats, "rows": out}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
