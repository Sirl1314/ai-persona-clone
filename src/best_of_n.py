#!/usr/bin/env python3
"""Best-of-N candidate reranking for a persona chat model.

The model is sampled N times per prompt. Each candidate gets a rule-based
score, and the highest scoring one is kept. The rules were reverse-engineered
from the training distribution (median reply length 11 chars in Chinese,
p90 = 31), not guessed.

Usage:
    python src/best_of_n.py \
        --adapter saves/qwen25-7b-lora \
        --input data/eval.jsonl \
        --output out/best_of_4.json \
        --n 4
"""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = "你是我本人，用短句、口语回复，想说的话可以分成多条，用换行分隔。"

# Phrases that show the model fell back to a generic opener instead of
# answering. Penalised because they dominate the sampled distribution.
LAZY_OPENERS = ("666", "我的发", "我的法", "我去", "稳了", "可以捏", "离谱", "哈哈", "？", "。。")
# Safety: never let an aggressive candidate win.
HARSH_WORDS = ("砍死", "我恨", "去死", "杀", "打死")


def score_candidate(prompt: str, reply: str) -> tuple[float, list[str]]:
    """Return (score, reasons). Higher is better."""
    s = 0.0
    reasons = []
    reply = reply.strip()
    lines = [x for x in reply.split("\n") if x.strip()]
    n = len(reply)

    # --- length: the target median is 11 chars, p90 is 31 ---
    if 5 <= n <= 40:
        s += 2
        reasons.append("len_ok")
    if n < 5 or n > 60:
        s -= 1
        reasons.append("len_bad")

    # --- opener check: generic fillers instead of an actual answer ---
    if reply.startswith(LAZY_OPENERS):
        s -= 2
        reasons.append("lazy_opener")

    # --- malformed output ---
    if "url" in reply.lower() or "'" in reply:
        s -= 6
        reasons.append("malformed")

    # --- segmentation: the real style is 1-4 short lines ---
    if 1 <= len(lines) <= 4:
        s += 1
        reasons.append("segmented_ok")
    elif len(lines) > 6:
        s -= 1
        reasons.append("too_many_lines")

    # --- topical overlap: does the reply share any Chinese 2-gram with the prompt ---
    pg = _zh_bigrams(prompt)
    rg = _zh_bigrams(reply)
    overlap = len(pg & rg)
    if overlap > 0:
        s += 3
        reasons.append(f"overlap_{overlap}")
    if len(prompt) >= 10 and overlap == 0:
        s -= 4
        reasons.append("zero_overlap_on_long_prompt")

    # --- very short reply to a long prompt is usually a dodge ---
    if len(reply) <= 3 and len(prompt) >= 12:
        s -= 2
        reasons.append("dodge")

    # --- safety ---
    if any(w in reply for w in HARSH_WORDS):
        s -= 3
        reasons.append("harsh")

    return s, reasons


_ZH = re.compile(r"[\u4e00-\u9fff]")


def _zh_bigrams(text: str) -> set[str]:
    chars = "".join(_ZH.findall(text))
    return {chars[i:i + 2] for i in range(len(chars) - 1)}


def load_model(base: str, adapter: str):
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, user_text: str) -> str:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def sample(model, tokenizer, prompt: str, n: int, quiet: bool = False) -> list[str]:
    ids = tokenizer(build_prompt(tokenizer, prompt), return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        do_sample=True,                 # greedy collapses into n-gram loops
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        max_new_tokens=64,
        num_return_sequences=n,
        pad_token_id=tokenizer.eos_token_id,
    )
    texts = tokenizer.batch_decode(out[:, ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return [t.strip() for t in texts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    model, tokenizer = load_model(args.base, args.adapter)

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    for i, row in enumerate(rows, 1):
        prompt = row["prompt"]
        cands = sample(model, tokenizer, prompt, args.n)
        scored = [(score_candidate(prompt, c), c) for c in cands]
        scored.sort(key=lambda x: x[0][0], reverse=True)
        best_score, best_reply = scored[0]
        results.append({
            "prompt": prompt,
            "reply": best_reply,
            "score": best_score[0],
            "reasons": best_score[1],
            "candidates": [{"reply": c, "score": sc[0], "reasons": sc[1]} for sc, c in scored],
        })
        print(f"[{i}/{len(rows)}] {prompt[:20]} => {best_reply[:40]} ({best_score[0]:+.0f})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(results)} rows -> {args.output}")


if __name__ == "__main__":
    main()
