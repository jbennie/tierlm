#!/usr/bin/env python3
"""oracle_ref.py - HF transformers teacher-forced per-position top-k dump.

The reference side of the P-121 oracle. Loads a Qwen3-MoE checkpoint
(4-bit on CPU via bitsandbytes, so it fits a single modest box) and, for a
fixed id sequence, runs ONE teacher-forced forward pass and dumps, per
position, the vocab top-k.

Convention must match `tierlm oracle` exactly: position i is the distribution
over the token following tokens[0..i] -- i.e. transformers'
model(input_ids).logits[i]. No BOS is prepended on either side; both feed the
identical raw ids.

Output: one JSON object per line,
    {"pos": i, "topk": [[id, score], ...]}
with topk[0] the argmax (descending). `score` is the raw pre-softmax logit;
the comparator (oracle_cmp.py) uses argmax and the top-k id set, both of
which are invariant to the raw-vs-logprob scale, so a 4-bit reference and the
quantized tierlm store stay comparable even though absolute magnitudes differ.
"""
import argparse
import json
import sys


def parse_tokens(s):
    return [int(x) for x in s.replace(",", " ").split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="HF model dir, e.g. ~/models/Qwen3-30B-A3B")
    ap.add_argument("--tokens", required=True, help="comma-separated teacher-forced ids")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--dtype", choices=["4bit", "bf16"], default="4bit",
                    help="reference weights: 4bit (bnb, bring-up proxy) or bf16 (strict gate)")
    args = ap.parse_args()

    ids = parse_tokens(args.tokens)
    if len(ids) < 2:
        sys.exit("oracle_ref: need at least 2 ids (position i needs the prefix t0..ti)")

    # Deferred: torch/transformers only exist in the oracle venv, not on a bare box.
    import torch
    from transformers import AutoModelForCausalLM

    if args.dtype == "bf16":
        # strict gate: full BF16 weights, no quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    else:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
    model.eval()

    inp = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=inp).logits[0]  # [seq, vocab], raw logits
    vocab = logits.shape[1]
    for bad in set(ids):
        if not (0 <= bad < vocab):
            sys.exit(f"oracle_ref: id {bad} out of range [0,{vocab})")
    k = min(args.topk, vocab)
    topk = logits.topk(k, dim=-1)  # values and indices, descending

    for i in range(len(ids)):
        idx = topk.indices[i].tolist()
        val = topk.values[i].tolist()
        print(json.dumps({"pos": i, "topk": [[int(idx[j]), float(val[j])] for j in range(k)]}))


if __name__ == "__main__":
    main()
