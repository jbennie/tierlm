#!/usr/bin/env python3
"""oracle_cmp.py - compare two per-position top-k dumps (P-121 gate).

Reads two JSONL dumps in the `tierlm oracle` / oracle_ref.py format:
    {"pos": i, "topk": [[id, score], ...]}
and, per position, compares:
  * argmax  = topk[0][0]   (the hard gate: must match at every position)
  * top-k id-set Jaccard  (a soft signal; 1.0 = identical top-k sets)

Score magnitudes are NOT compared: a 4-bit HF reference and the quantized
tierlm store differ in scale, but argmax and the top-k id set are
scale-invariant, so that is the right thing to gate on.

Exit 0 iff every position's argmax matches (and the position sets are equal).
Exit 1 otherwise, printing the first divergence.
"""
import json
import sys


def load(path):
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            d[o["pos"]] = [p[0] for p in o["topk"]]  # ids only, descending
    return d


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: oracle_cmp.py tierlm.jsonl ref.jsonl")
    tier = load(sys.argv[1])
    ref = load(sys.argv[2])
    positions = sorted(set(tier) | set(ref))
    n = len(positions)
    if n == 0:
        sys.exit("oracle_cmp: no positions in either dump")

    argmax_ok = 0
    jaccard_sum = 0.0
    missing = 0
    first = None
    for p in positions:
        t, r = tier.get(p), ref.get(p)
        if t is None or r is None:
            missing += 1
            if first is None:
                first = f"pos {p}: present in one dump only (tierlm={'yes' if t else 'no'}, ref={'yes' if r else 'no'})"
            continue
        if t[0] == r[0]:
            argmax_ok += 1
        elif first is None:
            first = f"pos {p}: argmax tierlm={t[0]} ref={r[0]}"
        a, b = set(t), set(r)
        union = a | b
        jaccard_sum += (len(a & b) / len(union)) if union else 1.0

    cmpn = n - missing
    print(f"positions          : {n}")
    print(f"argmax match       : {argmax_ok}/{cmpn} ({100.0 * argmax_ok / max(1, cmpn):.2f}%)")
    print(f"mean top-k Jaccard : {jaccard_sum / max(1, cmpn):.4f}")
    if missing:
        print(f"positions unmatched: {missing}")
    if first:
        print(f"FIRST DIFF         : {first}")
        sys.exit(1)
    print("RESULT: per-position argmax MATCH (P-121 gate pass)")


if __name__ == "__main__":
    main()
