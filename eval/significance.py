import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import read_jsonl, write_json


def event_structure(events):
    n_spans, n_facts, n_kept = [], [], []
    for ev in events:
        spans = ev["spans"]
        n_spans.append(len(spans))
        n_facts.append(sum(1 for s in spans if s.get("carries_fact")))
        n_kept.append(sum(1 for s in spans if s["kept"]))
    return np.array(n_spans), np.array(n_facts), np.array(n_kept)


def observed_facts_kept(events):
    return sum(1 for ev in events for s in ev["spans"]
               if s["kept"] and s.get("carries_fact"))


def permutation_test(events, n_iter=20000, seed=0):
    n, f, k = event_structure(events)
    keep = (n > 0) & (f > 0) & (k > 0) & (k <= n)
    n, f, k = n[keep], f[keep], k[keep]
    if not len(n):
        return None

    obs = observed_facts_kept(events)
    rng = np.random.default_rng(seed)
    draws = rng.hypergeometric(f, n - f, k, size=(n_iter, len(n))).sum(axis=1)

    null_mean = float(draws.mean())
    null_sd = float(draws.std(ddof=1))
    p = float(((draws >= obs).sum() + 1) / (n_iter + 1))
    sigma = (obs - null_mean) / null_sd if null_sd else float("nan")

    total_facts = int(f.sum())
    naive_p = float(k.sum()) / float(n.sum())
    naive_sd_counts = (total_facts * naive_p * (1 - naive_p)) ** 0.5
    naive_sigma = ((obs - total_facts * naive_p) / naive_sd_counts
                   if naive_sd_counts else float("nan"))

    return {
        "n_events": int(len(n)),
        "n_fact_spans": total_facts,
        "observed_facts_kept": int(obs),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "clustered_sigma": float(sigma),
        "naive_sigma": float(naive_sigma),
        "inflation": float(naive_sigma / sigma) if sigma else float("nan"),
        "p_one_sided": p,
        "n_iter": n_iter,
    }


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def main():
    ap = argparse.ArgumentParser(
        description="Clustered permutation test for salience, with Holm correction.")
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = args.labels or [Path(p).parent.name for p in args.events]
    results = []
    for path, label in zip(args.events, labels):
        r = permutation_test(read_jsonl(path), args.iters, args.seed)
        if r is None:
            print(f"{label}: no usable events")
            continue
        r["label"] = label
        results.append(r)

    if not results:
        return 1
    adj = holm([r["p_one_sided"] for r in results])
    for r, a in zip(results, adj):
        r["p_holm"] = a
        r["significant"] = a < args.alpha

    print(f"{'run':<44}{'obs':>6}{'null':>8}{'clust':>8}{'naive':>8}{'infl':>7}"
          f"{'p':>10}{'p_holm':>10}  verdict")
    for r in results:
        print(f"{r['label']:<44}{r['observed_facts_kept']:6d}{r['null_mean']:8.1f}"
              f"{r['clustered_sigma']:+8.2f}{r['naive_sigma']:+8.2f}{r['inflation']:7.2f}"
              f"{r['p_one_sided']:10.5f}{r['p_holm']:10.5f}  "
              f"{'SIGNIFICANT' if r['significant'] else 'not significant'}")

    print()
    print("clust = sigma against a null that permutes keep decisions within each event,")
    print("so it respects the fact that spans from one event share a keep budget.")
    print("naive = the binomial sigma that treats every span as independent.")
    print("infl  = how much the naive figure overstates the clustered one.")
    print(f"p_holm applies Holm-Bonferroni across the {len(results)} runs compared here.")

    if args.out:
        write_json(args.out, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
