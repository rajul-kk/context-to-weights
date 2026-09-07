import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import read_jsonl, write_json


def required_token_idx(row):
    req = row.get("required") or []
    if not req:
        return set()
    text = row["response"]
    hits = []
    for s in req:
        start = 0
        while True:
            c = text.find(s, start)
            if c < 0:
                break
            hits.append((c, c + len(s)))
            start = c + 1
    idx = set()
    for i, (a, b) in enumerate(row["offsets"]):
        if any(b > lo and a < hi for lo, hi in hits):
            idx.add(i)
    return idx


def units_for(row, granularity):
    if granularity == "token":
        return [{"token_idx": [i], "score": row["kl"][i]} for i in range(len(row["kl"]))]
    return [{"token_idx": s["token_idx"], "score": s["kl_mean"]} for s in row["spans"]]


def selected_idx(row, granularity, top_frac, rng=None):
    units = units_for(row, granularity)
    if not units:
        return set()
    n = len(row["kl"])
    k = max(1, round(len(units) * top_frac))
    ranked = sorted(range(len(units)), key=lambda i: -units[i]["score"])[:k]
    if rng is None:
        chosen = ranked
    else:
        budget = sum(len(units[i]["token_idx"]) for i in ranked)
        order = list(range(len(units)))
        rng.shuffle(order)
        chosen, taken = [], 0
        for i in order:
            size = len(units[i]["token_idx"])
            if taken + size > budget:
                continue
            chosen.append(i)
            taken += size
    return {i for c in chosen for i in units[c]["token_idx"] if 0 <= i < n}


def coverage(rows, granularity, top_frac, rng=None):
    hit = tot = active = total = 0
    for r in rows:
        sel = selected_idx(r, granularity, top_frac, rng)
        active += len(sel)
        total += len(r["kl"])
        req = required_token_idx(r)
        hit += len(req & sel)
        tot += len(req)
    return {
        "coverage": hit / tot if tot else float("nan"),
        "active_frac": active / total if total else 0.0,
        "n_required_tokens": tot,
    }


def control_coverage(rows, granularity, top_frac, trials=20, seed=0):
    runs = [coverage(rows, granularity, top_frac, random.Random(seed + t)) for t in range(trials)]
    cov = [r["coverage"] for r in runs]
    return {
        "coverage": statistics.fmean(cov),
        "sd": statistics.pstdev(cov) if len(cov) > 1 else 0.0,
        "active_frac": statistics.fmean(r["active_frac"] for r in runs),
        "trials": trials,
    }


def verdict_for(sigma):
    if sigma != sigma:
        return "no required tokens"
    if sigma >= 2.0:
        return "clears control"
    if sigma <= -2.0:
        return "below control"
    return "indistinguishable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--skills-root", default="skills/toy")
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = read_jsonl(args.scores)
    from kl_gate.skillset import load_all

    demos = {}
    for skill in load_all(args.skills_root):
        for i, d in enumerate(skill["demos"]):
            demos[(skill["name"], i)] = d
    for r in rows:
        r["required"] = demos.get((r["skill"], r["demo_idx"]), {}).get("required", [])

    all_kl = [k for r in rows for k in r["kl"]]
    gran = rows[0].get("granularity", "span") if rows else "span"
    gate = coverage(rows, gran, args.top_frac)
    ctrl = control_coverage(rows, gran, args.top_frac, seed=args.seed)
    sigma = (gate["coverage"] - ctrl["coverage"]) / ctrl["sd"] if ctrl["sd"] else float("nan")
    report = {
        "n_demos": len(rows),
        "n_tokens": len(all_kl),
        "granularity": gran,
        "kl_median": statistics.median(all_kl) if all_kl else float("nan"),
        "kl_mean": sum(all_kl) / len(all_kl) if all_kl else float("nan"),
        "kl_p90": statistics.quantiles(all_kl, n=10)[-1] if len(all_kl) > 10 else float("nan"),
        "top_frac": args.top_frac,
        "n_required_tokens": gate["n_required_tokens"],
        "required_coverage": gate["coverage"],
        "gate_active_frac": gate["active_frac"],
        "control_coverage": ctrl["coverage"],
        "control_active_frac": ctrl["active_frac"],
        "coverage_lift": gate["coverage"] / ctrl["coverage"] if ctrl["coverage"] else float("nan"),
        "coverage_sigma_over_control": sigma,
        "verdict": verdict_for(sigma),
    }
    for k, v in report.items():
        print(f"{k:<28} {v}")
    if report["verdict"] != "clears control":
        print(f"\nWARNING: the gate does not beat a matched-budget random control "
              f"({gate['coverage']:.3f} vs {ctrl['coverage']:.3f}). Training on it is "
              f"unlikely to beat the random-span baseline.")

    print("\nhighest-KL spans:")
    spans = [(s["kl_mean"], r["skill"], s["text"]) for r in rows for s in r["spans"]]
    spans.sort(reverse=True)
    for kl, skill, text in spans[: args.show * 3]:
        print(f"  {kl:7.3f}  [{skill}] {text.strip()[:90]!r}")

    print("\nlowest-KL spans:")
    for kl, skill, text in (spans[-args.show * 3:] if args.show else []):
        print(f"  {kl:7.3f}  [{skill}] {text.strip()[:90]!r}")

    print("\nper-demo token view:")
    for r in rows[: args.show]:
        pairs = sorted(zip(r["kl"], r["token_text"]), reverse=True)[:10]
        rendered = ", ".join(f"{t.strip()!r}:{k:.2f}" for k, t in pairs)
        print(f"  [{r['skill']} d{r['demo_idx']} {r['kind']}] {rendered}")

    if args.out:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
