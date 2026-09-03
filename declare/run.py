import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (ensure_dir, load_config, parse_overrides, read_jsonl, set_seed,
                       write_json, write_jsonl)
from common.schema import Trajectory
from declare.elicit import build_elicitor
from declare.regions import build_regions, gold_region, permute, region_tokens
from sleep.lm import load_backbone


def items_for(traj, n_regions, tokenizer):
    regions = build_regions(traj.turns, n_regions, tokenizer)
    facts = {f.key: f for f in traj.facts}
    out = []
    for probe in traj.probes:
        fact = facts.get(probe.fact_key)
        markers = [fact.statement] if fact else []
        gold = gold_region(regions, probe.answer, markers)
        if gold is None:
            continue
        out.append({"regions": regions, "question": probe.question,
                    "answer": probe.answer, "gold": gold})
    return out


def evaluate(elicitor, items, rng, shuffle_test=True):
    records = []
    for it in items:
        regions = it["regions"]
        n = len(regions)
        choice, raw = elicitor.declare(regions, it["question"])
        toks = region_tokens(regions)
        total = sum(toks) or 1

        rec = {
            "question": it["question"],
            "gold": it["gold"],
            "choice": choice,
            "hit": choice == it["gold"],
            "n_regions": n,
            "attended_frac": (toks[choice] / total) if choice is not None else 1.0,
            "raw": raw[:200],
        }

        if shuffle_test:
            permuted, order = permute(regions, rng)
            new_gold = order.index(it["gold"])
            pchoice, _ = elicitor.declare(permuted, it["question"])
            rec["shuffled_gold"] = new_gold
            rec["shuffled_choice"] = pchoice
            rec["shuffled_hit"] = pchoice == new_gold
            rec["slot_stable"] = (pchoice == choice)
        records.append(rec)
    return records


def summarize(records, label, n_regions):
    n = len(records)
    if not n:
        return {"label": label, "n": 0}
    hits = sum(int(r["hit"]) for r in records)
    shuffled = [r for r in records if "shuffled_hit" in r]
    gold_counts = Counter(r["gold"] for r in records)
    best_constant = max(gold_counts.values()) / n
    out = {
        "label": label,
        "n": n,
        "n_regions": n_regions,
        "hit_rate": hits / n,
        "random_control": 1.0 / n_regions,
        "best_constant_control": best_constant,
        "gold_distribution": {str(k): v for k, v in sorted(gold_counts.items())},
        "unparsed_rate": sum(1 for r in records if r["choice"] is None) / n,
        "mean_attended_frac": sum(r["attended_frac"] for r in records) / n,
    }
    if shuffled:
        out["shuffled_hit_rate"] = sum(int(r["shuffled_hit"]) for r in shuffled) / len(shuffled)
        out["slot_stable_rate"] = sum(int(r["slot_stable"]) for r in shuffled) / len(shuffled)
        out["content_dependence"] = out["shuffled_hit_rate"] - out["random_control"]
    counts = Counter(r["choice"] for r in records)
    top, top_n = counts.most_common(1)[0]
    out["modal_choice"] = top
    out["modal_share"] = top_n / n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/declare.yaml")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--modes", nargs="*", default=["generate", "read"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-shuffle-test", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    n_regions = cfg["declare"]["n_regions"]

    rows = read_jsonl(Path(cfg["data"]["dir"]) / cfg["data"][args.split])
    model, tokenizer = load_backbone(cfg)

    items = []
    for row in rows:
        items.extend(items_for(Trajectory.from_dict(row), n_regions, tokenizer))
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit("no probes had a locatable gold region")
    print(f"{len(items)} probes, {n_regions} regions each, model {cfg['model']['base']}")

    out_dir = ensure_dir(args.out or Path(cfg["run_root"]) / "declare")
    summaries = []
    for mode in args.modes:
        elicitor = build_elicitor(mode, model, tokenizer, cfg)
        records = evaluate(elicitor, items, rng, not args.no_shuffle_test)
        s = summarize(records, mode, n_regions)
        s["model"] = cfg["model"]["base"]
        s["elicitor_unparsed"] = elicitor.unparsed
        summaries.append(s)
        write_jsonl(out_dir / f"{mode}.records.jsonl", records)
        write_json(out_dir / f"{mode}.summary.json", s)
        print(f"\n== {mode}")
        for k in ("hit_rate", "random_control", "best_constant_control",
                  "shuffled_hit_rate", "content_dependence", "slot_stable_rate",
                  "modal_share", "unparsed_rate", "mean_attended_frac"):
            if k in s:
                print(f"  {k:<24} {s[k]:.3f}")
        print(f"  {'gold_distribution':<24} {s['gold_distribution']}")
        if s["hit_rate"] <= s["best_constant_control"]:
            print("  WARNING: no better than always naming one fixed region.")
        if s.get("slot_stable_rate", 0.0) >= 0.9:
            print("  WARNING: the declaration barely moves when region contents are shuffled,")
            print("  so it is naming a slot rather than reading content.")

    write_json(out_dir / "summary_all.json", summaries)
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
