import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (ensure_dir, load_config, parse_overrides, read_json, read_jsonl, set_seed,
                       write_json, write_jsonl)
from common.schema import Trajectory
from common.stats import verdict_for
from declare.elicit import build_elicitor
from declare.plan import SCHEMES, plan_attention
from declare.regions import build_regions, gold_region, permute, region_tokens, render
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


def make_layouts(items, rng, shuffle_test=True, balance=True):
    layouts = []
    for it in items:
        if balance:
            regions, order = permute(it["regions"], rng)
            gold = order.index(it["gold"])
        else:
            regions, gold = it["regions"], it["gold"]
        entry = {"regions": regions, "gold": gold, "question": it["question"]}
        if shuffle_test:
            permuted, order2 = permute(regions, rng)
            entry["shuffled_regions"] = permuted
            entry["shuffled_gold"] = order2.index(gold)
        layouts.append(entry)
    return layouts


def evaluate(elicitor, layouts):
    records = []
    for it in layouts:
        regions, gold = it["regions"], it["gold"]
        n = len(regions)
        choice, raw = elicitor.declare(regions, it["question"])
        toks = region_tokens(regions)
        total = sum(toks) or 1

        rec = {
            "question": it["question"],
            "gold": gold,
            "choice": choice,
            "hit": choice == gold,
            "n_regions": n,
            "attended_frac": (toks[choice] / total) if choice is not None else 1.0,
            "raw": raw[:200],
        }

        if "shuffled_regions" in it:
            pchoice, _ = elicitor.declare(it["shuffled_regions"], it["question"])
            rec["shuffled_gold"] = it["shuffled_gold"]
            rec["shuffled_choice"] = pchoice
            rec["shuffled_hit"] = pchoice == it["shuffled_gold"]
            rec["slot_stable"] = (pchoice == choice)
        records.append(rec)
    return records


def _sigma(rate, baseline, n):
    se = (rate * (1 - rate) / n) ** 0.5 if n else 0.0
    if not se:
        se = (baseline * (1 - baseline) / n) ** 0.5 if n else 0.0
    return (rate - baseline) / se if se else float("nan")


def summarize(records, label, n_regions):
    n = len(records)
    if not n:
        return {"label": label, "n": 0}
    hits = sum(int(r["hit"]) for r in records)
    shuffled = [r for r in records if "shuffled_hit" in r]
    gold_counts = Counter(r["gold"] for r in records)
    best_constant = max(gold_counts.values()) / n
    hit_rate = hits / n
    sig_rand = _sigma(hit_rate, 1.0 / n_regions, n)
    sig_const = _sigma(hit_rate, best_constant, n)
    out = {
        "label": label,
        "n": n,
        "n_regions": n_regions,
        "hit_rate": hit_rate,
        "random_control": 1.0 / n_regions,
        "best_constant_control": best_constant,
        "sigma_over_random": sig_rand,
        "sigma_over_constant": sig_const,
        "verdict": verdict_for(min(sig_rand, sig_const)),
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/declare.yaml")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--modes", nargs="*", default=["generate", "read"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--take", type=int, default=0,
                    help="keep only the first N finished layouts (after --offset), an exact slice "
                         "of the full run, applied after the gold-distribution checks")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N finished layouts, applied with --take after the "
                         "gold-distribution checks, so sessions can cover consecutive slices of "
                         "one seed's layouts")
    ap.add_argument("--no-shuffle-test", action="store_true")
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect-gold", default=None,
                    help="JSON map of region to gold count; refuses to run if the layouts "
                         "differ from the run this one must be comparable with")
    ap.add_argument("--gold-ref", default=None,
                    help="path holding the gold distribution for this seed; written on the "
                         "first run and checked against on every later one")
    ap.add_argument("--kv-bits", type=int, default=0,
                    help="fake-quantize keys and values to this many bits inside attention; "
                         "0 leaves the cache at full precision")
    ap.add_argument("--kv-scheme", default="token", choices=list(SCHEMES),
                    help="token quantizes keys and values per token; channel_keys quantizes keys "
                         "per channel over groups of 32 tokens and keeps values per token")
    ap.add_argument("--set", nargs="*", default=None)
    return ap.parse_args(argv)


def register_attention(plan):
    if plan.kind == "probe":
        from declare.elicit import register_probe

        impl = register_probe()
        print(f"attention probing on: loaded with the {impl} attention implementation, "
              "which keeps SDPA for the forward pass and captures only the final query row")
    elif plan.kind == "quant":
        from declare.elicit import register_quant

        impl = register_quant(plan.kv_bits, scheme=plan.kv_scheme)
        print(f"keys and values fake-quantized to {plan.kv_bits} bits, scheme {plan.kv_scheme}, "
              "groups of 32")
    else:
        from declare.elicit import register_repeat

        impl = register_repeat()
    return impl


def require_probe_parity(model, tokenizer):
    from declare.elicit import check_probe_parity

    diff, same_top, ok = check_probe_parity(model, tokenizer)
    print(f"probe parity vs sdpa on a left-padded batch: max logprob diff {diff:.4f}, "
          f"same top-1 {same_top}")
    if not ok:
        raise SystemExit(
            "the probe attention implementation does not reproduce SDPA on a padded batch, "
            "so every mode in this run - read included - would be measured on a corrupted "
            "model. Fix the probe before running.")


def load_items(rows, n_regions, tokenizer, limit):
    items = []
    for row in rows:
        items.extend(items_for(Trajectory.from_dict(row), n_regions, tokenizer))
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("no probes had a locatable gold region")
    return items


def gold_counts(layouts):
    return dict(sorted(Counter(lay["gold"] for lay in layouts).items()))


def gold_distribution(counts):
    return {str(k): int(v) for k, v in counts.items()}


def check_expected_gold(got, expect_gold):
    want = {str(k): int(v) for k, v in json.loads(expect_gold).items()}
    if got != want:
        raise SystemExit(
            f"gold distribution {got} does not match the expected {want}, so these layouts "
            f"are not the ones the other modes were measured on and the comparison would be "
            f"invalid. Check n_eval, the data directory and the seed.")
    print("gold distribution matches the reference run; layouts are comparable")


def check_gold_ref(got, ref):
    ref = Path(ref)
    if not ref.exists():
        write_json(ref, got)
        print(f"first run at this seed; wrote the gold distribution -> {ref}")
        return
    want = read_json(ref)
    if got != want:
        raise SystemExit(
            f"gold distribution {got} does not match {ref} ({want}), so these layouts "
            f"are not the ones the other runs at this seed were measured on and the "
            f"comparison would be invalid. Check n_eval, the data directory and the seed.")
    print(f"gold distribution matches {ref}; layouts are comparable")


def take_window(layouts, offset, take):
    if offset < 0 or take < 0:
        raise SystemExit(f"--offset ({offset}) and --take ({take}) must not be negative")
    if offset >= len(layouts) and offset:
        raise SystemExit(f"--offset {offset} leaves no layouts (there are {len(layouts)})")
    if not offset and not take:
        return layouts
    end = offset + take if take else len(layouts)
    window = layouts[offset:end]
    if offset:
        print(f"taking layouts {offset} to {offset + len(window) - 1} ({len(window)} layouts)")
    else:
        print(f"taking the first {len(window)} layouts")
    return window


def require_window(model, tokenizer, layouts):
    window = model.config.max_position_embeddings
    probe_len = max(len(tokenizer(render(lay["regions"], tokenizer)[0],
                                  add_special_tokens=False)["input_ids"]) for lay in layouts)
    print(f"longest rendered context {probe_len} tokens, model window {window}")
    if probe_len >= window:
        raise SystemExit(
            f"the generate prompt ({probe_len} tokens) does not fit the model window "
            f"({window}). It would be truncated and the comparison against read would be "
            f"confounded by context length rather than elicitation. Lower "
            f"data.per_trajectory or pick a model with a longer window.")


SUMMARY_KEYS = ("hit_rate", "random_control", "best_constant_control",
                "sigma_over_random", "sigma_over_constant",
                "shuffled_hit_rate", "content_dependence", "slot_stable_rate",
                "modal_share", "unparsed_rate", "mean_attended_frac")


def run_mode(mode, model, tokenizer, cfg, plan, layouts, n_regions, out_dir, offset=0):
    elicitor = build_elicitor(mode, model, tokenizer, cfg)
    records = evaluate(elicitor, layouts)
    s = summarize(records, mode, n_regions)
    s["model"] = cfg["model"]["base"]
    s["seed"] = cfg["seed"]
    s["kv_bits"] = plan.kv_bits
    s["kv_scheme"] = plan.recorded_scheme
    s["elicitor_unparsed"] = elicitor.unparsed
    s["layout_offset"] = offset
    write_jsonl(out_dir / f"{mode}.records.jsonl", records)
    write_json(out_dir / f"{mode}.summary.json", s)
    return s


def print_mode_summary(mode, s, n_probes, n_regions):
    print(f"\n== {mode}")
    for k in SUMMARY_KEYS:
        if k in s:
            print(f"  {k:<24} {s[k]:.3f}")
    print(f"  {'verdict':<24} {s['verdict']} (weaker of the two controls)")
    print(f"  {'gold_distribution':<24} {s['gold_distribution']}")
    if n_probes < 200:
        print(f"  NOTE: n={n_probes} is small; best_constant_control is the max of "
              f"{n_regions} empirical bins and is biased upward below ~200 probes.")
    if s["hit_rate"] <= s["best_constant_control"]:
        print("  WARNING: no better than always naming one fixed region.")
    if s.get("slot_stable_rate", 0.0) >= 0.9:
        print("  WARNING: the declaration barely moves when region contents are shuffled,")
        print("  so it is naming a slot rather than reading content.")


def merge_summaries(out_dir):
    merged = {}
    for path in sorted(out_dir.glob("*.summary.json")):
        s = read_json(path)
        if s.get("label"):
            merged[s["label"]] = s
    write_json(out_dir / "summary_all.json", list(merged.values()))
    print(f"\nwrote -> {out_dir} ({len(merged)} modes: {', '.join(sorted(merged))})")


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    n_regions = cfg["declare"]["n_regions"]

    rows = read_jsonl(Path(cfg["data"]["dir"]) / cfg["data"][args.split])
    plan = plan_attention(args.modes, args.kv_bits, args.kv_scheme)
    model, tokenizer = load_backbone(cfg, attn_implementation=register_attention(plan))
    if plan.kind == "probe":
        require_probe_parity(model, tokenizer)

    items = load_items(rows, n_regions, tokenizer, args.limit)
    print(f"{len(items)} probes, {n_regions} regions each, model {cfg['model']['base']}")
    out_dir = ensure_dir(args.out or Path(cfg["run_root"]) / "declare")
    layouts = make_layouts(items, rng, not args.no_shuffle_test, balance=not args.no_balance)
    counts = gold_counts(layouts)
    print(f"layouts fixed across modes, gold by region {counts}")
    got = gold_distribution(counts)
    if args.expect_gold:
        check_expected_gold(got, args.expect_gold)
    if args.gold_ref:
        check_gold_ref(got, args.gold_ref)
    layouts = take_window(layouts, args.offset, args.take)
    require_window(model, tokenizer, layouts)

    for mode in args.modes:
        s = run_mode(mode, model, tokenizer, cfg, plan, layouts, n_regions, out_dir,
                     offset=args.offset)
        print_mode_summary(mode, s, len(layouts), n_regions)
    merge_summaries(out_dir)


if __name__ == "__main__":
    main()
