import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_config, parse_overrides, read_jsonl, set_seed, write_json, write_jsonl
from data.banks import FACT_TEMPLATES
from eval.retention import QA_SYSTEM, build_prompt
from sleep.lm import chat_text, load_backbone, target_ce_batch

VALUE_SETS = {t["key"]: list(t["values"]) for t in FACT_TEMPLATES}


def candidates(fact_key, gold):
    pool = VALUE_SETS.get(fact_key, [])
    others = [v for v in pool if v.strip().lower() != gold.strip().lower()]
    return [gold] + others


def score_probe(model, tokenizer, context, probe, max_length):
    prompt = chat_text(tokenizer, QA_SYSTEM, build_prompt(context, probe["question"]))
    cands = candidates(probe["fact_key"], probe["answer"])
    if len(cands) < 2:
        return None
    ces = target_ce_batch(model, tokenizer, prompt, cands, max_length)
    ranked = sorted(range(len(cands)), key=lambda i: ces[i])
    return {
        "fact_key": probe["fact_key"],
        "gold": probe["answer"],
        "n_candidates": len(cands),
        "gold_ce": ces[0],
        "best_distractor_ce": min(ces[1:]),
        "correct": ranked[0] == 0,
        "margin": min(ces[1:]) - ces[0],
        "evicted": probe["answer"].lower() not in " ".join(context).lower(),
    }


def summarize(records, label):
    if not records:
        return {"label": label, "n": 0}
    ev = [r for r in records if r["evicted"]]
    chance = statistics.fmean(1.0 / r["n_candidates"] for r in records)
    out = {
        "label": label,
        "n": len(records),
        "mc_accuracy": statistics.fmean(r["correct"] for r in records),
        "chance": chance,
        "mean_margin": statistics.fmean(r["margin"] for r in records),
        "n_evicted": len(ev),
        "mc_accuracy_evicted": statistics.fmean(r["correct"] for r in ev) if ev else float("nan"),
        "mean_margin_evicted": statistics.fmean(r["margin"] for r in ev) if ev else float("nan"),
    }
    if ev:
        k = sum(r["correct"] for r in ev)
        n = len(ev)
        p = k / n
        se = (p * (1 - p) / n) ** 0.5 if n else 0.0
        ch = statistics.fmean(1.0 / r["n_candidates"] for r in ev)
        out["evicted_sigma_over_chance"] = (p - ch) / se if se else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="CE-ranking retention: does the gold answer beat its distractors on loss?")
    ap.add_argument("--config", default="configs/kaggle_synthetic.yaml")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default="method")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    contexts = read_jsonl(args.contexts)
    if args.limit:
        contexts = contexts[: args.limit]

    model, tokenizer = load_backbone(cfg)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    records = []
    for row in contexts:
        for probe in row["probes"]:
            r = score_probe(model, tokenizer, row["context"], probe, cfg["model"]["max_length"])
            if r:
                r["traj_id"] = row["traj_id"]
                records.append(r)

    summary = summarize(records, args.label)
    out = Path(args.out)
    write_jsonl(out.with_suffix(".records.jsonl"), records)
    write_json(out.with_suffix(".summary.json"), summary)
    for k, v in summary.items():
        print(f"{k:<26} {v}")


if __name__ == "__main__":
    main()
