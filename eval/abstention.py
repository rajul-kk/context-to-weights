import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import load_config, parse_overrides, read_jsonl, set_seed, write_json, write_jsonl
from eval.metrics import answer_match
from eval.retention import QA_SYSTEM, build_prompt
from sleep.abstain import survives
from sleep.lm import batch_generate, load_backbone

REFUSAL = re.compile(
    r"\b(do not have|don'?t have|not (?:in|available|present|mentioned|provided|stated)|"
    r"no information|cannot (?:find|determine|answer)|can'?t (?:find|tell)|unknown|"
    r"not specified|isn'?t (?:in|mentioned)|unable to)\b", re.IGNORECASE)


def is_abstention(text):
    return bool(REFUSAL.search(text or ""))


def _rate(rows, key):
    return sum(1 for r in rows if r[key]) / len(rows) if rows else float("nan")


def summarize(records, label):
    ev = [r for r in records if r["evicted"]]
    rt = [r for r in records if not r["evicted"]]
    out = {
        "label": label,
        "n": len(records),
        "n_evicted": len(ev),
        "n_retained": len(rt),
        "abstain_rate_evicted": _rate(ev, "abstained"),
        "abstain_rate_retained": _rate(rt, "abstained"),
        "accuracy_retained": _rate(rt, "correct"),
        "hallucination_rate_evicted": _rate(ev, "hallucinated"),
    }
    ae, ar = out["abstain_rate_evicted"], out["abstain_rate_retained"]
    out["abstention_margin"] = ae - ar if ae == ae and ar == ar else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Does the adapter refuse when the compactor evicted the answer, "
                    "without refusing when it did not?")
    ap.add_argument("--config", default="configs/kaggle_synthetic.yaml")
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--label", default="method")
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    contexts = read_jsonl(args.contexts)

    model, tokenizer = load_backbone(cfg)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        model.eval()

    pairs, index = [], []
    for row in contexts:
        for probe in row["probes"]:
            pairs.append((QA_SYSTEM, build_prompt(row["context"], probe["question"])))
            index.append((row["traj_id"], probe, not survives(row["context"], probe)))

    preds = batch_generate(model, tokenizer, pairs,
                           max_new_tokens=cfg["eval"]["max_new_tokens"],
                           batch_size=cfg["eval"]["batch_size"])

    records = []
    for (traj_id, probe, evicted), pred in zip(index, preds):
        abstained = is_abstention(pred)
        correct = answer_match(pred, probe["answer"], probe.get("aliases"))
        records.append({
            "traj_id": traj_id,
            "fact_key": probe["fact_key"],
            "question": probe["question"],
            "gold": probe["answer"],
            "prediction": pred,
            "evicted": evicted,
            "abstained": abstained,
            "correct": correct,
            "hallucinated": evicted and not abstained and not correct,
        })

    summary = summarize(records, args.label)
    summary["model"] = cfg["model"]["base"]
    summary["adapter"] = args.adapter
    out = Path(args.out)
    write_jsonl(out.with_suffix(".records.jsonl"), records)
    write_json(out.with_suffix(".summary.json"), summary)
    for k, v in summary.items():
        print(f"{k:<28} {v}")
    if summary["abstain_rate_retained"] > 0.5:
        print("\nWARNING: it refuses on more than half the probes whose answer is still in "
              "context, so this is a refusal-biased model rather than a calibrated one.")


if __name__ == "__main__":
    main()
