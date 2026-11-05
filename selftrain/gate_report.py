import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, read_json
from selftrain.signals import verdict


def load(run_root):
    return [read_json(p) for p in sorted(Path(run_root).glob("*/g1.summary.json"))]


def group(summaries):
    by_model = defaultdict(list)
    for s in summaries:
        by_model[s["model"]].append(s)
    return {m: sorted(v, key=lambda s: s["seed"]) for m, v in by_model.items()}


def mean(values):
    return sum(values) / len(values)


def short(model):
    return model.split("/")[-1]


def render(grouped):
    lines = ["# G1: is there a measurable correctness signal", "",
             "AUROC is the mean over task families that have at least 5 correct and 5 incorrect "
             "samples, so a signal that only identifies the family scores 0.5.", "",
             "| model | seeds | n | pass@1 | token logprob | P(yes) | verbal | pooled logprob "
             "| ECE P(yes) | ECE verbal | unparsed verbal |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    for model, runs in grouped.items():
        a = [r["auroc"] for r in runs]
        pooled = [r["auroc_pooled"]["logprob"] for r in runs]
        e = [r["ece"] for r in runs]
        usable = all(x[k] is not None for x in a for k in ("pyes", "logprob", "verbal"))
        lines.append(
            f"| {short(model)} | {len(runs)} | {runs[0]['n']} "
            f"| {mean([r['pass_rate'] for r in runs]):.3f} "
            f"| {cell(a, 'logprob')} | {cell(a, 'pyes')} | {cell(a, 'verbal')} "
            f"| {mean(pooled):.3f} | {mean([x['pyes'] for x in e]):.3f} "
            f"| {mean([x['verbal'] for x in e]):.3f} "
            f"| {mean([r['unparsed_verbal_rate'] for r in runs]):.3f} |")
        verdicts[model] = verdict(a) if usable else {
            "passed": False, "reason": "a seed has no family with 5 correct and 5 incorrect samples"}
    lines += ["", "## Verdict", ""]
    for model, v in verdicts.items():
        lines.append(f"- {short(model)}: {'PASS' if v['passed'] else 'FAIL'}, {v['reason']}")
    lines += ["", "## Pass rate by family", ""]
    for model, runs in grouped.items():
        families = sorted(runs[0]["pass_by_family"])
        lines.append(f"- {short(model)}: " + ", ".join(
            f"{f} {mean([r['pass_by_family'][f] for r in runs]):.2f}" for f in families))
    lines += ["", "## Token log-prob AUROC by family", ""]
    for model, runs in grouped.items():
        lines.append(f"- {short(model)}: " + family_cells(runs))
    return "\n".join(lines) + "\n", verdicts


def family_cells(runs):
    families = sorted({f for r in runs for f in r["auroc_by_family"]["logprob"]})
    cells = []
    for f in families:
        values = [r["auroc_by_family"]["logprob"][f] for r in runs
                  if f in r["auroc_by_family"]["logprob"]]
        cells.append(f"{f} {mean(values):.2f}")
    return ", ".join(cells)


def cell(rows, key):
    values = [x[key] for x in rows if x[key] is not None]
    return f"{mean(values):.3f}" if values else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="artifacts/selftrain")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    summaries = load(args.run_root)
    if not summaries:
        raise SystemExit(f"no g1.summary.json under {args.run_root}")
    text, _ = render(group(summaries))
    out = Path(args.out or Path(args.run_root) / "g1_report.md")
    ensure_dir(out.parent)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
