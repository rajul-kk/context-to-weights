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
             "| model | seeds | n | pass@1 | AUROC P(yes) | AUROC token logprob | AUROC verbal "
             "| ECE P(yes) | ECE verbal | unparsed verbal |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    for model, runs in grouped.items():
        a = [r["auroc"] for r in runs]
        e = [r["ece"] for r in runs]
        lines.append(
            f"| {short(model)} | {len(runs)} | {runs[0]['n']} "
            f"| {mean([r['pass_rate'] for r in runs]):.3f} "
            f"| {mean([x['pyes'] for x in a]):.3f} | {mean([x['logprob'] for x in a]):.3f} "
            f"| {mean([x['verbal'] for x in a]):.3f} | {mean([x['pyes'] for x in e]):.3f} "
            f"| {mean([x['verbal'] for x in e]):.3f} "
            f"| {mean([r['unparsed_verbal_rate'] for r in runs]):.3f} |")
        verdicts[model] = verdict(a)
    lines += ["", "## Verdict", ""]
    for model, v in verdicts.items():
        lines.append(f"- {short(model)}: {'PASS' if v['passed'] else 'FAIL'}, {v['reason']}")
    lines += ["", "## Pass rate by family", ""]
    for model, runs in grouped.items():
        families = sorted(runs[0]["pass_by_family"])
        lines.append(f"- {short(model)}: " + ", ".join(
            f"{f} {mean([r['pass_by_family'][f] for r in runs]):.2f}" for f in families))
    return "\n".join(lines) + "\n", verdicts


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
