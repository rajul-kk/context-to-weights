import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, read_json
from common.stats import critical, paired_t


def load(run_root):
    rows = []
    for path in sorted(Path(run_root).glob("*/read.summary.json")):
        s = read_json(path)
        if s.get("n") and "kv_bits" in s:
            rows.append(s)
    return rows


def short(model):
    return model.split("/")[-1].replace("-Instruct", "")


def organise(rows):
    out = {}
    for s in rows:
        key = (s["model"], s["kv_scheme"], s["kv_bits"])
        out.setdefault(key, {})[s["seed"]] = s
    return out


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def compare(base, cond, metric="hit_rate"):
    seeds = sorted(set(base) & set(cond))
    if len(seeds) < 2:
        return None
    a = [base[s][metric] for s in seeds]
    b = [cond[s][metric] for s in seeds]
    drop, t, dof = paired_t(a, b)
    return {"seeds": seeds, "drop": drop, "t": t, "dof": dof,
            "significant": drop > 0 and abs(t) >= critical(len(seeds))}


def baseline_check(base):
    sigma = [s["sigma_over_constant"] for s in base.values()]
    return {"valid": all(x >= 2.0 for x in sigma), "sigma": sigma}


def classify(drops):
    if not any(drops.values()):
        return "not enough paired seeds"
    ordered = sorted(drops, reverse=True)
    top = ordered[0]
    first = next((b for b in ordered if drops[b] and drops[b]["significant"]), None)
    if first is None:
        return f"survives down to {ordered[-1]} bits, so there is nothing to guide"
    if first == top:
        return f"already degraded at {first} bits"
    return f"cliff at {first} bits, holds at {ordered[ordered.index(first) - 1]}"


def render(rows):
    table = organise(rows)
    models = sorted({k[0] for k in table})
    lines = ["# KV quantization gate: does routing survive low-bit keys and values", "",
             "Drop is baseline hit rate minus quantized hit rate, paired by seed. "
             "Significant means a two-sided paired t at 5%.", ""]
    verdicts = {}
    for model in models:
        base = table.get((model, "none", 0))
        lines += [f"## {short(model)}", ""]
        if not base:
            lines += ["No full-precision baseline in this run root.", ""]
            verdicts[model] = {"valid": False}
            continue
        check = baseline_check(base)
        mean, sd = mean_sd([s["hit_rate"] for s in base.values()])
        lines += [f"Baseline hit rate {mean:.3f} ± {sd:.3f} over {len(base)} seeds, sigma over "
                  f"best constant {', '.join(f'{x:.1f}' for x in check['sigma'])}: "
                  f"{'valid' if check['valid'] else 'INVALID, below 2 sigma on a seed'}.", "",
                  "| scheme | bits | hit rate | content dependence | drop | t | significant |",
                  "|---|---|---|---|---|---|---|"]
        verdicts[model] = {"valid": check["valid"], "schemes": {}}
        for scheme in sorted({k[1] for k in table if k[0] == model and k[1] != "none"}):
            drops = {}
            for bits in sorted({k[2] for k in table if k[0] == model and k[1] == scheme}):
                cond = table[(model, scheme, bits)]
                hit, hit_sd = mean_sd([s["hit_rate"] for s in cond.values()])
                content = statistics.mean(s["content_dependence"] for s in cond.values())
                cmp = compare(base, cond)
                drops[bits] = cmp
                tail = (f"{cmp['drop']:+.3f} | {cmp['t']:.2f} | {'yes' if cmp['significant'] else 'no'}"
                        if cmp else "n/a | n/a | n/a")
                lines.append(f"| {scheme} | {bits} | {hit:.3f} ± {hit_sd:.3f} | {content:.3f} "
                             f"| {tail} |")
            verdicts[model]["schemes"][scheme] = classify(drops)
        lines += ["", "Verdict:"]
        for scheme, text in verdicts[model]["schemes"].items():
            lines.append(f"- {scheme}: {text}")
        lines.append("")
    return "\n".join(lines) + "\n", verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="artifacts/runs_kvquant")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load(args.run_root)
    if not rows:
        raise SystemExit(f"no read.summary.json with kv_bits under {args.run_root}")
    text, _ = render(rows)
    out = Path(args.out or Path(args.run_root) / "kvquant_report.md")
    ensure_dir(out.parent)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
