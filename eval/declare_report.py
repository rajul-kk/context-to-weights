import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir

METRICS = ("hit_rate", "random_control", "best_constant_control", "sigma_over_random",
           "sigma_over_constant", "shuffled_hit_rate", "content_dependence",
           "slot_stable_rate", "unparsed_rate", "mean_attended_frac", "modal_share")


def load(run_root):
    rows = []
    for p in sorted(glob.glob(str(Path(run_root) / "*" / "*.summary.json"))):
        s = json.load(open(p))
        if s.get("n"):
            s.setdefault("seed", 0)
            rows.append(s)
    return rows


def short(name):
    return name.split("/")[-1].replace("-Instruct", "")


def group(rows):
    by = {}
    for s in rows:
        by.setdefault((s["model"], s["label"]), []).append(s)
    out = []
    for (model, label), runs in by.items():
        runs = sorted(runs, key=lambda s: s["seed"])
        g = {"model": model, "label": label, "seeds": [s["seed"] for s in runs],
             "k": len(runs), "n": runs[0].get("n", 0), "runs": runs}
        for key in METRICS:
            vals = [s[key] for s in runs if key in s]
            if not vals:
                continue
            g[key] = sum(vals) / len(vals)
            g[key + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out.append(g)
    return sorted(out, key=lambda g: (short(g["model"]), g["label"]))


def cell(g, key, places=3):
    if key not in g:
        return "-"
    if g["k"] > 1:
        return f"{g[key]:.{places}f} +/- {g[key + '_sd']:.{places}f}"
    return f"{g[key]:.{places}f}"


def table(groups):
    multi = any(g["k"] > 1 for g in groups)
    head = ["model", "mode", "seeds", "hit", "chance", "const", "sigma(const)",
            "content dep", "slot stable", "unparsed", "attended"]
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join("---" for _ in head) + "|"]
    for g in groups:
        lines.append("| " + " | ".join([
            short(g["model"]), g["label"], str(g["k"]),
            cell(g, "hit_rate"), cell(g, "random_control"), cell(g, "best_constant_control"),
            cell(g, "sigma_over_constant", 1), cell(g, "content_dependence"),
            cell(g, "slot_stable_rate"), cell(g, "unparsed_rate"),
            cell(g, "mean_attended_frac")]) + " |")
    if multi:
        lines += ["",
                  "Each cell is the mean over seeds +/- the standard deviation across them. The "
                  "spread is over layout permutations, not over probes: the probe set is fixed "
                  "by the data file and only the gold-region assignment and the content shuffle "
                  "are reseeded."]
    return "\n".join(lines)


def se(p, n):
    return (p * (1 - p) / n) ** 0.5 if n else float("nan")


CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228}


def critical(k):
    if k <= 1:
        return 2.0
    return CRIT.get(k - 1, 2.0)


def paired(read_g, att_g, key):
    by_seed = {s["seed"]: s for s in att_g["runs"]}
    diffs = [(s[key] - by_seed[s["seed"]][key]) for s in read_g["runs"]
             if s["seed"] in by_seed and key in s and key in by_seed[s["seed"]]]
    if len(diffs) > 1:
        mean = sum(diffs) / len(diffs)
        sd = statistics.stdev(diffs)
        stat = mean / (sd / len(diffs) ** 0.5) if sd else float("nan")
        return mean, stat, f"t({len(diffs) - 1})", len(diffs)
    n = read_g.get("n", 0)
    d = read_g.get(key, 0.0) - att_g.get(key, 0.0)
    pooled = (se(read_g.get(key, 0.0), n) ** 2 + se(att_g.get(key, 0.0), n) ** 2) ** 0.5
    return d, (d / pooled if pooled else float("nan")), "sigma", 1


def plot(groups, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({g["model"] for g in groups}, key=short)
    order = ["generate", "read", "attention", "attention_mean", "attention_late"]
    modes = sorted({g["label"] for g in groups},
                   key=lambda m: order.index(m) if m in order else 99)
    colors = {"generate": "#8c9bb0", "read": "#c44e52", "attention": "#4c72b0",
              "attention_mean": "#7ba7d4", "attention_late": "#2f4b6e"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.8 / max(1, len(modes))
    x = range(len(models))

    for ax, key, title, ref in (
        (axes[0], "hit_rate", "hit rate", None),
        (axes[1], "content_dependence", "content dependence (shuffled hit - chance)", 0.0),
    ):
        for j, mode in enumerate(modes):
            picked = [next((g for g in groups if g["model"] == m and g["label"] == mode), None)
                      for m in models]
            vals = [g.get(key, 0.0) if g else 0.0 for g in picked]
            errs = [g.get(key + "_sd", 0.0) if g else 0.0 for g in picked]
            off = (j - (len(modes) - 1) / 2) * width
            ax.bar([i + off for i in x], vals, width, label=mode,
                   color=colors.get(mode, "#999999"),
                   yerr=errs if any(errs) else None, capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": "#333"})
        if key == "hit_rate":
            ch = groups[0].get("random_control", 0.125)
            ax.axhline(ch, ls="--", lw=1, color="#333", label=f"chance {ch:.3f}")
        if ref is not None:
            ax.axhline(ref, lw=1, color="#333")
        ax.set_xticks(list(x))
        ax.set_xticklabels([short(m) for m in models])
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", default="artifacts/runs_declare")
    ap.add_argument("--out", default="docs/results_declare.md")
    args = ap.parse_args()

    rows = load(args.run_root)
    if not rows:
        print(f"no summaries under {args.run_root}")
        return 1
    groups = group(rows)

    fig_dir = ensure_dir(Path(args.run_root) / "figures")
    plot(groups, fig_dir / "declare.png")

    seeds = sorted({s["seed"] for s in rows})
    k = min(g["k"] for g in groups)
    kmax = max(g["k"] for g in groups)

    body = ["# Declarative attention: asking versus measuring", ""]
    body += [f"Seeds: {', '.join(str(s) for s in seeds)}.", ""]
    if kmax > 1 and k != kmax:
        body += [f"**Unbalanced seeds.** Some model/mode pairs have {k} run(s) and others "
                 f"{kmax}. Every comparison below is paired only over the seeds both arms "
                 f"share, so the table and the margins may rest on different run counts.", ""]
    body += [table(groups), ""]

    body += ["## Reading", ""]
    body += ["`read` against the best attention variant, per model. Comparing pooled means "
             "across models hides a reversal, and comparing against all-layer `attention` "
             "alone understates probing, so neither is reported.", ""]
    if kmax > 1:
        body += ["With more than one seed the margin is a paired t across seeds, not a "
                 "binomial sigma: the seeds share a probe set, so the layout permutation is "
                 "the only thing resampled and a within-seed binomial would understate the "
                 "true variability.", ""]

    margins = []
    for model in sorted({g["model"] for g in groups}, key=short):
        got = {g["label"]: g for g in groups if g["model"] == model}
        r = got.get("read")
        atts = [g for lbl, g in got.items() if lbl.startswith("attention")]
        if not r or not atts:
            continue
        best = max(atts, key=lambda g: g.get("content_dependence", 0.0))
        for key, label in (("hit_rate", "hit rate"), ("content_dependence", "content dependence")):
            d, stat, name, kk = paired(r, best, key)
            if key == "content_dependence":
                margins.append((stat, critical(kk)))
            body.append(f"- **{short(model)}**, {label}: read {r.get(key, 0.0):.3f} vs "
                        f"{best['label']} {best.get(key, 0.0):.3f}, {d:+.3f} "
                        f"({stat:+.2f} {name}, {kk} seed{'s' if kk > 1 else ''})")
    body += [""]

    stats = [m[0] for m in margins]
    if margins and min(stats) < 0 < max(stats):
        sig = [f"{s:+.2f} vs +/-{c:.2f}" for s, c in margins if abs(s) >= c]
        body += ["**The ordering reverses with scale.** The self-query wins at the smaller model "
                 "and loses at the larger one, so neither elicitation dominates. Report the "
                 "crossover rather than a single winner.", ""]
        if sig:
            body += [f"Margins clearing their own critical value: {', '.join(sig)}. A crossover "
                     f"is only a finding if at least one side of it is significant.", ""]
        else:
            body += ["**Neither side of the crossover is significant at its own critical value**, "
                     "so this is a tie, not a reversal. Do not report it as one.", ""]
    elif margins and all(s >= c for s, c in margins):
        body += ["The semantic self-query beats attention probing at every model tested.", ""]
    elif margins and all(s <= -c for s, c in margins):
        body += ["Attention probing beats the semantic self-query at every model tested.", ""]
    elif margins:
        body += ["The two elicitations are within noise of each other; no winner is established "
                 "by these runs.", ""]

    if kmax == 1:
        body += ["**Single seed.** Every margin above is one run at one layout draw, so a "
                 "crossover this small cannot be distinguished from layout noise. Re-run with "
                 "`--set seed=N` into a separate `--out` before citing any of it.", ""]

    trend = {}
    for g in groups:
        trend.setdefault(g["label"], []).append((short(g["model"]), g.get("content_dependence", 0.0)))
    body += ["Content dependence by scale:", ""]
    for label in sorted(trend):
        pts = sorted(trend[label])
        arrow = " -> ".join(f"{v:.3f}" for _, v in pts)
        delta = f" ({pts[-1][1] - pts[0][1]:+.3f})" if len(pts) > 1 else ""
        body.append(f"- `{label}`: {arrow}{delta}")
    body += [""]

    body += ["![declare](figures/declare.png)", "",
             "*Left: hit rate by model and elicitation, dashed line at chance. Right: content "
             "dependence, the shuffled hit rate above chance. generate (grey) asks the model to "
             "state a region; read (red) scores each region off the logits; attention (blue) "
             "reads where the model attends when answering. Error bars are the standard "
             "deviation across seeds.*", ""]

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text("\n".join(body), encoding="utf-8")
    print("\n".join(body))
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
