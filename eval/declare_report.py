import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir


def load(run_root):
    rows = []
    for p in sorted(glob.glob(str(Path(run_root) / "*" / "summary_all.json"))):
        for s in json.load(open(p)):
            if s.get("n"):
                rows.append(s)
    return rows


def short(name):
    return name.split("/")[-1].replace("-Instruct", "")


def table(rows):
    head = ("| model | mode | hit | chance | const | sigma(const) | content dep | "
            "slot stable | unparsed | attended |")
    rule = "|" + "|".join("---" for _ in range(10)) + "|"
    lines = [head, rule]
    for s in rows:
        lines.append(
            f"| {short(s['model'])} | {s['label']} | {s['hit_rate']:.3f} | "
            f"{s['random_control']:.3f} | {s['best_constant_control']:.3f} | "
            f"{s['sigma_over_constant']:+.1f} | {s.get('content_dependence', float('nan')):.3f} | "
            f"{s.get('slot_stable_rate', float('nan')):.3f} | {s['unparsed_rate']:.3f} | "
            f"{s['mean_attended_frac']:.3f} |")
    return "\n".join(lines)


def plot(rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = sorted({s["model"] for s in rows}, key=short)
    order = ["generate", "read", "attention", "attention_mean", "attention_late"]
    modes = sorted({s["label"] for s in rows},
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
            vals = [next((s.get(key, 0.0) for s in rows
                          if s["model"] == m and s["label"] == mode), 0.0) for m in models]
            off = (j - (len(modes) - 1) / 2) * width
            ax.bar([i + off for i in x], vals, width, label=mode,
                   color=colors.get(mode, "#999999"))
        if key == "hit_rate":
            ch = rows[0]["random_control"]
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

    fig_dir = ensure_dir(Path(args.run_root) / "figures")
    plot(rows, fig_dir / "declare.png")

    body = ["# Declarative attention: asking versus measuring", ""]
    body += [table(rows), ""]

    def mean_of(mode, key):
        vals = [s.get(key, 0.0) for s in rows if s["label"] == mode]
        return sum(vals) / len(vals) if vals else float("nan")

    modes = sorted({s["label"] for s in rows})
    body += ["## Reading", ""]
    line = ", ".join(f"{m} {mean_of(m, 'content_dependence'):.3f}" for m in modes)
    body += [f"Mean `content_dependence` by elicitation: {line}.", ""]

    rd, ad = mean_of("read", "content_dependence"), mean_of("attention", "content_dependence")
    if rd == rd and ad == ad:
        if rd > ad:
            verdict = (f"The semantic self-query beats attention probing ({rd:.3f} vs {ad:.3f}). "
                       f"Asking the model which region holds the answer recovers more content "
                       f"signal than reading where it actually attends.")
        elif ad > rd:
            verdict = (f"Attention probing beats the semantic self-query ({ad:.3f} vs {rd:.3f}). "
                       f"Where the model attends is a better guide than what it says about the "
                       f"region, so the read elicitation is not the cheapest route to the signal.")
        else:
            verdict = f"Self-query and attention probing are level ({rd:.3f} vs {ad:.3f})."
        body += [verdict, ""]

    body += ["![declare](figures/declare.png)", "",
             "*Left: hit rate by model and elicitation, dashed line at chance. Right: content "
             "dependence, the shuffled hit rate above chance. generate (grey) asks the model to "
             "state a region; read (red) scores each region off the logits; attention (blue) "
             "reads where the model attends when answering.*", ""]

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text("\n".join(body), encoding="utf-8")
    print("\n".join(body))
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    raise SystemExit(main())
