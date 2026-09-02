import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, read_json, write_json

ORDER = ["no-skill", "random-control", "s2l-uniform", "ours", "prompt-full",
         "prompt-mismatched", "ours-mismatched"]
PRETTY = {
    "no-skill": "(c) no skill (floor)",
    "random-control": "(d) random-span control",
    "s2l-uniform": "(b) S2L-style uniform distillation",
    "ours": "ours: KL-gated distillation",
    "prompt-full": "(a) full skill text in prompt",
    "prompt-mismatched": "(a) prompt, mismatched doc",
    "ours-mismatched": "ours, mismatched doc",
}

COLUMNS = [
    ("label", "method"),
    ("pass_rate", "pass"),
    ("pass_rate_in_group", "pass(in-group)"),
    ("pass_rate_out_group", "pass(out-group)"),
    ("mean_prompt_tokens", "tokens"),
]


def collect(report_dir):
    rows = [read_json(p) for p in sorted(Path(report_dir).glob("*.summary.json"))]
    rows.sort(key=lambda r: ORDER.index(r["label"]) if r["label"] in ORDER else 99)
    return rows


def to_markdown(rows):
    lines = ["| " + " | ".join(h for _, h in COLUMNS) + " |",
             "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for r in rows:
        cells = []
        for key, _ in COLUMNS:
            v = r.get(key)
            if key == "label":
                cells.append(PRETTY.get(v, v))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def token_saving(rows):
    by = {r["label"]: r for r in rows}
    if "prompt-full" not in by or "ours" not in by:
        return None
    full = by["prompt-full"]["mean_prompt_tokens"]
    ours = by["ours"]["mean_prompt_tokens"]
    if not full:
        return None
    return {"prompt_tokens": full, "internalised_tokens": ours,
            "saving": 1.0 - ours / full}


def plot_headline(rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = [r for r in rows if r["label"] in
            ("no-skill", "random-control", "s2l-uniform", "ours", "prompt-full")]
    labels = [PRETTY.get(r["label"], r["label"]) for r in keep]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    bars = axes[0].bar(range(len(keep)), [r["pass_rate"] for r in keep], color="#4c72b0")
    for i, r in enumerate(keep):
        if r["label"] == "ours":
            bars[i].set_color("#c44e52")
        if r["label"] == "random-control":
            bars[i].set_color("#8c8c8c")
    axes[0].set_ylabel("task pass rate")
    axes[0].set_ylim(0, 1)
    axes[1].bar(range(len(keep)), [r["mean_prompt_tokens"] for r in keep], color="#55a868")
    axes[1].set_ylabel("mean prompt tokens")
    for ax in axes:
        ax.set_xticks(range(len(keep)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sweep(run_root, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    for d in sorted(Path(run_root).glob("sweep_*/report")):
        parts = d.parent.name.split("_")
        if len(parts) < 3:
            continue
        gran, frac = parts[1], parts[2]
        s = d / "ours.summary.json"
        if s.exists():
            series.setdefault(gran, []).append((float(frac), read_json(s)["pass_rate"]))
    if not series:
        return False

    colors = {"span": "#c44e52", "token": "#4c72b0"}
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for gran, points in sorted(series.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o",
                label=f"{gran}-level gating", color=colors.get(gran))
    ax.set_xlabel("KL gate top fraction")
    ax.set_ylabel("task pass rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    if len(series) > 1:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="artifacts/runs_skill/report")
    ap.add_argument("--run-root", default="artifacts/runs_skill")
    ap.add_argument("--out", default="docs/results_skills.md")
    args = ap.parse_args()

    rows = collect(args.report_dir)
    if not rows:
        print(f"no summaries in {args.report_dir}")
        return

    fig_dir = ensure_dir(Path(args.report_dir) / "figures")
    plot_headline(rows, fig_dir / "headline_skills.png")
    has_sweep = plot_sweep(args.run_root, fig_dir / "kl_threshold_sweep.png")

    table = to_markdown(rows)
    saving = token_saving(rows)
    body = ["# Skill distillation results", "", "## Pass rate and token cost", "", table, ""]

    by = {r["label"]: r for r in rows}
    warnings = []
    prompt = by.get("prompt-full", {}).get("pass_rate")
    floor = by.get("no-skill", {}).get("pass_rate")
    if prompt is not None and floor is not None and prompt - floor < 0.15:
        warnings.append(
            f"The skill document barely helps in-prompt ({prompt:.3f} against a {floor:.3f} "
            "floor), so there is almost nothing for distillation to internalise. Use a larger "
            "backbone or easier tasks before comparing methods.")
    distilled = [by[k]["pass_rate"] for k in ("ours", "s2l-uniform", "random-control") if k in by]
    if distilled and floor is not None and max(distilled) <= floor:
        warnings.append(
            "No distilled adapter beats the no-skill floor. Either training is too short or "
            "the backbone is too small; the gate cannot be evaluated from this run.")
    for w in warnings:
        body += [f"> **Configuration is uninformative.** {w}", ""]
        print(f"\nWARNING: {w}")
    if saving:
        body += [f"Runtime token saving vs keeping the skill in the prompt: "
                 f"**{saving['saving'] * 100:.1f}%** "
                 f"({saving['prompt_tokens']:.0f} -> {saving['internalised_tokens']:.0f} tokens).", ""]
    body += ["## Figures", "",
             "![headline](../artifacts/runs_skill/report/figures/headline_skills.png)", ""]
    if has_sweep:
        body += ["![sweep](../artifacts/runs_skill/report/figures/kl_threshold_sweep.png)", ""]

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text("\n".join(body), encoding="utf-8")
    write_json(Path(args.report_dir) / "summary_all.json", {"rows": rows, "token_saving": saving})
    print(table)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
