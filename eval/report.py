import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, read_json, read_jsonl, write_json

ORDER = ["floor", "cascading", "uniform", "reflection", "ours", "ours+mask", "full"]
PRETTY = {
    "floor": "no context (floor)",
    "cascading": "(c) cascading, no update",
    "uniform": "(a) uniform replay",
    "reflection": "(b) reflection",
    "ours": "ours: compaction-supervised",
    "ours+mask": "ours + mask head",
    "full": "(d) full context (ceiling)",
}

COLUMNS = [
    ("label", "method"),
    ("retention_accuracy", "acc"),
    ("evicted_accuracy", "acc(evicted)"),
    ("n_evicted", "n_ev"),
    ("evicted_ce_median", "medCE(ev)"),
    ("token_ce_median", "medCE"),
    ("token_ce_mean", "meanCE"),
    ("mean_prompt_tokens", "tokens"),
]


def collect(report_dir):
    rows = []
    for path in sorted(Path(report_dir).glob("*.summary.json")):
        rows.append(read_json(path))
    rows.sort(key=lambda r: ORDER.index(r["label"]) if r["label"] in ORDER else 99)
    return rows


def sleep_cost(run_dirs):
    out = {}
    for label, d in run_dirs.items():
        path = Path(d) / "metrics.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        secs = [r.get("gpu_seconds", 0.0) for r in rows]
        out[label] = {
            "phases": len(rows),
            "total_gpu_seconds": sum(secs),
            "mean_gpu_seconds_per_phase": sum(secs) / len(secs) if secs else 0.0,
        }
    return out


def to_markdown(rows):
    head = "| " + " | ".join(h for _, h in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    lines = [head, rule]
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


def plot_headline(rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [PRETTY.get(r["label"], r["label"]) for r in rows]
    acc = [r.get("evicted_accuracy") or 0.0 for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(range(len(rows)), acc, color="#4c72b0")
    for i, r in enumerate(rows):
        if r["label"].startswith("ours"):
            bars[i].set_color("#c44e52")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("retention accuracy on evicted facts")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_ce_curves(run_dirs, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for label, d in run_dirs.items():
        path = Path(d) / "metrics.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        phases = [r["phase"] for r in rows]
        med = [r.get("final", {}).get("val_ce_median") for r in rows]
        mean = [r.get("final", {}).get("val_ce_mean") for r in rows]
        axes[0].plot(phases, med, marker="o", label=PRETTY.get(label, label))
        axes[1].plot(phases, mean, marker="o", label=PRETTY.get(label, label))
    axes[0].set_title("median per-token val CE (trustworthy)")
    axes[1].set_title("mean per-token val CE (misleading)")
    for ax in axes:
        ax.set_xlabel("sleep phase")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default="artifacts/runs/report")
    ap.add_argument("--run-root", default="artifacts/runs")
    ap.add_argument("--out", default="docs/results.md")
    args = ap.parse_args()

    rows = collect(args.report_dir)
    if not rows:
        print(f"no summaries in {args.report_dir}")
        return

    run_dirs = {
        "ours": Path(args.run_root) / "sleep_compaction",
        "ours+mask": Path(args.run_root) / "sleep_compaction+mask",
        "uniform": Path(args.run_root) / "sleep_uniform",
        "reflection": Path(args.run_root) / "sleep_reflection",
    }
    run_dirs = {k: v for k, v in run_dirs.items() if v.exists()}
    costs = sleep_cost(run_dirs)

    fig_dir = ensure_dir(Path(args.report_dir) / "figures")
    plot_headline(rows, fig_dir / "headline_retention.png")
    if run_dirs:
        plot_ce_curves(run_dirs, fig_dir / "ce_curves.png")

    table = to_markdown(rows)
    body = ["# Results", "", "## Retention", "", table, ""]
    casc = next((r for r in rows if r["label"] == "cascading"), None)
    if casc is not None and casc.get("n_evicted", 0) == 0:
        warning = ("The compactor evicted no probed fact, so `acc(evicted)` is undefined and "
                   "this table cannot separate the methods. Lower `compaction.keep_frac` or "
                   "lengthen trajectories and rerun.")
        body += [f"> **Configuration is degenerate.** {warning}", ""]
        print(f"\nWARNING: {warning}")

    full = next((r for r in rows if r["label"] == "full"), None)
    if casc is not None and full is not None and full["retention_accuracy"] < casc["retention_accuracy"]:
        warning = ("The full-context arm scores below cascading, so it is not acting as a "
                   "ceiling. The backbone is too small to use the long context; scale up "
                   "before reading (d) as an upper bound.")
        body += [f"> **Ceiling inverted.** {warning}", ""]
        print(f"\nWARNING: {warning}")
    if costs:
        body += ["## Consolidation cost", "", "| method | phases | total GPU-s | GPU-s/phase |",
                 "|---|---|---|---|"]
        for k, v in costs.items():
            body.append(
                f"| {PRETTY.get(k, k)} | {v['phases']} | {v['total_gpu_seconds']:.1f} | "
                f"{v['mean_gpu_seconds_per_phase']:.1f} |"
            )
        body.append("")
    body += ["## Figures", "", "![headline](../artifacts/runs/report/figures/headline_retention.png)", "",
             "![ce](../artifacts/runs/report/figures/ce_curves.png)", ""]

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text("\n".join(body), encoding="utf-8")
    write_json(Path(args.report_dir) / "summary_all.json", {"rows": rows, "costs": costs})
    print(table)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
