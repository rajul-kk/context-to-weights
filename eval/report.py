import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, read_json, read_jsonl, write_json

ORDER = ["floor", "cascading", "uniform", "coverage", "reflection", "ours", "ours+mask",
         "oracle-closedbook", "oracle", "full"]
PRETTY = {
    "floor": "no context (floor)",
    "cascading": "(c) cascading, no update",
    "uniform": "(a) uniform replay",
    "coverage": "coverage: salience within fact keys (U6)",
    "reflection": "(b) reflection",
    "ours": "ours: compaction-supervised",
    "ours+mask": "ours + mask head",
    "oracle": "oracle: facts + augmentation",
    "oracle-closedbook": "oracle, closed book",
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


def _is_nan(v):
    return v is None or (isinstance(v, float) and v != v)


SCORE_COLUMNS = [
    ("label", "method"),
    ("mc_accuracy", "MC acc"),
    ("mc_accuracy_evicted", "MC acc(evicted)"),
    ("chance", "chance"),
    ("mean_margin_evicted", "margin(ev)"),
    ("evicted_sigma_over_chance", "sigma>chance"),
]


def _sorted(rows):
    rows.sort(key=lambda r: ORDER.index(r["label"]) if r["label"] in ORDER else 99)
    return rows


def collect(report_dir):
    rows = [read_json(p) for p in sorted(Path(report_dir).glob("*.summary.json"))
            if not p.name.startswith(("score_", "abstain_"))]
    return _sorted(rows)


def collect_abstain(report_dir):
    return [read_json(p) for p in sorted(Path(report_dir).glob("abstain_*.summary.json"))]


def collect_scores(report_dir):
    rows = [read_json(p) for p in sorted(Path(report_dir).glob("score_*.summary.json"))]
    return _sorted(rows)


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


def to_markdown(rows, columns=COLUMNS):
    head = "| " + " | ".join(h for _, h in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [head, rule]
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r.get(key)
            if key == "label":
                cells.append(PRETTY.get(v, v))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _se(acc, n):
    if not n or _is_nan(acc):
        return 0.0
    return (acc * (1 - acc) / n) ** 0.5


def plot_headline(rows, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [PRETTY.get(r["label"], r["label"]) for r in rows]
    x = range(len(rows))
    over = [0.0 if _is_nan(r.get("retention_accuracy")) else r["retention_accuracy"] for r in rows]
    over_se = [_se(r.get("retention_accuracy"), r.get("n")) for r in rows]
    evi = [0.0 if _is_nan(r.get("evicted_accuracy")) else r["evicted_accuracy"] for r in rows]
    evi_se = [_se(r.get("evicted_accuracy"), r.get("n_evicted")) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, vals, ses, title in (
        (axes[0], over, over_se, "all probes"),
        (axes[1], evi, evi_se, "evicted probes (answer not in retained context)"),
    ):
        bars = ax.bar(x, vals, yerr=ses, capsize=3, color="#8c9bb0")
        for i, r in enumerate(rows):
            if r["label"].startswith("ours"):
                bars[i].set_color("#c44e52")
            elif r["label"].startswith("oracle"):
                bars[i].set_color("#55a868")
            elif r["label"] in ("cascading", "full"):
                bars[i].set_color("#4c72b0")
        base = next((v for v, r in zip(vals, rows) if r["label"] == "cascading"), None)
        if base is not None:
            ax.axhline(base, ls="--", lw=1, color="#333",
                       label="no-adapter baseline")
            ax.legend(fontsize=8, loc="upper left")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("retention accuracy")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, max(0.4, max(vals) * 1.25))
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


SLEEP_RUNS = {
    "ours": "sleep_compaction",
    "ours+mask": "sleep_compaction+mask",
    "uniform": "sleep_uniform",
    "coverage": "sleep_coverage",
    "reflection": "sleep_reflection",
}
ADAPTERS = ("ours", "uniform", "coverage", "reflection", "ours+mask")
DEGENERATE = ("The compactor evicted no probed fact, so `acc(evicted)` is undefined and "
              "this table cannot separate the methods. Lower `compaction.keep_frac` or "
              "lengthen trajectories and rerun.")
CEILING_INVERTED = ("The full-context arm scores below cascading, so it is not acting as a "
                    "ceiling. The backbone is too small to use the long context; scale up "
                    "before reading (d) as an upper bound.")


def _row(rows, label):
    return next((r for r in rows if r["label"] == label), None)


def existing_sleep_runs(run_root):
    dirs = {k: Path(run_root) / v for k, v in SLEEP_RUNS.items()}
    return {k: v for k, v in dirs.items() if v.exists()}


def header_section(rows, run_root):
    cfg_path = Path(run_root) / "cascading" / "config.json"
    if not cfg_path.exists():
        return []
    run_cfg = read_json(cfg_path)
    target = next((r["model"] for r in rows if r.get("model")), None)
    compactor = run_cfg["model"]["base"]
    who = (f"Compactor `{compactor}`, consolidation and eval backbone `{target}`"
           if target and target != compactor else f"Backbone `{compactor}`")
    comp = run_cfg["compaction"]
    lines = [f"{who}, compaction backend `{comp['backend']}`, keep_frac {comp['keep_frac']}, "
             f"context budget {comp['context_budget']}.", ""]
    if target is None:
        lines += ["> Eval summaries predate the `model` field, so the backbone above is the "
                  "compactor's and may not be the model that was evaluated.", ""]
    return lines


def retention_warnings(rows):
    casc, full = _row(rows, "cascading"), _row(rows, "full")
    out = []
    if casc is not None and casc.get("n_evicted", 0) == 0:
        out.append(("Configuration is degenerate.", DEGENERATE))
    if casc is not None and full is not None and full["retention_accuracy"] < casc["retention_accuracy"]:
        out.append(("Ceiling inverted.", CEILING_INVERTED))
    return out


def retention_section(table, warnings):
    lines = ["## Retention", "", table, ""]
    for title, text in warnings:
        lines += [f"> **{title}** {text}", ""]
    return lines


def cost_section(costs):
    if not costs:
        return []
    lines = ["## Consolidation cost", "", "| method | phases | total GPU-s | GPU-s/phase |", "|---|---|---|---|"]
    for k, v in costs.items():
        lines.append(f"| {PRETTY.get(k, k)} | {v['phases']} | {v['total_gpu_seconds']:.1f} | "
                     f"{v['mean_gpu_seconds_per_phase']:.1f} |")
    return lines + [""]


def _oracle_reading(orc, casc):
    oe = orc.get("evicted_accuracy", 0.0)
    ce = casc.get("evicted_accuracy", 0.0)
    se = _se(oe, orc.get("n_evicted")) + _se(ce, casc.get("n_evicted"))
    if _is_nan(oe) or oe - ce <= max(se, 0.02):
        return [f"**The oracle control does not clear the baseline** ({oe:.3f} vs {ce:.3f} "
                f"on evicted probes). Perfect fact selection plus 20 augmented surface "
                f"forms per fact still recovers nothing, so this configuration cannot "
                f"register a consolidation win at all and the comparisons between the "
                f"other arms are uninformative about the compaction signal.", ""]
    return [f"**The oracle control clears the baseline** ({oe:.3f} vs {ce:.3f} on "
            f"evicted probes). The setup can register a consolidation win, so the "
            f"gap between the other arms and the oracle is a real measure of what "
            f"the compaction signal costs relative to perfect selection.", ""]


def _ours_vs_uniform(ours, uni):
    d = ours.get("evicted_accuracy", 0.0) - uni.get("evicted_accuracy", 0.0)
    se = (_se(ours.get("evicted_accuracy"), ours.get("n_evicted"))
          + _se(uni.get("evicted_accuracy"), uni.get("n_evicted")))
    verdict = "within noise" if abs(d) < se else ("ours ahead" if d > 0 else "uniform ahead")
    return [f"ours vs uniform on evicted probes: {ours.get('evicted_accuracy', 0.0):.3f} "
            f"vs {uni.get('evicted_accuracy', 0.0):.3f}, difference {d:+.3f} "
            f"(combined SE {se:.3f}) - **{verdict}**.", ""]


def reading_section(rows):
    casc = _row(rows, "cascading")
    adapters = [r for r in rows if r["label"] in ADAPTERS]
    if not (casc and adapters):
        return []
    base = casc.get("retention_accuracy", 0.0)
    best = max(adapters, key=lambda r: r.get("retention_accuracy", 0.0))
    lines = ["## Reading", ""]
    if best.get("retention_accuracy", 0.0) < base:
        lines += [f"No consolidated adapter beats the no-adapter baseline "
                  f"({base:.3f} all-probe accuracy). The best adapter is "
                  f"{PRETTY.get(best['label'], best['label'])} at "
                  f"{best.get('retention_accuracy', 0.0):.3f}. On this run the sleep pass is "
                  f"net-negative: it costs more than the knowledge it adds.", ""]
    orc = _row(rows, "oracle")
    if orc is not None:
        lines += _oracle_reading(orc, casc)
    ours, uni = _row(adapters, "ours"), _row(adapters, "uniform")
    if ours and uni:
        lines += _ours_vs_uniform(ours, uni)
    return lines


def abstention_section(abst):
    if not abst:
        return []
    lines = ["## Abstention arm", "",
             "The same compaction mask, asked to refuse when the fact was evicted rather "
             "than to recall it.", "",
             "| arm | refuses when evicted | refuses when retained | accuracy retained | "
             "hallucinates when evicted |", "|---|---|---|---|---|"]
    for r in sorted(abst, key=lambda r: r["label"]):
        lines.append(f"| {r['label']} | {r['abstain_rate_evicted']:.3f} | "
                     f"{r['abstain_rate_retained']:.3f} | {r['accuracy_retained']:.3f} | "
                     f"{r['hallucination_rate_evicted']:.3f} |")
    lines += [""]
    base = _row(abst, "no-adapter")
    best = max(abst, key=lambda r: r.get("abstention_margin", 0.0))
    if base is not None and best is not base:
        was, now = base["hallucination_rate_evicted"], best["hallucination_rate_evicted"]
        rel = 100 * (was - now) / was if was else 0.0
        lines += [f"`{best['label']}` cuts hallucination on evicted probes by {rel:.1f}% "
                  f"({was:.3f} to {now:.3f}) while refusing "
                  f"{best['abstain_rate_retained']:.3f} of still-answerable probes and "
                  f"holding retained accuracy at {best['accuracy_retained']:.3f}.", ""]
    return lines


def scores_section(scores):
    if not scores:
        return []
    lines = ["## Scoring eval (CE ranking)", "",
             "Ranks the gold answer against distractors drawn from the same fact bank and "
             "scores a hit when gold has the lowest cross-entropy. This separates *is the "
             "knowledge in the weights* from *can the greedy decoder say it*.", "",
             to_markdown(scores, SCORE_COLUMNS), ""]
    ev = [r for r in scores if not _is_nan(r.get("evicted_sigma_over_chance"))]
    if ev:
        best = max(ev, key=lambda r: r["evicted_sigma_over_chance"])
        if best["evicted_sigma_over_chance"] < 2.0:
            lines += [f"**No arm ranks evicted answers above chance.** The best is "
                      f"{PRETTY.get(best['label'], best['label'])} at "
                      f"{best['mc_accuracy_evicted']:.3f} against a chance rate of "
                      f"{best['chance']:.3f} ({best['evicted_sigma_over_chance']:+.2f} sigma). "
                      f"A drop in evicted CE is therefore not evidence of knowledge transfer: "
                      f"the adapter lowers loss on the gold answer and on its distractors alike, "
                      f"having learned the answer vocabulary without the binding.", ""]
    casc = _row(scores, "cascading")
    if casc is not None:
        worse = [r for r in scores if r["label"] != "cascading"
                 and not _is_nan(r.get("mc_accuracy"))
                 and r["mc_accuracy"] < casc.get("mc_accuracy", 0.0)]
        if len(worse) == len(scores) - 1:
            lines += [f"Every adapter also ranks *retained* answers worse than no adapter "
                      f"({casc['mc_accuracy']:.3f} all-probe), so the sleep pass degrades "
                      f"the backbone rather than trading retained accuracy for evicted.", ""]
    return lines


FIGURES = ["## Figures", "",
           "![retention](figures/headline_retention.png)", "",
           "*Left: accuracy on all held-out probes. Right: accuracy restricted to probes "
           "whose answer is absent from the retained context. Dashed line is the no-adapter "
           "baseline; error bars are binomial standard error.*", "",
           "![ce](figures/ce_curves.png)", "",
           "*Median (left) and mean (right) per-token validation cross-entropy across sleep "
           "phases. Rising median CE indicates the adapter is getting worse, not better.*", ""]


def build_report(rows, table, warnings, costs, abst, scores, run_root):
    return (["# Results", ""] + header_section(rows, run_root) + retention_section(table, warnings)
            + cost_section(costs) + reading_section(rows) + abstention_section(abst)
            + scores_section(scores) + FIGURES)


def plot_figures(rows, run_dirs, report_dir):
    fig_dir = ensure_dir(Path(report_dir) / "figures")
    plot_headline(rows, fig_dir / "headline_retention.png")
    if run_dirs:
        plot_ce_curves(run_dirs, fig_dir / "ce_curves.png")


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
    run_dirs = existing_sleep_runs(args.run_root)
    costs = sleep_cost(run_dirs)
    plot_figures(rows, run_dirs, args.report_dir)
    table = to_markdown(rows)
    warnings = retention_warnings(rows)
    for _, text in warnings:
        print(f"\nWARNING: {text}")
    body = build_report(rows, table, warnings, costs, collect_abstain(args.report_dir),
                        collect_scores(args.report_dir), args.run_root)

    out = Path(args.out)
    ensure_dir(out.parent)
    out.write_text("\n".join(body), encoding="utf-8")
    write_json(Path(args.report_dir) / "summary_all.json", {"rows": rows, "costs": costs})
    print(table)
    print(f"\nwrote -> {out}")


if __name__ == "__main__":
    main()
