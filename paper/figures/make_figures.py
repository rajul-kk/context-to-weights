import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = json.load(open(HERE / "data.json"))
NO_DATE = {"CreationDate": None, "Creator": None, "Producer": None}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})

STYLE = {
    "read": ("#c0392b", "o", "read (logit relevance)"),
    "attention_late": ("#2471a3", "s", "attention, late layers"),
    "attention": ("#85c1e9", "^", "attention, all layers"),
    "generate": ("#7f8c8d", "D", "generate (stated FOCUS)"),
}


def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def routing():
    scales = ["0.5B", "1.5B", "7B"]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4))
    for ax, key, label in ((axes[0], "hit_rate", "hit rate"),
                           (axes[1], "content_dependence", "content dependence")):
        for mode, (color, marker, name) in STYLE.items():
            stats = [mean_sd([r[key] for r in DATA["routing"][s][mode]]) for s in scales]
            ax.errorbar(range(3), [m for m, _ in stats], yerr=[sd for _, sd in stats], color=color,
                        marker=marker, markersize=4, linewidth=1.2, capsize=2, label=name)
        ax.set_xticks(range(3), scales)
        ax.set_xlabel("Qwen2.5-Instruct")
        ax.set_ylabel(label)
        ax.set_xlim(-0.3, 2.3)
    axes[0].axhline(DATA["routing_chance"], color="black", linestyle=":", linewidth=0.8)
    axes[0].text(2.28, DATA["routing_chance"] + 0.006, "chance", ha="right", fontsize=7)
    axes[1].axhline(0, color="black", linestyle=":", linewidth=0.8)
    axes[1].legend(loc="upper left", fontsize=7, bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(HERE / "fig_routing.pdf", bbox_inches="tight", metadata=NO_DATE)


def slopes(ax, columns, labels, colors, ylabel):
    n = len(columns[0])
    for i in range(n):
        ax.plot(range(len(columns)), [c[i] for c in columns], color="#b3b3b3", linewidth=0.8, zorder=1)
    for x, (col, color) in enumerate(zip(columns, colors)):
        ax.scatter([x] * n, col, color=color, s=16, zorder=2)
        ax.plot([x - 0.18, x + 0.18], [statistics.mean(col)] * 2, color="black", linewidth=1.6, zorder=3)
    ax.set_xticks(range(len(columns)), labels)
    ax.set_xlim(-0.4, len(columns) - 0.6)
    ax.set_ylabel(ylabel)


def rates(counts, totals):
    totals = totals if isinstance(totals, list) else [totals] * len(counts)
    return [c / n for c, n in zip(counts, totals)]


def training():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), gridspec_kw={"width_ratios": [2, 3]})
    c = DATA["consolidation"]
    slopes(axes[0], [rates(c["compaction_recovered"], c["n_evicted"]),
                     rates(c["uniform_recovered"], c["n_evicted"])],
           ["compaction-gated", "uniform"], ["#7f8c8d", "#c0392b"], "evicted facts recovered")
    axes[0].set_title("(a) consolidation, five corpora\npaired t(4) = 3.10", fontsize=8.5)
    d = DATA["distillation"]
    n = d["n_tasks"]
    slopes(axes[1], [rates(d["kl_top"], n), rates(d["random"], n), rates(d["uniform"], n)],
           ["KL-gated", "random subset", "uniform"], ["#2471a3", "#7f8c8d", "#c0392b"],
           "held-out task pass rate")
    axes[1].set_title("(b) distillation, five seeds\nKL vs random: t(4) = 0.74", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(HERE / "fig_training.pdf", bbox_inches="tight", metadata=NO_DATE)


if __name__ == "__main__":
    routing()
    training()
