import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.stats import holm, paired_t, t_two_sided_p

ROOT = Path(__file__).resolve().parents[1]
SCALES = ("0.5B", "1.5B", "7B")
ALPHA = 0.05


def _series(routing, scale, mode, key="hit_rate"):
    return [r[key] for r in routing[scale][mode]]


def _better_attention(routing, scale):
    return max(("attention", "attention_late"), key=lambda m: sum(_series(routing, scale, m)))


def _rates(counts, totals):
    totals = totals if isinstance(totals, list) else [totals] * len(counts)
    return [c / n for c, n in zip(counts, totals)]


def families(data):
    routing = data["routing"]
    rows = {"routing": []}
    for scale in SCALES:
        att = _better_attention(routing, scale)
        for other in (att, "generate"):
            a, b = _series(routing, scale, "read"), _series(routing, scale, other)
            rows["routing"].append((f"{scale} read vs {other}, hit rate", a, b))
    c = data["consolidation"]
    rows["consolidation"] = [(
        "uniform vs compaction, evicted facts recovered",
        _rates(c["uniform_recovered"], c["n_evicted"]),
        _rates(c["compaction_recovered"], c["n_evicted"]),
    )]
    d = data["distillation"]
    rows["distillation"] = [(
        "kl_top vs random, held-out pass rate",
        _rates(d["kl_top"], d["n_tasks"]),
        _rates(d["random"], d["n_tasks"]),
    )]
    out = {}
    for fam, tests in rows.items():
        stats = []
        for name, a, b in tests:
            mean, stat, dof = paired_t(a, b)
            stats.append({"comparison": name, "mean_diff": mean, "t": stat, "dof": dof,
                          "p": t_two_sided_p(stat, dof)})
        for s, adj in zip(stats, holm([s["p"] for s in stats])):
            s["p_holm"] = adj
            s["survives"] = adj < ALPHA
        out[fam] = stats
    return out


def to_markdown(fams):
    lines = ["| family | comparison | mean diff | t (dof) | p | Holm p | survives |",
             "|---|---|---|---|---|---|---|"]
    for fam, stats in fams.items():
        for s in stats:
            lines.append(f"| {fam} ({len(stats)}) | {s['comparison']} | {s['mean_diff']:+.3f} | "
                         f"{s['t']:.2f} ({s['dof']}) | {s['p']:.4f} | {s['p_holm']:.4f} | "
                         f"{'yes' if s['survives'] else 'no'} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "paper" / "figures" / "data.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fams = families(json.loads(Path(args.data).read_text(encoding="utf-8")))
    print(to_markdown(fams))
    if args.out:
        Path(args.out).write_text(json.dumps(fams, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
