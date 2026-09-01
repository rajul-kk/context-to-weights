import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import read_jsonl, write_json


def required_hit_rate(rows, top_frac):
    hit = 0
    total = 0
    for r in rows:
        req = r.get("required") or []
        if not req:
            continue
        spans = sorted(r["spans"], key=lambda s: -s["kl_mean"])
        k = max(1, round(len(spans) * top_frac))
        top_text = " ".join(s["text"] for s in spans[:k])
        for token in req:
            total += 1
            hit += int(token in top_text)
    return hit / total if total else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--skills-root", default="skills/toy")
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--show", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = read_jsonl(args.scores)
    from kl_gate.skillset import load_all

    demos = {}
    for skill in load_all(args.skills_root):
        for i, d in enumerate(skill["demos"]):
            demos[(skill["name"], i)] = d
    for r in rows:
        r["required"] = demos.get((r["skill"], r["demo_idx"]), {}).get("required", [])

    all_kl = [k for r in rows for k in r["kl"]]
    report = {
        "n_demos": len(rows),
        "n_tokens": len(all_kl),
        "kl_median": statistics.median(all_kl) if all_kl else float("nan"),
        "kl_mean": sum(all_kl) / len(all_kl) if all_kl else float("nan"),
        "kl_p90": statistics.quantiles(all_kl, n=10)[-1] if len(all_kl) > 10 else float("nan"),
        "required_in_top_frac": required_hit_rate(rows, args.top_frac),
        "top_frac": args.top_frac,
    }
    for k, v in report.items():
        print(f"{k:<22} {v}")

    print("\nhighest-KL spans:")
    spans = [(s["kl_mean"], r["skill"], s["text"]) for r in rows for s in r["spans"]]
    spans.sort(reverse=True)
    for kl, skill, text in spans[: args.show * 3]:
        print(f"  {kl:7.3f}  [{skill}] {text.strip()[:90]!r}")

    print("\nlowest-KL spans:")
    for kl, skill, text in spans[-args.show * 3:]:
        print(f"  {kl:7.3f}  [{skill}] {text.strip()[:90]!r}")

    print("\nper-demo token view:")
    for r in rows[: args.show]:
        pairs = sorted(zip(r["kl"], r["token_text"]), reverse=True)[:10]
        rendered = ", ".join(f"{t.strip()!r}:{k:.2f}" for k, t in pairs)
        print(f"  [{r['skill']} d{r['demo_idx']} {r['kind']}] {rendered}")

    if args.out:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
