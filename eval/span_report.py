import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import read_jsonl, write_json


def summarize(events):
    n_spans = 0
    n_kept = 0
    fact_total = 0
    fact_kept = 0
    ratios = []
    per_key = Counter()
    per_key_kept = Counter()

    for ev in events:
        ratios.append(ev["compaction_ratio"])
        for s in ev["spans"]:
            n_spans += 1
            n_kept += int(s["kept"])
            key = s.get("carries_fact")
            if key:
                fact_total += 1
                per_key[key] += 1
                if s["kept"]:
                    fact_kept += 1
                    per_key_kept[key] += 1

    filler_total = n_spans - fact_total
    filler_kept = n_kept - fact_kept
    return {
        "n_events": len(events),
        "n_spans": n_spans,
        "keep_rate": _div(n_kept, n_spans),
        "mean_compaction_ratio": _div(sum(ratios), len(ratios)),
        "fact_spans": fact_total,
        "fact_keep_rate": _div(fact_kept, fact_total),
        "filler_spans": filler_total,
        "filler_keep_rate": _div(filler_kept, filler_total),
        "salience_lift": _lift(_div(fact_kept, fact_total), _div(filler_kept, filler_total)),
        "per_fact_keep_rate": {k: _div(per_key_kept[k], per_key[k]) for k in sorted(per_key)},
    }


def _div(a, b):
    return float(a) / b if b else 0.0


def _lift(fact_rate, filler_rate):
    if filler_rate <= 0.0:
        return float("inf") if fact_rate > 0.0 else 0.0
    return fact_rate / filler_rate


def positional_lift(events):
    fact_kept = fact_total = filler_kept = filler_total = 0
    for ev in events:
        spans = ev["spans"]
        budget = sum(1 for s in spans if s["kept"])
        for i, s in enumerate(spans):
            hit = int(i < budget)
            if s.get("carries_fact"):
                fact_total += 1
                fact_kept += hit
            else:
                filler_total += 1
                filler_kept += hit
    fact_rate = _div(fact_kept, fact_total)
    filler_rate = _div(filler_kept, filler_total)
    return {
        "fact_keep_rate": fact_rate,
        "filler_keep_rate": filler_rate,
        "salience_lift": _lift(fact_rate, filler_rate),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=6)
    args = ap.parse_args()

    events = read_jsonl(args.events)
    report = summarize(events)

    provenance = Counter(e.get("decided_by", "unknown") for e in events)
    report["provenance"] = dict(provenance)
    if len(provenance) > 1:
        report["by_provenance"] = {
            src: summarize([e for e in events if e.get("decided_by") == src])
            for src in provenance
        }

    print(f"events                {report['n_events']}")
    print(f"spans                 {report['n_spans']}")
    print(f"keep rate             {report['keep_rate']:.3f}")
    print(f"mean compaction ratio {report['mean_compaction_ratio']:.3f}")
    print(f"fact-span keep rate   {report['fact_keep_rate']:.3f}  (n={report['fact_spans']})")
    print(f"filler-span keep rate {report['filler_keep_rate']:.3f}  (n={report['filler_spans']})")
    print(f"salience lift         {report['salience_lift']:.2f}x")
    print("per-fact keep rate:")
    for k, v in report["per_fact_keep_rate"].items():
        print(f"  {k:<16} {v:.3f}")

    if 0 < report["fact_spans"] < 100:
        print(f"\nNOTE: only {report['fact_spans']} fact spans. At a keep rate of "
              f"{report['keep_rate']:.2f} the fact keep rate carries roughly "
              f"+/-{(report['keep_rate'] * (1 - report['keep_rate']) / report['fact_spans']) ** 0.5:.3f} "
              "of sampling noise.\nUse more eval trajectories before treating a lift near 1.0 "
              "as a real effect.")

    pos = positional_lift(events)
    report["positional_control"] = pos
    print(f"\npositional control (keep the first N spans):")
    print(f"  fact {pos['fact_keep_rate']:.3f}  filler {pos['filler_keep_rate']:.3f}  "
          f"lift {pos['salience_lift']:.2f}x")
    if report["fact_spans"] and report["salience_lift"] <= pos["salience_lift"] * 1.1:
        print("\nWARNING: the compactor does not beat 'keep the first N spans'. Its lift is a\n"
              "positional artifact of where facts sit in the trajectory, not judgment. Shuffle\n"
              "the spans shown to the compactor and confirm the lift survives.")

    if report.get("by_provenance"):
        print("\nby decision source:")
        for src, sub in sorted(report["by_provenance"].items()):
            print(f"  {src:<20} events {sub['n_events']:>4}  fact {sub['fact_keep_rate']:.3f}  "
                  f"filler {sub['filler_keep_rate']:.3f}  lift {sub['salience_lift']:.2f}x")
        model = report["by_provenance"].get("model")
        if model and model["fact_spans"]:
            print(f"\nModel-decided lift is {model['salience_lift']:.2f}x. "
                  "This is the only honest number;\nthe aggregate above is contaminated by "
                  "heuristic fallbacks.")

    if report["fact_spans"] and report["salience_lift"] <= 1.0:
        print("\nWARNING: salience lift is at or below 1.0. This compactor keeps probed facts "
              "no more often than filler,\nso its keep/drop decision carries no supervision. "
              "Compaction-supervised consolidation cannot\nbeat uniform replay under this "
              "compactor. Use a stronger compactor model before running the\nmain comparison.")

    if args.show and events:
        print("\nsample kept spans:")
        shown = 0
        for ev in events:
            for s in ev["spans"]:
                if s["kept"] and shown < args.show:
                    print(f"  + {s['text'][:110]}")
                    shown += 1
        print("sample dropped spans:")
        shown = 0
        for ev in events:
            for s in ev["spans"]:
                if not s["kept"] and shown < args.show:
                    print(f"  - {s['text'][:110]}")
                    shown += 1

    if args.out:
        write_json(args.out, report)


if __name__ == "__main__":
    main()
