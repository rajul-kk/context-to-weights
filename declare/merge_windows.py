import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from declare.run import merge_summaries, summarize

SAME = ("label", "model", "seed", "n_regions", "kv_bits", "kv_scheme")


def merge_mode(windows, mode):
    parts = []
    for w in windows:
        s = read_json(Path(w) / f"{mode}.summary.json")
        parts.append((s.get("layout_offset", 0), s, read_jsonl(Path(w) / f"{mode}.records.jsonl")))
    parts.sort(key=lambda p: p[0])
    first = parts[0][1]
    expect = parts[0][0]
    records = []
    for offset, s, recs in parts:
        for k in SAME:
            if s.get(k) != first.get(k):
                raise SystemExit(f"{mode}: {k} differs between windows ({s.get(k)!r} vs {first.get(k)!r})")
        if offset != expect:
            raise SystemExit(f"{mode}: windows do not tile: expected offset {expect}, found {offset}")
        if len(recs) != s["n"]:
            raise SystemExit(f"{mode}: window at offset {offset} has {len(recs)} records but n={s['n']}")
        records.extend(recs)
        expect = offset + len(recs)
    summary = summarize(records, mode, first["n_regions"])
    for k in ("model", "seed", "kv_bits", "kv_scheme"):
        summary[k] = first.get(k)
    summary["elicitor_unparsed"] = sum(p[1].get("elicitor_unparsed", 0) for p in parts)
    summary["layout_offset"] = parts[0][0]
    summary["merged_windows"] = [[p[0], p[1]["n"]] for p in parts]
    return records, summary


def modes_in(window):
    return sorted(p.name[: -len(".summary.json")] for p in Path(window).glob("*.summary.json")
                  if p.name != "summary_all.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Join declare runs over consecutive layout windows (--offset/--take) into one run.")
    ap.add_argument("--out", required=True)
    ap.add_argument("windows", nargs="+")
    args = ap.parse_args(argv)
    modes = modes_in(args.windows[0])
    for w in args.windows[1:]:
        if modes_in(w) != modes:
            raise SystemExit(f"{w} has modes {modes_in(w)}, the first window has {modes}")
    out = ensure_dir(args.out)
    for mode in modes:
        records, summary = merge_mode(args.windows, mode)
        write_jsonl(out / f"{mode}.records.jsonl", records)
        write_json(out / f"{mode}.summary.json", summary)
        print(f"{mode}: {summary['n']} layouts from {len(args.windows)} windows, hit rate {summary['hit_rate']:.3f}")
    merge_summaries(out)


if __name__ == "__main__":
    main()
