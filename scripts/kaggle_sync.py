import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import ensure_dir

KEEP_PATTERNS = ["metrics.jsonl", "state.json", "config.json", "summary.json",
                 "result.json", "*.summary.json", "*_events.jsonl", "*_contexts.jsonl",
                 "*_reflections.jsonl", "scores_*.jsonl"]


def prune_phase_dirs(run_root, keep_last=1):
    parents = {p.parent for p in Path(run_root).rglob("phase_*") if p.is_dir()}
    removed = 0
    for parent in sorted(parents):
        siblings = sorted((p for p in parent.glob("phase_*") if p.is_dir()),
                          key=lambda p: p.name)
        doomed = siblings[:-keep_last] if keep_last else siblings
        for old in doomed:
            shutil.rmtree(old)
            removed += 1
    return removed


def save(run_root, out, keep_last, prune):
    run_root = Path(run_root)
    if not run_root.exists():
        raise SystemExit(f"nothing at {run_root}")
    if prune:
        n = prune_phase_dirs(run_root, keep_last)
        print(f"pruned {n} old phase directories")
    out = Path(out)
    ensure_dir(out.parent)
    archive = shutil.make_archive(str(out.with_suffix("")), "zip", str(run_root))
    size = Path(archive).stat().st_size / 1e6
    print(f"saved {run_root} -> {archive} ({size:.1f} MB)")


def restore(archive, run_root):
    archive = Path(archive)
    if not archive.exists():
        print(f"no archive at {archive}, starting fresh")
        return
    run_root = ensure_dir(run_root)
    shutil.unpack_archive(str(archive), str(run_root))
    print(f"restored {archive} -> {run_root}")
    for p in sorted(run_root.glob("*")):
        print(f"  {p.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["save", "restore"])
    ap.add_argument("--run-root", default="/kaggle/working/artifacts/runs")
    ap.add_argument("--archive", default="/kaggle/working/runs.zip")
    ap.add_argument("--keep-last", type=int, default=1)
    ap.add_argument("--prune", action="store_true")
    args = ap.parse_args()

    if args.action == "save":
        save(args.run_root, args.archive, args.keep_last, args.prune)
    else:
        restore(args.archive, args.run_root)


if __name__ == "__main__":
    main()
