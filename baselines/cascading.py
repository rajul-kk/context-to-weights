import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import (ensure_dir, load_config, parse_overrides, read_jsonl, set_seed,
                       write_json, write_jsonl)
from common.schema import Trajectory
from compactor.base import HeuristicCompactor, ModelCompactor
from compactor.runner import CompactionRunner


def build_compactor(cfg, tokenizer=None, model=None):
    backend = cfg["compaction"]["backend"]
    if backend == "heuristic":
        return HeuristicCompactor(), None
    from sleep.lm import load_backbone, make_generator

    if model is None:
        model, tokenizer = load_backbone(cfg)
    gen = make_generator(model, tokenizer)
    return ModelCompactor(gen, max_new_tokens=cfg["compaction"]["max_new_tokens"]), tokenizer


def run_split(cfg, split, out_dir, tokenizer=None, compactor=None):
    data_dir = Path(cfg["data"]["dir"])
    rows = read_jsonl(data_dir / cfg["data"][split])
    if compactor is None:
        compactor, tokenizer = build_compactor(cfg, tokenizer)

    runner = CompactionRunner(
        compactor=compactor,
        context_budget=cfg["compaction"]["context_budget"],
        keep_recent_turns=cfg["compaction"]["keep_recent_turns"],
        keep_frac=cfg["compaction"]["keep_frac"],
        granularity=cfg["compaction"]["granularity"],
        tokenizer=tokenizer,
    )

    event_rows = []
    context_rows = []
    for row in rows:
        traj = Trajectory.from_dict(row)
        events, final_context = runner.run(traj)
        for ev in events:
            event_rows.append(ev.to_dict())
        context_rows.append(
            {
                "traj_id": traj.traj_id,
                "context": final_context,
                "n_events": len(events),
                "probes": [p.to_dict() for p in traj.probes],
            }
        )

    out_dir = ensure_dir(out_dir)
    write_jsonl(out_dir / f"{split}_events.jsonl", event_rows)
    write_jsonl(out_dir / f"{split}_contexts.jsonl", context_rows)
    return event_rows, context_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="eval", choices=["train", "eval", "both"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg["seed"])
    out_dir = Path(args.out or Path(cfg["run_root"]) / "cascading")
    write_json(out_dir / "config.json", cfg)

    splits = ["train", "eval"] if args.split == "both" else [args.split]
    compactor, tokenizer = build_compactor(cfg)
    for split in splits:
        events, contexts = run_split(cfg, split, out_dir, tokenizer, compactor)
        ratios = [e["compaction_ratio"] for e in events] or [0.0]
        print(
            f"{split}: {len(contexts)} trajectories, {len(events)} compaction events, "
            f"mean ratio {sum(ratios) / len(ratios):.3f}"
        )
    if getattr(compactor, "calls", 0):
        empty_rate = compactor.empty_keeps / compactor.calls
        print(f"compactor: {compactor.calls} calls, {compactor.fallbacks} heuristic fallbacks "
              f"({compactor.fallback_rate:.1%}), {compactor.empty_keeps} empty keeps "
              f"({empty_rate:.1%})")
        if empty_rate > 0.2:
            print("WARNING: the compactor kept nothing on a large share of events. An empty keep "
                  "set yields no\nSFT examples, so those events contribute no supervision at all. "
                  "Check the keep budget and\nthe prompt before running the comparison.")
        if compactor.fallback_rate > 0.0:
            dump = out_dir / "failed_replies.jsonl"
            write_jsonl(dump, [{"reply": r} for r in compactor.failed_replies])
            print("WARNING: the model compactor failed to parse and fell back to the heuristic.\n"
                  "Results under this run are not purely model-decided. Sample of failures "
                  f"written to {dump}. First failure:\n"
                  f"{(compactor.failed_replies[0] if compactor.failed_replies else '')[:400]!r}")
    print(f"wrote -> {out_dir}")


if __name__ == "__main__":
    main()
