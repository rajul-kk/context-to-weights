import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, load_config, read_jsonl, write_jsonl
from common.schema import Trajectory


def build_contexts(rows, mode, max_turns=0):
    out = []
    for row in rows:
        traj = Trajectory.from_dict(row)
        if mode == "none":
            context = []
        else:
            turns = traj.turns[-max_turns:] if max_turns else traj.turns
            context = [f"{t.role}: {t.content}" for t in turns]
        out.append(
            {
                "traj_id": traj.traj_id,
                "context": context,
                "n_events": 0,
                "probes": [p.to_dict() for p in traj.probes],
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--mode", default="full", choices=["full", "none"])
    ap.add_argument("--max-turns", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = read_jsonl(Path(cfg["data"]["dir"]) / cfg["data"][args.split])
    contexts = build_contexts(rows, args.mode, args.max_turns)

    out_dir = ensure_dir(args.out or Path(cfg["run_root"]) / f"{args.mode}_context")
    path = out_dir / f"{args.split}_contexts.jsonl"
    write_jsonl(path, contexts)
    print(f"{args.mode}: {len(contexts)} contexts -> {path}")


if __name__ == "__main__":
    main()
