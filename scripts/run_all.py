import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import load_config


def sh(cmd, dry):
    print("$ " + " ".join(str(c) for c in cmd))
    if dry:
        return
    subprocess.run([sys.executable, "-u"] + [str(c) for c in cmd], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--stages", default="data,compact,reflect,sleep,eval,report")
    ap.add_argument("--methods", default="compaction,uniform,reflection")
    ap.add_argument("--mask-head", action="store_true")
    ap.add_argument("--max-phases", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--n-turns", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    root = Path(cfg["run_root"])
    stages = set(args.stages.split(","))
    methods = [m for m in args.methods.split(",") if m]
    dry = args.dry_run

    casc = root / "cascading"
    report_dir = root / "report"

    if "data" in stages:
        sh(["data/generate_synthetic.py", "--out", cfg["data"]["dir"],
            "--n-train", args.n_train, "--n-eval", args.n_eval,
            "--n-turns", args.n_turns, "--seed", cfg["seed"]], dry)

    if "compact" in stages:
        sh(["baselines/cascading.py", "--config", args.config, "--split", "both"], dry)
        sh(["eval/span_report.py", "--events", casc / "train_events.jsonl",
            "--out", casc / "train_span_report.json", "--show", 0], dry)
        sh(["baselines/full_context.py", "--config", args.config, "--split", "eval", "--mode", "full"], dry)
        sh(["baselines/full_context.py", "--config", args.config, "--split", "eval", "--mode", "none"], dry)

    if "reflect" in stages and "reflection" in methods:
        sh(["baselines/reflection.py", "--config", args.config,
            "--events", casc / "train_events.jsonl",
            "--out", casc / "train_reflections.jsonl"], dry)

    sleep_runs = {}
    for method in methods:
        tag = method
        run_dir = root / f"sleep_{tag}"
        sleep_runs[tag] = run_dir
        if "sleep" not in stages:
            continue
        cmd = ["sleep/loop.py", "--config", args.config, "--method", method,
               "--events", casc / "train_events.jsonl",
               "--val-events", casc / "eval_events.jsonl",
               "--run-dir", run_dir]
        if method == "reflection":
            cmd += ["--reflections", casc / "train_reflections.jsonl"]
        if args.max_phases:
            cmd += ["--max-phases", args.max_phases]
        sh(cmd, dry)

    if args.mask_head:
        run_dir = root / "sleep_compaction+mask"
        sleep_runs["compaction+mask"] = run_dir
        if "sleep" in stages:
            cmd = ["sleep/loop.py", "--config", args.config, "--method", "compaction",
                   "--events", casc / "train_events.jsonl",
                   "--val-events", casc / "eval_events.jsonl",
                   "--run-dir", run_dir, "--mask-head"]
            if args.max_phases:
                cmd += ["--max-phases", args.max_phases]
            sh(cmd, dry)

    if "eval" in stages:
        arms = [
            ("floor", root / "none_context" / "eval_contexts.jsonl", None),
            ("cascading", casc / "eval_contexts.jsonl", None),
            ("full", root / "full_context" / "eval_contexts.jsonl", None),
        ]
        name_map = {"compaction": "ours", "compaction+mask": "ours+mask",
                    "uniform": "uniform", "reflection": "reflection"}
        for tag, run_dir in sleep_runs.items():
            arms.append((name_map.get(tag, tag), casc / "eval_contexts.jsonl",
                         run_dir / "latest" / "adapter"))
        for label, contexts, adapter in arms:
            cmd = ["eval/retention.py", "--config", args.config, "--contexts", contexts,
                   "--label", label, "--out", report_dir / label]
            if adapter is not None:
                cmd += ["--adapter", adapter]
            sh(cmd, dry)

    if "report" in stages:
        sh(["eval/report.py", "--report-dir", report_dir, "--run-root", root,
            "--out", "docs/results.md"], dry)


if __name__ == "__main__":
    main()
