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
    ap.add_argument("--fracs", default="0.1,0.25,0.5")
    ap.add_argument("--methods", default="compaction,uniform")
    ap.add_argument("--max-phases", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    root = Path(cfg["run_root"]) / "sweep"
    dry = args.dry_run

    for frac in [f.strip() for f in args.fracs.split(",") if f.strip()]:
        slug = f"kf{frac.replace('.', '')}"
        casc = root / slug / "cascading"
        sh(["baselines/cascading.py", "--config", args.config, "--split", "both",
            "--out", casc, "--set", f"compaction.keep_frac={frac}"], dry)
        sh(["eval/span_report.py", "--events", casc / "train_events.jsonl",
            "--out", casc / "train_span_report.json", "--show", 0], dry)

        for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
            run_dir = root / slug / f"sleep_{method}"
            cmd = ["sleep/loop.py", "--config", args.config, "--method", method,
                   "--events", casc / "train_events.jsonl",
                   "--val-events", casc / "eval_events.jsonl",
                   "--run-dir", run_dir]
            if args.max_phases:
                cmd += ["--max-phases", args.max_phases]
            sh(cmd, dry)
            label = {"compaction": "ours", "uniform": "uniform"}.get(method, method)
            sh(["eval/retention.py", "--config", args.config,
                "--contexts", casc / "eval_contexts.jsonl",
                "--label", label, "--adapter", run_dir / "latest" / "adapter",
                "--out", root / slug / "report" / label], dry)

        sh(["eval/retention.py", "--config", args.config,
            "--contexts", casc / "eval_contexts.jsonl", "--label", "cascading",
            "--out", root / slug / "report" / "cascading"], dry)
        sh(["eval/report.py", "--report-dir", root / slug / "report",
            "--run-root", root / slug, "--out", f"docs/results_{slug}.md"], dry)


if __name__ == "__main__":
    main()
