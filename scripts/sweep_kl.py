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
    ap.add_argument("--config", default="configs/skill_base.yaml")
    ap.add_argument("--fracs", default="0.1,0.25,0.5")
    ap.add_argument("--granularities", default="span,token")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    root = Path(cfg["run_root"])
    dry = args.dry_run

    for gran in [g.strip() for g in args.granularities.split(",") if g.strip()]:
        scores = root / f"scores_{gran}.jsonl"
        if not (ROOT / scores).exists():
            sh(["kl_gate/score.py", "--config", args.config,
                "--skills-root", cfg["skills"]["root"], "--granularity", gran,
                "--out", scores], dry)

        for frac in [f.strip() for f in args.fracs.split(",") if f.strip()]:
            slug = f"sweep_{gran}_{frac}"
            run_dir = root / slug / "distill"
            report_dir = root / slug / "report"
            sh(["distill/train.py", "--config", args.config, "--scores", scores,
                "--policy", "kl_top", "--run-dir", run_dir, "--resume",
                "--set", f"gate.granularity={gran}", f"gate.top_frac={frac}"], dry)
            sh(["eval/skill_eval.py", "--config", args.config, "--run-dir", run_dir,
                "--doc-mode", "none", "--label", "ours",
                "--out", report_dir / "ours"], dry)


if __name__ == "__main__":
    main()
