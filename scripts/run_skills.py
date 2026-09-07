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
    ap.add_argument("--stages", default="skills,score,distill,eval,report")
    ap.add_argument("--policies", default="kl_top,uniform,random")
    ap.add_argument("--granularity", default="span")
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    root = Path(cfg["run_root"])
    stages = set(args.stages.split(","))
    policies = [p for p in args.policies.split(",") if p]
    dry = args.dry_run
    scores = root / f"scores_{args.granularity}.jsonl"
    report_dir = root / ("report" if args.granularity == "span" else f"report_{args.granularity}")

    if "skills" in stages:
        sh(["skills/generate_toy_skills.py", "--out", cfg["skills"]["root"],
            "--seed", cfg["seed"]], dry)

    if "score" in stages:
        sh(["kl_gate/score.py", "--config", args.config, "--skills-root", cfg["skills"]["root"],
            "--granularity", args.granularity, "--out", scores], dry)
        sh(["kl_gate/inspect_gate.py", "--scores", scores,
            "--skills-root", cfg["skills"]["root"], "--top-frac", args.top_frac,
            "--out", root / f"gate_report_{args.granularity}.json"], dry)

    runs = {}
    for policy in policies:
        tag = f"{policy}_{args.granularity}_{args.top_frac}"
        run_dir = root / f"distill_{tag}"
        runs[policy] = run_dir
        if "distill" not in stages:
            continue
        sh(["distill/train.py", "--config", args.config, "--scores", scores,
            "--policy", policy, "--run-dir", run_dir, "--resume",
            "--set", f"gate.granularity={args.granularity}", f"gate.top_frac={args.top_frac}"], dry)

    lim = ["--limit-tasks", args.limit_tasks] if args.limit_tasks else []
    if "eval" in stages:
        label_of = {"kl_top": "ours", "uniform": "s2l-uniform", "random": "random-control"}
        sh(["eval/skill_eval.py", "--config", args.config, "--doc-mode", "correct",
            "--label", "prompt-full", "--out", report_dir / "prompt-full"] + lim, dry)
        sh(["eval/skill_eval.py", "--config", args.config, "--doc-mode", "none",
            "--label", "no-skill", "--out", report_dir / "no-skill"] + lim, dry)
        sh(["eval/skill_eval.py", "--config", args.config, "--doc-mode", "mismatched",
            "--label", "prompt-mismatched", "--out", report_dir / "prompt-mismatched"] + lim, dry)
        for policy, run_dir in runs.items():
            label = label_of.get(policy, policy)
            sh(["eval/skill_eval.py", "--config", args.config, "--run-dir", run_dir,
                "--doc-mode", "none", "--label", label, "--out", report_dir / label] + lim, dry)
        sh(["eval/skill_eval.py", "--config", args.config, "--run-dir", runs["kl_top"],
            "--doc-mode", "mismatched", "--label", "ours-mismatched",
            "--out", report_dir / "ours-mismatched"] + lim, dry)

    if "report" in stages:
        out = "docs/results_skills.md" if args.granularity == "span" \
            else f"docs/results_skills_{args.granularity}.md"
        sh(["eval/skill_report.py", "--report-dir", report_dir, "--run-root", root,
            "--out", out], dry)


if __name__ == "__main__":
    main()
