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


def eval_arm(config, out_dir, resume, label_args, lim, dry):
    summary = Path(out_dir).with_suffix(".summary.json")
    if resume and summary.exists() and not dry:
        print(f"skip {out_dir} (already evaluated)")
        return
    sh(["eval/skill_eval.py", "--config", config, "--out", out_dir] + label_args + lim, dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/skill_base.yaml")
    ap.add_argument("--stages", default="skills,score,distill,eval,report")
    ap.add_argument("--policies", default="kl_top,uniform,random")
    ap.add_argument("--granularity", default="span")
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run every eval arm even if its summary already exists")
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
    resume = not args.no_resume
    if "eval" in stages:
        label_of = {"kl_top": "ours", "uniform": "s2l-uniform", "random": "random-control"}
        eval_arm(args.config, report_dir / "prompt-full", resume,
                ["--doc-mode", "correct", "--label", "prompt-full"], lim, dry)
        eval_arm(args.config, report_dir / "no-skill", resume,
                ["--doc-mode", "none", "--label", "no-skill"], lim, dry)
        eval_arm(args.config, report_dir / "prompt-mismatched", resume,
                ["--doc-mode", "mismatched", "--label", "prompt-mismatched"], lim, dry)
        for policy, run_dir in runs.items():
            label = label_of.get(policy, policy)
            eval_arm(args.config, report_dir / label, resume,
                    ["--run-dir", run_dir, "--doc-mode", "none", "--label", label], lim, dry)
        eval_arm(args.config, report_dir / "ours-mismatched", resume,
                ["--run-dir", runs["kl_top"], "--doc-mode", "mismatched",
                 "--label", "ours-mismatched"], lim, dry)

    if "report" in stages:
        out = "docs/results_skills.md" if args.granularity == "span" \
            else f"docs/results_skills_{args.granularity}.md"
        sh(["eval/skill_report.py", "--report-dir", report_dir, "--run-root", root,
            "--out", out], dry)


if __name__ == "__main__":
    main()
