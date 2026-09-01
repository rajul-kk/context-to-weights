import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import load_config, read_jsonl
from eval.span_report import summarize

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--keep-frac", type=float, default=0.25)
    ap.add_argument("--split", default="eval")
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    root = Path(cfg["run_root"]) / "compactor_check"
    rows = []

    for model in args.models:
        slug = model.split("/")[-1]
        out = root / slug
        cmd = [sys.executable, "-u", "baselines/cascading.py", "--config", args.config,
               "--split", args.split, "--out", str(out), "--set",
               "compaction.backend=model", f"model.base={model}",
               f"compaction.keep_frac={args.keep_frac}"]
        print("$ " + " ".join(cmd[2:]))
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:])
            rows.append({"model": slug, "error": True})
            continue

        fallback = "unknown"
        for line in proc.stdout.splitlines():
            if line.startswith("compactor:"):
                fallback = line.split("(")[-1].split(")")[0]
        report = summarize(read_jsonl(out / f"{args.split}_events.jsonl"))
        rows.append({"model": slug, "fallback": fallback, **report})

    print("\n| compactor | fallback | fact keep | filler keep | salience lift | verdict |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r.get("error"):
            print(f"| {r['model']} | - | - | - | - | run failed |")
            continue
        lift = r["salience_lift"]
        verdict = "usable" if lift > 1.0 else "NO SIGNAL"
        print(f"| {r['model']} | {r['fallback']} | {r['fact_keep_rate']:.3f} | "
              f"{r['filler_keep_rate']:.3f} | {lift:.2f}x | {verdict} |")
    print("\nA compactor at or below 1.00x carries no supervision. Do not run the main "
          "comparison with it.")


if __name__ == "__main__":
    main()
