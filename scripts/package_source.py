import argparse
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_DIRS = ["common", "data", "compactor", "sleep", "skills", "kl_gate", "distill",
                "baselines", "eval", "configs", "scripts", "notebooks", "docs", "paper"]
INCLUDE_FILES = ["requirements.txt", "README.md"]
SKIP_SUFFIX = {".pyc", ".pyo", ".zip", ".safetensors", ".pt", ".png"}
SKIP_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", "artifacts"}


def collect():
    files = []
    for name in INCLUDE_FILES:
        p = ROOT / name
        if p.exists():
            files.append(p)
    for d in INCLUDE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in SKIP_DIRS for part in p.parts):
                continue
            if p.suffix in SKIP_SUFFIX:
                continue
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(
        description="Zip the source tree for upload as a Kaggle Dataset.")
    ap.add_argument("--out", default="artifacts/myrios_src.zip")
    args = ap.parse_args()

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    files = collect()

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())

    size = out.stat().st_size / 1e6
    print(f"{len(files)} files -> {out} ({size:.1f} MB)")
    print()
    print("Upload it as a Kaggle Dataset:")
    print("  1. kaggle.com/datasets -> New Dataset -> upload this zip")
    print("  2. name it 'myrios-src' (or anything; note the slug)")
    print("  3. in the notebook, Add Input -> your dataset")
    print("  4. set SRC_DATASET in the boot cell to the slug you chose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
