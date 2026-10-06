import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "artifacts", "__pycache__", ".venv", "venv", ".pytest_cache", ".ruff_cache"}


def repo_files(pattern):
    done = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", pattern],
                          cwd=ROOT, capture_output=True, text=True)
    if done.returncode == 0 and done.stdout.strip():
        return sorted(ROOT / line for line in done.stdout.splitlines() if (ROOT / line).exists())
    return sorted(p for p in ROOT.rglob(pattern) if not SKIP & set(p.relative_to(ROOT).parts))
