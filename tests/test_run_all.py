import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dry_run(config):
    return subprocess.run([sys.executable, "scripts/run_all.py", "--config", str(config),
                           "--stages", "data", "--dry-run"], cwd=ROOT, capture_output=True, text=True)


def test_synthetic_source_generates_synthetic_data():
    done = dry_run("configs/base.yaml")
    assert done.returncode == 0
    assert "generate_synthetic.py" in done.stdout
    assert "--unmarked" not in done.stdout


def test_unmarked_flag_is_forwarded():
    done = dry_run("configs/kaggle_unmarked.yaml")
    assert done.returncode == 0
    assert "--unmarked" in done.stdout


def test_hotpotqa_source_loads_hotpotqa():
    done = dry_run("configs/hotpotqa.yaml")
    assert done.returncode == 0
    assert "load_hotpotqa.py" in done.stdout
    assert "generate_synthetic" not in done.stdout


def test_unknown_source_is_refused(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("run_root: out\nseed: 0\ndata:\n  source: nowhere\n  dir: x\n", encoding="utf-8")
    done = dry_run(config)
    assert done.returncode != 0
    assert "no loader" in done.stderr
