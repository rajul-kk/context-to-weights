import shutil
from pathlib import Path

from common.io import ensure_dir, read_json, write_json

STATE_FILE = "state.json"
ADAPTER_DIR = "adapter"


def latest_dir(run_dir):
    return Path(run_dir) / "latest"


def phase_dir(run_dir, phase):
    return Path(run_dir) / f"phase_{phase:04d}"


def save_phase(run_dir, phase, model, state, keep_phase_copy=True):
    latest = ensure_dir(latest_dir(run_dir))
    adapter = latest / ADAPTER_DIR
    if adapter.exists():
        shutil.rmtree(adapter)
    model.save_pretrained(str(adapter))
    write_json(latest / STATE_FILE, state)

    if keep_phase_copy:
        pdir = ensure_dir(phase_dir(run_dir, phase))
        if (pdir / ADAPTER_DIR).exists():
            shutil.rmtree(pdir / ADAPTER_DIR)
        shutil.copytree(adapter, pdir / ADAPTER_DIR)
        write_json(pdir / STATE_FILE, state)
    return latest / ADAPTER_DIR


def load_state(run_dir):
    path = latest_dir(run_dir) / STATE_FILE
    if not path.exists():
        return None
    return read_json(path)


def resume_adapter(run_dir):
    path = latest_dir(run_dir) / ADAPTER_DIR
    return str(path) if path.exists() else None
