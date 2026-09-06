import shutil
from pathlib import Path

from common.io import ensure_dir, read_json, write_json

STATE_FILE = "state.json"
ADAPTER_DIR = "adapter"


def latest_dir(run_dir):
    return Path(run_dir) / "latest"


def phase_dir(run_dir, phase):
    return Path(run_dir) / f"phase_{phase:04d}"


MASK_HEAD_FILE = "mask_head.pt"


def save_phase(run_dir, phase, model, state, mask_head=None, keep_phase_copy=True):
    latest = ensure_dir(latest_dir(run_dir))
    adapter = latest / ADAPTER_DIR
    if adapter.exists():
        shutil.rmtree(adapter)
    model.save_pretrained(str(adapter))
    write_json(latest / STATE_FILE, state)
    if mask_head is not None:
        import torch

        torch.save(mask_head.state_dict(), latest / MASK_HEAD_FILE)

    if keep_phase_copy:
        pdir = ensure_dir(phase_dir(run_dir, phase))
        if (pdir / ADAPTER_DIR).exists():
            shutil.rmtree(pdir / ADAPTER_DIR)
        shutil.copytree(adapter, pdir / ADAPTER_DIR)
        write_json(pdir / STATE_FILE, state)
        if mask_head is not None:
            shutil.copyfile(latest / MASK_HEAD_FILE, pdir / MASK_HEAD_FILE)
    return latest / ADAPTER_DIR


def mark_best(run_dir, phase):
    best = ensure_dir(Path(run_dir) / "best")
    src = phase_dir(run_dir, phase)
    for name in (ADAPTER_DIR, STATE_FILE, MASK_HEAD_FILE):
        s = src / name
        if not s.exists():
            continue
        d = best / name
        if d.exists():
            shutil.rmtree(d) if d.is_dir() else d.unlink()
        shutil.copytree(s, d) if s.is_dir() else shutil.copyfile(s, d)
    write_json(best / "best.json", {"phase": phase})
    return best


def resume_mask_head(run_dir, mask_head):
    path = latest_dir(run_dir) / MASK_HEAD_FILE
    if mask_head is None or not path.exists():
        return mask_head
    import torch

    mask_head.load_state_dict(torch.load(path, map_location="cpu"))
    return mask_head


def load_state(run_dir):
    path = latest_dir(run_dir) / STATE_FILE
    if not path.exists():
        return None
    return read_json(path)


def resume_adapter(run_dir):
    path = latest_dir(run_dir) / ADAPTER_DIR
    return str(path) if path.exists() else None
