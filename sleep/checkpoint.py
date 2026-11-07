import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from common.io import ensure_dir, read_json, write_json

STATE_FILE = "state.json"
ADAPTER_DIR = "adapter"
PHASE_STAMP = "phase.json"


def latest_dir(run_dir):
    return Path(run_dir) / "latest"


def phase_dir(run_dir, phase):
    return Path(run_dir) / f"phase_{phase:04d}"


MASK_HEAD_FILE = "mask_head.pt"


class ResumeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Resume:
    state: dict
    adapter: str
    mask_head: str


def save_phase(run_dir, phase, model, state, mask_head=None, keep_phase_copy=True):
    state = {**state, "mask_head": mask_head is not None}
    latest = ensure_dir(latest_dir(run_dir))
    adapter = latest / ADAPTER_DIR
    staged = latest / (ADAPTER_DIR + ".new")
    if staged.exists():
        shutil.rmtree(staged)
    model.save_pretrained(str(staged))
    write_json(staged / PHASE_STAMP, {"phase": phase})
    if adapter.exists():
        shutil.rmtree(adapter)
    os.replace(staged, adapter)

    head_path = latest / MASK_HEAD_FILE
    if mask_head is not None:
        import torch

        torch.save(mask_head.state_dict(), str(head_path) + ".tmp")
        os.replace(str(head_path) + ".tmp", head_path)
    elif head_path.exists():
        head_path.unlink()
    write_json(latest / STATE_FILE, state)

    if keep_phase_copy:
        pdir = ensure_dir(phase_dir(run_dir, phase))
        if (pdir / ADAPTER_DIR).exists():
            shutil.rmtree(pdir / ADAPTER_DIR)
        shutil.copytree(adapter, pdir / ADAPTER_DIR)
        write_json(pdir / STATE_FILE, state)
        if mask_head is not None:
            shutil.copyfile(head_path, pdir / MASK_HEAD_FILE)
    return adapter


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


def load_mask_head(path, mask_head):
    import torch

    mask_head.load_state_dict(torch.load(path, map_location="cpu"))
    return mask_head


def resume_run(run_dir, use_mask):
    latest = latest_dir(run_dir)
    adapter = latest / ADAPTER_DIR
    state_path = latest / STATE_FILE
    if not state_path.exists():
        if adapter.exists():
            raise ResumeError(f"{latest} holds an adapter but no state.json, so the event "
                              "cursor and replay buffer are lost")
        return None
    state = read_json(state_path)
    if not adapter.exists():
        raise ResumeError(f"{latest} holds state for phase {state.get('phase')} but no "
                          "adapter, which means a save was interrupted")
    stamp = adapter / PHASE_STAMP
    if stamp.exists() and read_json(stamp)["phase"] != state["phase"]:
        raise ResumeError(f"the adapter is at phase {read_json(stamp)['phase']} but state.json "
                          f"is at phase {state['phase']}, which means a save was interrupted")
    saved_mask = state.get("mask_head")
    if saved_mask is not None and saved_mask != use_mask:
        raise ResumeError(f"this run was saved with mask_head={saved_mask} but is being "
                          f"resumed with mask_head={use_mask}")
    head = latest / MASK_HEAD_FILE
    if use_mask and not head.exists():
        raise ResumeError(f"mask_head is on but {head} is missing")
    return Resume(state, str(adapter), str(head) if use_mask else None)
