import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import load_config


def check(label, ok, detail=""):
    mark = "ok  " if ok else "FAIL"
    print(f"[{mark}] {label}{('  ' + detail) if detail else ''}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/kaggle.yaml")
    ap.add_argument("--require-gpu", action="store_true")
    args = ap.parse_args()

    results = []
    cfg = load_config(ROOT / args.config)

    try:
        import torch

        cuda = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if cuda else "cpu"
        total = (torch.cuda.get_device_properties(0).total_memory / 1e9) if cuda else 0.0
        results.append(check("torch", True, f"{torch.__version__}"))
        results.append(check("gpu", cuda or not args.require_gpu, f"{name} {total:.1f} GB"))
    except ImportError:
        results.append(check("torch", False, "not installed"))
        cuda = False

    for mod in ("transformers", "peft", "accelerate"):
        try:
            m = __import__(mod)
            results.append(check(mod, True, getattr(m, "__version__", "")))
        except ImportError:
            results.append(check(mod, False, "not installed"))

    try:
        import importlib.metadata as md

        tv = md.version("torchao")
        ok = tuple(int(x) for x in tv.split(".")[:2]) >= (0, 16)
        results.append(check("torchao", ok,
                             f"{tv}" + ("" if ok else " - peft raises on this; pip uninstall -y torchao")))
    except Exception:
        results.append(check("torchao", True, "absent, which is what peft wants"))

    try:
        import peft
        import torch.nn as nn
        from peft import LoraConfig, get_peft_model

        probe = nn.Sequential()
        probe.add_module("q_proj", nn.Linear(8, 8))
        get_peft_model(probe, LoraConfig(r=2, target_modules=["q_proj"]))
        results.append(check("lora attach", True, "get_peft_model works"))
    except Exception as exc:
        results.append(check("lora attach", False, f"{type(exc).__name__}: {exc}"))

    data_dir = ROOT / cfg["data"]["dir"]
    for split in ("train", "eval"):
        p = data_dir / cfg["data"][split]
        n = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
        results.append(check(f"data/{split}", p.exists() and n > 0, f"{n} trajectories"))

    backend = cfg["compaction"]["backend"]
    results.append(check("compaction.backend", backend == "scoring",
                         f"{backend} (scoring is the only configuration that clears the control)"))

    run_root = Path(cfg["run_root"])
    writable = True
    try:
        run_root.mkdir(parents=True, exist_ok=True)
        probe = run_root / ".preflight"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        writable = False
        print(f"       {exc}")
    results.append(check("run_root writable", writable, str(run_root)))

    free = shutil.disk_usage(run_root if run_root.exists() else ROOT).free / 1e9
    results.append(check("disk free", free > 5.0, f"{free:.1f} GB"))

    print(f"\nbackbone      {cfg['model']['base']}")
    print(f"lora rank     {cfg['sleep']['lora_r']}")
    print(f"sleep steps   {cfg['sleep']['steps']}")
    print(f"keep_frac     {cfg['compaction']['keep_frac']}")

    if all(results):
        print("\npreflight passed")
        return 0
    print("\npreflight FAILED - fix the above before spending GPU time")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
