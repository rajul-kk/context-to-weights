import os
import re
import shlex
import subprocess
import sys
import time

QUIET_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "TQDM_DISABLE": "1",
    "PYTHONWARNINGS": "ignore",
    "PYTHONIOENCODING": "utf-8",
}

_PROGRESS = re.compile(r"(\d+%\|)|(\bit/s\])|(\bs/it\])|(Materializing param=)")


def is_progress(line):
    return bool(_PROGRESS.search(line))


def run(cmd, check=True, quiet=False, show_progress=False):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(c) for c in cmd]
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, "-u"] + cmd[1:]
    env = dict(os.environ)
    env.update(QUIET_ENV)

    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env, errors="replace")
    kept = []
    skipped = 0
    for line in proc.stdout:
        if not show_progress and is_progress(line):
            skipped += 1
            continue
        kept.append(line)
        if not quiet:
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    secs = time.time() - t0

    note = f" ({skipped} progress lines hidden)" if skipped else ""
    if proc.returncode != 0:
        print(f"\nFAILED (exit {proc.returncode}) after {secs:.0f}s{note}", flush=True)
        if quiet:
            print("".join(kept[-40:]), flush=True)
        if check:
            raise RuntimeError(f"command failed: {' '.join(cmd)}")
    else:
        print(f"[ok {secs:.0f}s]{note}", flush=True)
    return "".join(kept)
