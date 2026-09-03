import shlex
import subprocess
import sys
import time


def run(cmd, check=True, quiet=False):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    cmd = [str(c) for c in cmd]
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, "-u"] + cmd[1:]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if not quiet:
            sys.stdout.write(line)
            sys.stdout.flush()
    proc.wait()
    secs = time.time() - t0
    if proc.returncode != 0:
        tail = "".join(lines[-40:])
        print(f"\nFAILED (exit {proc.returncode}) after {secs:.0f}s", flush=True)
        if quiet:
            print(tail, flush=True)
        if check:
            raise RuntimeError(f"command failed: {' '.join(cmd)}")
    else:
        print(f"[ok {secs:.0f}s]", flush=True)
    return "".join(lines)
