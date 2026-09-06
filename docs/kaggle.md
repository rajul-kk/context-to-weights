# Running on Kaggle

Single T4 or P100, 9–12 hour sessions, ~30 GPU-hours per week. Nothing here needs more than
one GPU and every stage is resumable.

## Session layout

Data generation is CPU-only. Run it locally, commit the JSONL, and never spend GPU minutes
on it.

```bash
python data/generate_synthetic.py --n-train 48 --n-eval 16 --n-turns 120
```

## Before you run anything

**Session options -> Accelerator -> GPU T4 x2.** Kaggle installs a CPU-only build of torch
when no accelerator is attached, so `torch.cuda.is_available()` returns False and every stage
runs at CPU speed. The boot cell prints a banner if this happens. Changing the accelerator
restarts the session with a CUDA image.

**Session options -> Internet -> On**, unless you are booting from a Kaggle Dataset.

Kaggle currently ships `transformers 5.0.0`, `peft 0.19.1`, `accelerate 1.13.0`. Everything
here was developed against transformers 4.57 locally. The calls we make are the stable ones —
`AutoModelForCausalLM.from_pretrained(name, dtype=...)` (already the 5.x spelling, not the
removed `torch_dtype`), `apply_chat_template`, `generate`, and a plain forward for logits — so
no breakage is expected, but the first GPU session is also the first run against 5.x. If
something fails on import or generation, that is where to look first.

`scripts/preflight.py` prints the installed versions, so they are recorded in every run's
output. It also attaches a throwaway LoRA to a two-layer stub, which catches peft/backend
incompatibilities in seconds rather than after the compaction pass.

**torchao.** Kaggle ships torchao 0.10.0; peft requires above 0.16 and *raises* rather than
skipping its torchao dispatcher, so `get_peft_model` dies with an ImportError the moment a
sleep phase starts. Nothing here uses torchao, so the boot cell uninstalls it when the
version is too old. Preflight reports it either way.

## Notebooks

| Notebook | Purpose | Budget |
|---|---|---|
| `notebooks/a0_precondition.ipynb` | Salience lift per compactor. **Run first.** | 15 min quick / ~1.5 h full |
| `notebooks/a1_main.ipynb` | Compaction, all sleep runs, all eval arms, figures | ~4 h |
| `notebooks/b1_skills.ipynb` | KL gate, distillation, report | ~3 h (sweep is a second session) |
| `notebooks/c1_declare.ipynb` | Declaration reliability vs scale, generate vs read | 20 min quick / ~2 h full |

`a1_main.ipynb` has a `DATASET` switch in its first cell: `hotpotqa` (primary — natural
prose, no marker, positional control at chance) or `synthetic` (for the compaction-ratio
sweep and debugging).

`a0_precondition.ipynb` has a `QUICK` switch in its boot cell. `QUICK = True` is one model,
one backend, 4 eval trajectories, about fifteen minutes — enough to confirm the chain works.

`QUICK = False` is the real sweep: **7 runs at 48 eval trajectories**, ordered so the
load-bearing measurement comes first.

| | |
|---|---|
| models | Qwen2.5-1.5B, Qwen2.5-0.5B |
| datasets | HotpotQA (first), synthetic, unmarked |
| backends | logit scoring, plus one index-list confirmation |

Two things are deliberately dropped from the full sweep. **SmolLM2-360M**, measured at +0.50,
-3.57 and -0.76 sigma against its controls, carries no demonstrable signal and is not worth
more GPU time. **The index-list backend** produced 100% prefix answers at every scale tested,
so it is reduced to a single confirmation run rather than a full arm.

48 eval trajectories puts HotpotQA at roughly 200 fact spans instead of 60. That matters:
the 1.5B HotpotQA result is +2.79 sigma at n=60, and the whole compaction arm rests on it.

**Each notebook must use a config whose `run_root` matches the path the notebook saves and
restores.** The `skill_base` and `declare` configs use relative run roots, which resolve
under the cloned repo rather than `/kaggle/working` — so `b1` and `c1` point at
`configs/kaggle_skills.yaml` and `configs/kaggle_declare.yaml` instead. `b1` asserts the
match in its boot cell, because a mismatch means distillation silently cannot find the KL
scores written a cell earlier.

Run `python scripts/check_notebooks.py` after editing any notebook. It compiles every code
cell's Python, ignoring `!` and `%` lines and their continuations, and catches the broken
line continuations that are easy to introduce when generating notebook JSON.

Each notebook clones the repo, installs `peft`, `accelerate` and `datasets`, runs
`scripts/preflight.py` and restores the previous session's archive before doing any work.

Every subprocess goes through `run()` from `notebooks/_runner.py` rather than a `!` shell
magic. `!` swallows a non-zero exit code, so a failing script leaves the cell looking
finished with nothing useful printed, and the rest of the notebook runs against missing
files. `run()` streams output line by line, prints elapsed time, and raises on failure so
the notebook stops at the first real error.

It also suppresses progress bars. Hugging Face renders weight loading and dataset mapping as
carriage-return updates, and a subprocess pipe turns every one of those into its own line —
a single 360M model load emits several hundred. `run()` sets `HF_HUB_DISABLE_PROGRESS_BARS`
and friends in the child environment, filters any residual progress lines, and reports how
many it hid. Pass `show_progress=True` if you actually want them.

**Getting the code onto Kaggle.** Uploading only the `.ipynb` is not enough — the notebook
needs the repo. The boot cell now defaults to cloning
`https://github.com/rajul-kk/context-to-weights.git`, so an uploaded notebook works with no
edits as long as the Kaggle session has internet enabled (Settings -> Internet -> On).

If internet is off, or you would rather not clone, use a Dataset instead:

**Kaggle Dataset (no GitHub account needed).** Locally:

```bash
python scripts/package_source.py
```

That writes `artifacts/myrios_src.zip` — about 0.1 MB, source only, no artifacts or
adapters. Upload it at kaggle.com/datasets as a new dataset, then in the notebook use
*Add Input* to attach it. The boot cell scans `/kaggle/input/*`, finds either an extracted
tree or a zip, and unpacks it to `/kaggle/working/myrios`. Nothing to edit.

**Git.** Set `GIT_URL` at the top of the boot cell to your repo URL and leave the dataset
unattached.

If neither is present the boot cell asserts with the list of what it actually found under
`/kaggle/input/`, rather than failing silently three cells later.

`scripts/preflight.py` checks torch and GPU, the dependency versions, that the data splits
exist and are non-empty, that `compaction.backend` is `scoring`, that the run root is
writable and that there is disk headroom. It exits non-zero on failure, so a broken session
stops in the first cell rather than after two hours.

## Checkpoints

Every sleep phase writes:

```
<run_dir>/latest/adapter/          most recent LoRA weights
<run_dir>/latest/state.json        phase, cursor, reservoir contents
<run_dir>/phase_0007/              frozen copy of both
<run_dir>/metrics.jsonl            one line per phase
```

Only `latest/` is needed to resume. The per-phase copies exist so a run can be replayed or
a regression bisected without retraining.

## Persisting across sessions

`/kaggle/working` is wiped between sessions. After each session, save the run root as a
Kaggle Dataset and mount it read-only next time:

```python
import shutil
shutil.make_archive("/kaggle/working/runs", "zip", "/kaggle/working/artifacts/runs")
```

Then in the next session, before anything else:

```python
!mkdir -p /kaggle/working/artifacts && unzip -q /kaggle/input/<dataset>/runs.zip \
    -d /kaggle/working/artifacts/runs
```

A phase directory is a few megabytes — rank-16 LoRA on a 0.5B backbone is about 35 MB, and
only `latest/` plus `metrics.jsonl` are worth keeping if the dataset gets large.

## Suggested session split

| Session | Work | Budget |
|---|---|---|
| 0 | `scripts/check_compactor.py` — salience lift per model. Stop here if nothing clears 1.0 | ~1 h |
| 1 | compaction over both splits, span report, floor/ceiling contexts, reflections | ~2 h |
| 2 | sleep runs for ours + uniform | ~4 h |
| 3 | sleep runs for reflection + mask-head ablation | ~4 h |
| 4 | all eval arms, report, figures | ~2 h |
| 5–6 | ratio sweep, no-replay ablation | ~5 h |
| 7–8 | second model (Qwen2.5-1.5B), third (SmolLM2-360M) | ~6 h |
