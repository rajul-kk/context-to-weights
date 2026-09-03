# Running on Kaggle

Single T4 or P100, 9–12 hour sessions, ~30 GPU-hours per week. Nothing here needs more than
one GPU and every stage is resumable.

## Session layout

Data generation is CPU-only. Run it locally, commit the JSONL, and never spend GPU minutes
on it.

```bash
python data/generate_synthetic.py --n-train 48 --n-eval 16 --n-turns 120
```

## Notebooks

| Notebook | Purpose | Budget |
|---|---|---|
| `notebooks/a0_precondition.ipynb` | Salience lift per compactor. **Run first.** | 15 min quick / ~6 h full |
| `notebooks/a1_main.ipynb` | Compaction, all sleep runs, all eval arms, figures | ~4 h |
| `notebooks/b1_skills.ipynb` | KL gate, distillation, sweeps | ~5 h |

`a1_main.ipynb` has a `DATASET` switch in its first cell: `hotpotqa` (primary — natural
prose, no marker, positional control at chance) or `synthetic` (for the compaction-ratio
sweep and debugging).

`a0_precondition.ipynb` has `QUICK = True` in its boot cell. Leave it on for the first run:
one model, one backend, 4 eval trajectories, about fifteen minutes end to end. It exercises
the whole chain and prints the summary table, so you find out whether the plumbing works
before committing to the full sweep. Set `QUICK = False` for the real run — 3 datasets x 3
models x 2 backends is 18 compaction passes and takes most of a session.

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

**Before the first session** the code has to reach Kaggle. Uploading only the `.ipynb` is not
enough — the notebook needs the repo. Two ways, and the boot cell handles both:

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
