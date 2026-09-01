# Running on Kaggle

Single T4 or P100, 9–12 hour sessions, ~30 GPU-hours per week. Nothing here needs more than
one GPU and every stage is resumable.

## Session layout

Data generation is CPU-only. Run it locally, commit the JSONL, and never spend GPU minutes
on it.

```bash
python data/generate_synthetic.py --n-train 48 --n-eval 16 --n-turns 120
```

## Notebook cell

```python
!pip -q install peft accelerate
%cd /kaggle/working/compaction-sleep

!python scripts/run_all.py --config configs/kaggle.yaml --mask-head
```

To pick up a killed session, point the same run directory at `--resume`:

```bash
!python sleep/loop.py --config configs/kaggle.yaml --method compaction \
  --events artifacts/runs/cascading/train_events.jsonl \
  --val-events artifacts/runs/cascading/eval_events.jsonl \
  --run-dir /kaggle/working/artifacts/runs/sleep_compaction --resume
```

The loop reads `latest/state.json`, restores the adapter, the event cursor and the reservoir
buffer, and continues from the next phase.

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
