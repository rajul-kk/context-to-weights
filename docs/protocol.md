# Experiment protocol

## Trajectories

`data/generate_synthetic.py` builds multi-turn software-development conversations on CPU.
Each trajectory plants `--n-facts` load-bearing facts inside the first `--early-window`
exchange units — datastore choices, header names, rate limits, queue names, flag names,
version pins — surrounded by filler exchanges that are topically plausible but carry no
retrievable commitment. Every fact ships with a held-out probe question and gold answer.

Facts land early on purpose: by the time the context budget trips, they are the first
content a cascading compactor is asked to throw away.

Defaults: 48 train / 16 eval trajectories, 120 turns, 6 facts each.

## Compaction

`compactor/runner.py` walks the trajectory turn by turn. When the running context exceeds
`compaction.context_budget`, everything but the last `keep_recent_turns` turns is segmented
into spans (sentence-level by default) and handed to the compactor, which keeps at most
`ceil(keep_frac * n_spans)` of them verbatim and writes a summary of the rest. The new
context is the kept spans plus the summary plus the recent turns.

Two backends:

- `model` — the frozen backbone under the fixed prompt in `compactor/prompts.py`, which
  replies `KEEP: <indices>` / `SUMMARY: <text>`. Used for all reported results.
- `heuristic` — a cue/identifier/number scorer. CPU-only debugging. It shares vocabulary
  with the generator's fact templates, so it separates facts from filler almost perfectly.
  Treat its numbers as a plumbing check, never as a result.

Every event is logged as a `CompactionEvent` holding `(span_text, kept)` for every span.
That log is the free supervision.

## Sleep phases

Every `sleep.every_k_events` compaction events, `sleep/loop.py` runs one phase:

1. Build examples from the window under the chosen method.
2. Draw up to `replay.capacity / 2` examples from the reservoir buffer.
3. Train the LoRA adapter for `sleep.steps` steps of cross-entropy, loss on the target
   span only.
4. Add the window's examples to the reservoir.
5. Checkpoint adapter, cursor and reservoir state.

The adapter is never reset — consolidation is cumulative across phases, and the loop
resumes mid-run from `latest/state.json`.

## Methods

| Tag | Training examples |
|---|---|
| ours | spans the compactor kept |
| a, uniform | the same *number* of spans, drawn uniformly at random from the window |
| b, reflection | one model-written reflection per event |
| c, cascading | none — context only |
| d, full | none — full trajectory in context |
| floor | none — no context at all |

Baseline (a) is the load-bearing control: it holds the consolidation machinery, the example
budget and the compute fixed, and varies only whether the compactor chose the spans.

## Evaluation

Every arm is scored by `eval/retention.py` on the same eval trajectories. Arms (c), (a),
(b) and ours all see the *post-compaction* context, so the only difference between them is
the adapter. (d) sees everything, floor sees nothing.

Reported per arm:

- `retention_accuracy` — normalised containment match of the gold answer over all probes.
- `evicted_accuracy` — the headline number. Restricted to probes whose gold answer is
  **not** present in the retained context, i.e. the facts compaction actually removed. If
  a method scores here, the knowledge is in the weights.
- `token_ce_median` / `token_ce_mean` — per-token cross-entropy on the gold answer. Read
  the **median**; mean CE is logged only to show it moving the wrong way.
- `mean_prompt_tokens` — inference-time token cost.
- GPU-seconds per sleep phase, from `metrics.jsonl`.

## Precondition: the compactor must carry signal

Everything in this project rests on one assumption — that the compactor's keep/drop decision
correlates with what a later question will need. That assumption is checkable directly, and
it is not free.

Run `eval/span_report.py` and read **salience lift**: the fact-span keep rate divided by the
filler-span keep rate. Above 1.0 the compactor's decision carries supervision. At or below
1.0 it carries none, and compaction-supervised consolidation has nothing to learn that
uniform replay would not learn too. `span_report.py` warns when this happens.

Measured so far:

| Compactor | Fact keep | Filler keep | Salience lift |
|---|---|---|---|
| heuristic (debug) | 1.000 | 0.224 | 4.47x |
| SmolLM2-360M-Instruct | 0.000 | 0.136 | 0.00x |

The 360M model is **worse than random** — it keeps chit-chat and drops every planted fact.
The heuristic's 4.47x is not evidence either, since it scores on the same cue words the fact
templates use.

`scripts/check_compactor.py` runs this check across models and prints the table above:

```bash
python scripts/check_compactor.py --config configs/kaggle.yaml
```

So the first GPU session must establish salience lift for Qwen2.5-0.5B-Instruct and
Qwen2.5-1.5B-Instruct before any consolidation runs. If 0.5B does not clear 1.0, the
compactor and the consolidation target have to be decoupled: use 1.5B as the compactor and
consolidate into 0.5B. That is still within scope — the brief permits a separate small model
as the compactor — but it changes the story and must be stated in the paper.

## Sanity gate before trusting a comparison

Check `n_evicted` in the cascading arm's summary. If it is 0, the compactor kept every
probed fact and `evicted_accuracy` is undefined — every method will look identical because
nothing was ever consolidated away. `eval/retention.py` and `eval/report.py` both warn when
this happens.

This is the normal outcome of the CPU debug configuration: short trajectories plus the
heuristic compactor plus `keep_frac 0.25` evicts nothing. Lower `keep_frac`, lengthen
trajectories, or switch to the model compactor before reading any result off the table.

Second check: the full-context arm (d) must score at or above cascading (c). If it does not,
the backbone is too small to use a long context and (d) is not functioning as a ceiling. The
CPU debug run shows exactly this — SmolLM2-360M reaches 0.67 on a 990-token full context
against 0.94 on a 481-token compacted one, because the compacted context puts the answer
close to the question. Read (d) as a ceiling only once this ordering holds.

## Ablations

| Ablation | How |
|---|---|
| mask-prediction head | `sleep/loop.py --mask-head` |
| no replay buffer | `sleep/loop.py --no-replay` |
| compaction-ratio sweep | `scripts/sweep_ratio.py --fracs 0.1,0.25,0.5` |

The sweep is the interesting one: as `keep_frac` falls the compactor evicts more facts, so
(c) degrades. The claim to test is that compaction-supervised consolidation degrades more
slowly than uniform replay does.

## Budget

| Stage | Cost |
|---|---|
| data generation | CPU, seconds |
| compaction over 64 trajectories | ~0.3 GPU-h |
| one sleep run (200 steps x ~28 phases) | ~1.5 GPU-h |
| one eval arm | ~0.2 GPU-h |
| main table, one model | ~6 GPU-h |
| ratio sweep + ablations | ~6 GPU-h |
| second and third model | ~6 GPU-h |
