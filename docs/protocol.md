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
- `scoring` — the frozen backbone, with a per-span keep probability read off the logits.
  The only configuration that clears the positional control. Used for all reported results.
- `model` — the frozen backbone asked to generate a ranked index list. Fails at every scale
  tested; kept only to reproduce that failure.
- `heuristic` — a cue/identifier/number scorer. CPU-only debugging. Eight of its fourteen
  cue phrases appear verbatim in `data/banks.py`, so its lift measures that overlap rather
  than any judgment. Never report its numbers.

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

All rows below use the same 6 eval trajectories, 12 compaction events and 36 fact spans, so
they are directly comparable. The positional control ("keep the first N spans") is recomputed
per run and shown beside each lift; a compactor must beat *its own* control, not a fixed 1.0.

| Compactor | Elicitation | Fact keep | Filler keep | Lift | Control | Verdict |
|---|---|---|---|---|---|---|
| heuristic (debug) | rule-based | 1.000 | 0.226 | 4.43x | 1.68x | cheats, see below |
| SmolLM2-360M | logit scoring | 0.528 | 0.243 | **2.17x** | 1.32x | **clears** |
| Qwen2.5-0.5B | index list | 0.139 | 0.260 | 0.53x | 1.68x | fails |
| Qwen2.5-0.5B | logit scoring | 0.111 | 0.262 | 0.42x | 1.68x | fails |
| Qwen2.5-1.5B | index list | 0.139 | 0.260 | 0.53x | 1.69x | fails |
| Qwen2.5-1.5B | logit scoring | 0.611 | 0.239 | **2.56x** | 1.68x | **clears** |
| Qwen2.5-1.5B | logit scoring, unmarked | 0.278 | 0.252 | 1.10x | 1.57x | fails |
| Qwen2.5-1.5B | logit scoring, HotpotQA | 0.400 | 0.248 | **1.61x** | 0.98x | **clears** |
| SmolLM2-360M | logit scoring, HotpotQA | 0.225 | 0.257 | 0.88x | 1.08x | fails |

**Elicitation is what matters, not scale.** Every index-list row fails at every size: both Qwen
models answer with a contiguous prefix `0, 1, 2, ...` on 100% of events regardless of the text
at those positions, which under span shuffling is exactly random selection. Every model we
tested that clears the control does so through logit scoring.

**Scale is not monotonic, but 360M is not enough on natural text.** On the marked synthetic
set SmolLM2-360M reaches 2.17x while Qwen2.5-0.5B manages 0.42x, so ordering by size does not
predict which model carries signal. But on HotpotQA — natural prose, no marker phrase —
SmolLM2-360M drops to 0.88x against a 1.08x control, while Qwen2.5-1.5B clears at 1.61x. The
360M result on synthetic was exploiting the same marker the 1.5B model exploited there, and
it does not transfer.

The rule that survives both observations: **measure the compactor on the data you will
actually use.** Neither size nor a score on another dataset predicts it.

**The scorer hand-check does not predict lift.** `compactor/inspect_scorer.py` reports
salient-versus-filler separation on twelve curated lines: +0.050 for SmolLM2-360M, +0.040 for
Qwen2.5-0.5B, +0.197 for Qwen2.5-1.5B. Those figures do not order the models the way real lift
does (2.17x, 0.42x, 2.56x). Treat the hand-check as a sanity pass — does the scorer fire on
identifiers rather than markdown — and never as a substitute for measuring lift.

**Dataset matters as much as model.** The same 1.5B scoring compactor moves from 2.56x on the
marked synthetic set to 1.10x on the unmarked variant to 1.61x on HotpotQA. See
[benchmarks.md](benchmarks.md) for why the unmarked set is a floor case rather than a neutral
test.

Three earlier measurements of ours were wrong, each inflated by a different bug:

| Reported | Cause |
|---|---|
| 3.99x | 58% of decisions were silent heuristic fallbacks |
| 1.68x | model answered with the first N spans; facts sit early in the trajectory |
| 1.69x | `sorted(kept)[:budget]` restored positional bias after the shuffle |

Every one erred favourably. Treat a positive lift as unproven until the positional control
fails to reproduce it.

## Consequences if no compactor clears the gate

Compaction-supervised consolidation cannot beat uniform replay when the compactor selects at
random, because the two methods then draw from the same distribution. Options, in order of
preference:

Both were needed, and both are now the default. If a future backbone fails the gate again:

1. **Use a larger compactor than the consolidation target.** The compaction pass is one-off
   and offline, so the compactor need not be the model being consolidated into. The event
   log is plain text, so the two stages simply run under different `model.base` overrides.
2. **Never ask a small model for a ranked index list.** Read a per-span keep probability off
   the logits instead. `compactor/inspect_scorer.py` checks salient-versus-filler separation
   before any run.
3. **Report the negative if it comes to that.** "The compaction decision is free but only
   carries signal above a model-scale threshold, and only when read rather than generated"
   is a finding, and this infrastructure measures that threshold.

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
