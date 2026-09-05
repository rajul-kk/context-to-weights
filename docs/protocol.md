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

**GPU sweep, 48 eval trajectories, 288-386 fact spans per row.** Logit-scoring backend.
Two margins are reported: against chance (the overall keep rate) and against the positional
control. The verdict takes the weaker of the two.

| Compactor | Data | Fact keep | Lift | Control | vs chance | vs control | Verdict |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | HotpotQA | 0.337 | 1.34x | 0.45x | **+3.75σ** | +10.11σ | **clears** |
| Qwen2.5-0.5B | HotpotQA | 0.360 | 1.44x | 0.45x | **+4.77σ** | +11.13σ | **clears** |
| Qwen2.5-1.5B | synthetic | 0.646 | 2.72x | 1.66x | +15.17σ | +9.82σ | clears (marker) |
| Qwen2.5-0.5B | synthetic | 0.097 | 0.37x | 1.66x | −6.18σ | −13.20σ | below |
| Qwen2.5-1.5B | unmarked | 0.344 | 1.37x | 1.71x | +3.47σ | −3.32σ | below control |
| Qwen2.5-0.5B | unmarked | 0.014 | 0.05x | 1.71x | −9.41σ | −17.21σ | below |

**The compaction signal is real on natural text, at both model sizes.** On HotpotQA both
Qwen models keep fact spans well above chance — 0.5B slightly ahead of 1.5B — with no
marker phrase and no shared vocabulary to exploit. This is the result the compaction arm
rests on.

**Quote the margin over chance, not over the control, on HotpotQA.** Its positional control
is 0.45x, *below* chance, because `compactor/runner.py` prepends carry spans (kept spans and
the summary from the previous event, all tagged filler) to each span list. "Keep the first N"
therefore picks disproportionately from carry content and makes an undemanding baseline. The
+10σ figures are inflated; +3.75σ and +4.77σ are the honest numbers. `span_report.py` now
prints both and warns whenever the control falls below chance.

**Model and dataset interact, and neither predicts the other.** Qwen2.5-0.5B is the *best*
compactor on HotpotQA (+4.77σ) and the *worst* on synthetic (−6.18σ). Qwen2.5-1.5B is
enormous on marked synthetic (+15.17σ) because the marker phrase is trivially detectable,
modest on HotpotQA (+3.75σ), and beaten by position on unmarked synthetic. There is no
ordering of models that holds across datasets.

**Unmarked synthetic remains the floor case.** Qwen2.5-1.5B is +3.47σ over chance there —
it does find something — but position predicts better than it does, so it loses to the
control at −3.32σ. Facts and filler drawn from one template bank in one register is harder
than real prose, exactly as documented in [benchmarks.md](benchmarks.md).

Earlier small-n measurements on this row were unreliable in both directions. SmolLM2-360M
read as 2.17x (+2.85σ) on CPU at n=36 and 1.86x (+0.50σ) on GPU at n=24; Qwen2.5-0.5B read
as failing on HotpotQA at n=60 and clears at +4.77σ at n=383. **Do not report a compactor
from fewer than ~200 fact spans.**

Three earlier measurements were wrong for reasons other than sample size, each inflated by a
different bug:

| Reported | Cause |
|---|---|
| 3.99x | 58% of decisions were silent heuristic fallbacks |
| 1.68x | model answered with the first N spans; facts sit early in the trajectory |
| 1.69x | `sorted(kept)[:budget]` restored positional bias after the shuffle |

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
