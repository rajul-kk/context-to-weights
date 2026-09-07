# Project A results

Two questions, answered separately.

1. **Does the compactor's keep/drop decision carry a salience signal?** Yes.
2. **Does consolidating on that signal help retention?** No. Definitively negative under a
   corrected, well-controlled setup, by generation *and* by CE ranking. Closed.

## The compaction signal is real

`Qwen2.5-1.5B` with the logit-scoring backend, salience lift against both a chance baseline
and a positional control ("keep the first N spans"), 48 eval trajectories:

| Dataset | Fact keep | Lift | vs chance | vs control | Clustered permutation |
|---|---|---|---|---|---|
| HotpotQA | 0.337 | 1.34x | +3.75σ | +10.1σ | +6.07σ, p < 0.0001 |
| synthetic (marked) | 0.646 | 2.72x | +15.2σ | +9.8σ | — |
| synthetic (unmarked) | 0.344 | 1.37x | +3.47σ | −3.3σ | — |

`Qwen2.5-0.5B` is the better HotpotQA compactor (+4.77σ over chance) and fails on synthetic.
Neither model size nor a score on one dataset predicts the other. The positional control on
HotpotQA sits below chance because carry spans lead each span list, so the margin over chance
is the honest figure — and it is comfortably significant under a clustered permutation test
that respects the shared keep budget within each event.

**Every model asked to *generate* a ranked span list answers `0, 1, 2, ...` regardless of
content, at every scale.** The signal is only accessible by reading a per-span keep
probability off the logits. See [protocol.md](protocol.md).

## Consolidation does not help

The first two runs (HotpotQA, marked synthetic) consolidated the *train* trajectories and
tested on *eval* trajectory facts the adapter never saw — cross-trajectory transfer, ~0 by
construction. The notebook was fixed to consolidate `eval_events.jsonl` (the conversations
the adapter is then tested on), and `sleep/loop.py` now refuses to resume when the events
file changed.

**Corrected run** (adapter consolidates the eval trajectories; `--resume` refuses a mismatched
events file; all four methods trained fresh for 24 phases). Marked synthetic set, compactor
`Qwen2.5-1.5B-Instruct`, consolidation and eval backbone `Qwen2.5-0.5B-Instruct`, compaction
signal **+15.53σ** (clustered permutation, p < 0.0001):

| method | all-probe acc | evicted acc (n=120) | evicted median CE |
|---|---|---|---|
| (c) cascading, **no adapter** | **0.455** | 0.000 | 6.29 |
| ours: compaction-supervised | 0.420 | 0.008 (1 of 120) | 5.52 |
| ours + mask head | 0.358 | 0.000 | 4.83 |
| (a) uniform replay | 0.326 | 0.000 | 2.24 |
| (b) reflection | 0.323 | 0.000 | 3.73 |
| (d) full context (ceiling) | 0.795 | — (nothing evicted) | — |
| floor (no context) | 0.000 | 0.000 | 4.45 |

![retention](../artifacts/runs_synth/report/figures/headline_retention.png)

Three findings, all negative for the headline claim:

1. **Consolidation recovers no evicted facts.** ours got 1 probe of 120; everything else got
   0. This is with a compaction signal at +15.5σ — the strongest this method will produce.
2. **ours does not beat uniform.** 0.008 vs 0.000 is one question, exactly one standard
   error. If anything the evicted-CE column favours uniform (2.24 vs ours' 5.52), meaning
   uniform's adapter assigns the right answers lower loss even though neither can generate
   them.
3. **Every adapter degrades general QA.** No consolidated adapter beats the no-adapter
   baseline on all-probe accuracy (0.455), on either dataset.

### Why it fails

The failure is entirely in the consolidation *mechanism*, not the signal. Four pieces of
evidence.

**1. The compactor did its job.** Fact-span keep rate 0.62 against a 0.24 filler rate,
+15.53σ under a clustered permutation test. The spans handed to the sleep loop were the
load-bearing ones. Garbage-in is not the story.

**2. The adapter memorised the target perfectly and it did not help.** Validation
cross-entropy on held-out kept-span text fell to **0.0003** by phase 24 for compaction,
uniform and the mask-head variant alike. The adapter can reproduce
`retained note: We settled on SQLite 3.45 as the primary datastore for rate-limiter.`
essentially losslessly. Evicted-fact QA accuracy is still 0.

**3. The knowledge is not in the weights at all.** Evicted-probe median CE falls under every
adapter — cascading 6.29, ours 5.52, reflection 3.73, uniform **2.24** — which looks like
partial encoding that greedy decoding cannot reach. It is not. A CE-ranking eval
(`eval/scoring_retention.py`) scores the gold answer against distractors drawn from the same
fact bank and counts a hit when gold has the lowest loss:

| method | MC acc (all) | MC acc (evicted) | margin(ev) | σ vs chance |
|---|---|---|---|---|
| cascading (no adapter) | **0.642** | 0.233 | −0.88 | −0.28 |
| uniform | 0.601 | 0.242 | −1.72 | −0.06 |
| reflection | 0.590 | 0.275 | −2.18 | +0.76 |
| ours | 0.573 | 0.183 | −1.88 | −1.72 |
| ours + mask | 0.566 | 0.183 | −1.72 | −1.72 |

Chance is 0.240. **No arm ranks evicted answers above chance**, and ours sits below it. Mean
margin is negative everywhere — the best distractor beats gold — and grows *more* negative
after training (−0.88 → −1.88).

So the CE drop is an artifact of how the answers are distributed. Every candidate value comes
from the same template bank, and the adapter trained on span text containing many of them, so
it lowered loss on the gold answer and its distractors alike. It learned the answer
vocabulary without the binding. **A CE reduction on the gold answer is not evidence of
knowledge acquisition unless it is checked against distractors from the same distribution** —
and an earlier version of this document drew exactly that unsupported inference from the
table above.

Every adapter also ranks *retained* answers worse than no adapter (0.642 → 0.573), so the
sleep pass is not trading retained accuracy for evicted accuracy. It is degrading the
backbone.

**4. Training format ≠ eval format.** The adapter is trained to continue a session-id cue
into a declarative sentence. At eval it is given a question and must produce a short answer.
Nothing in training connected the question form to the answer. This is a reversal-curse-shaped
problem: fine-tuning on "A is B" does not reliably yield "what is B? → A" in a new phrasing.
Reflection trains on LLM-written prose summaries instead of raw spans and fails the same way,
worse — its 0.5B summariser produces targets the adapter cannot even memorise (val CE stays
near 4).

**What would plausibly change the outcome**, none of it in scope here: a consolidation target
that is itself QA-shaped (synthetic `Q → A` pairs built from the kept spans, so training and
eval formats match — but that is close to "reflection with structure"), or a larger
consolidation target where the know-but-cannot-say gap is smaller. The scoring eval was the
third candidate and has now been run: it is null, so there is no hidden signal for a weaker
multiple-choice claim to recover. **Project A is closed.**

The clean statement: **the compaction decision is a strong salience signal, and LoRA SFT on
the selected span text — or on random spans, or on reflections — does not turn that signal
into knowledge at this scale, whether measured by generation or by ranking.** The bottleneck
is knowledge injection via fine-tuning, a known-hard problem, not the compaction signal.

## Consequence for the writeup

The headline claim — compaction-supervised consolidation beats uniform replay — is
**refuted** by the corrected run. The consolidation half of Project A is a negative result. The contributions that survive:

- **The elicitation finding.** A salience decision must be *read* from the model, not
  *generated* by it, below the scales where structured instruction-following is reliable.
  Supported independently: the signal is only accessible at +6σ via logit scoring.
- **The measurement methodology.** Salience lift with a positional control and a clustered
  permutation test, and the record of construction artifacts it caught (seven so far, every
  one inflating a result in the favourable direction).

These are the spine of the combined paper's §6 and §7, and neither depends on consolidation
working. See [paper/draft_combined.md](../paper/draft_combined.md).
