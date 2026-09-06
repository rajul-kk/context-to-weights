# Project A results

Two questions, answered separately.

1. **Does the compactor's keep/drop decision carry a salience signal?** Yes.
2. **Does consolidating on that signal help retention?** No. Definitively negative under a corrected, well-controlled setup.

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

**Corrected run** (adapter consolidates the eval trajectories; `--resume` refuses a mismatched events file; all four methods trained fresh for 24 phases). Marked synthetic set, compaction signal **+15.53σ** (clustered permutation, p < 0.0001):

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

The training target is a bare LM objective on `retained note: <fact sentence>`. The adapter
learns to reproduce that string — validation CE on kept spans drops to 0.001 for the
mask-head variant — but reproducing a sentence is not the same as answering a question about
it, and the greedy decoder cannot retrieve the fact when prompted with a question form it
never trained on. Reflection, which trains on richer LLM-written summaries, fails the same
way. The gap is between memorisation and retrieval, and at 0.5B with a few hundred training
steps it does not close.

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
