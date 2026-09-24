# Project A results

Two questions, answered separately.

1. **Does the compactor's keep/drop decision carry a salience signal?** Yes, +15.85σ.
2. **Does consolidating on that signal beat consolidating on random spans?** **No — it is
   strictly worse.** Uniform replay recovers 12.3% of evicted facts; compaction-supervised
   consolidation recovers none.

The second answer only became measurable after a benchmark defect was fixed. Earlier runs put
every arm at 0.000 and were reported first as a definitive negative and then as unresolved;
both readings were premature. This document describes the corrected run.

## The benchmark defect, found and fixed

The synthetic generator drew each probe question from a template filled with a `{service}`
from a pool of 8, and only 2 of 10 templates mentioned the project. Across 48 eval
trajectories that produced **58 distinct question strings for 288 probes**, 53 of which had
more than one gold answer — a mean of 2.7 each.

| | distinct questions | ambiguous probes | closed-book ceiling |
|---|---|---|---|
| before | 58 | 283/288 (98.3%) | **50.0%** |
| after | 288 | 0/288 (0.0%) | 100.0% |

For an evicted probe the retained context does not contain the answer, so the model saw
"Which header carries the auth token?" with nothing to say which of four projects was meant.
A model with perfect recall could not have exceeded 50%.

Fixed in `data/banks.py` and `data/generate_synthetic.py`: every question is scoped by a
project name unique to its trajectory. This is what unpinned the evicted column from zero.

## The compaction signal is real

`Qwen2.5-1.5B` with the logit-scoring backend, 48 eval trajectories, 97 compaction events,
6244 spans, mean compaction ratio 0.630. Zero heuristic fallbacks, zero empty keeps, zero
prefix answers.

| | rate |
|---|---|
| fact-span keep rate | 0.663 (n=288) |
| filler-span keep rate | 0.236 (n=5956) |
| salience lift | **2.81x** |
| positional control ("keep the first N") | 1.89x |
| margin over chance | **+15.85σ** |
| margin over control | **+8.40σ** |
| verdict | clears control |

Per-fact keep rates are uneven and worth noting: `db_engine`, `owner` and `version_pin` are
kept 100% of the time, while `auth_header` is kept **0%** and `config_flag` 3.4%. The
compactor is not uniformly good; it has blind spots that a single lift number hides.

**Every model asked to *generate* a ranked span list answers `0, 1, 2, ...` regardless of
content, at every scale.** The signal is only accessible by reading a per-span keep
probability off the logits. See [protocol.md](protocol.md).

## Consolidation works, and the compaction gate makes it worse

Compactor `Qwen2.5-1.5B-Instruct`, consolidation and eval backbone `Qwen2.5-0.5B-Instruct`,
25 sleep phases, n = 288 probes of which **114 evicted**.

| method | all-probe | **evicted** | evicted median CE | val CE |
|---|---|---|---|---|
| floor (no context) | 0.000 | 0.000 | 3.91 | — |
| (c) cascading, no adapter | 0.413 | 0.000 | 6.19 | — |
| ours: compaction-supervised | 0.451 | **0.000** | 5.20 | 0.0003 |
| **(a) uniform replay** | 0.465 | **0.123** (14/114) | **1.63** | 0.0005 |
| (b) reflection | 0.389 | 0.000 | 7.02 | 4.31 |
| ours + mask head | 0.323 | 0.009 | 5.26 | 0.0008 |
| (d) full context (ceiling) | 0.712 | — | — | — |

Three findings.

**1. Consolidation recovers evicted facts.** Uniform replay recovers 14 of 114, **+4.0σ**
above zero. Every earlier run had all arms pinned at 0.000, which is what an unmeasurable
benchmark looks like rather than a failed method.

**2. The compaction gate is strictly harmful.** Ours recovers zero where uniform recovers
0.123 — a **3.1σ gap in favour of uniform**, with a +15.85σ salience signal feeding the gate.
Selecting on a strong, verified salience signal is *worse than not selecting at all*.

**3. Evicted CE tracks recovery here.** Uniform has both the lowest evicted median CE (1.63
against cascading's 6.19) and the only non-zero recovery. On the ambiguous benchmark those two
signals disagreed; on the corrected one they agree, which is a point in favour of the
corrected setup rather than of CE as a metric.

The all-probe column separates the methods much less: uniform 0.465 vs cascading 0.413 is only
**+0.9σ**. The effect lives in the evicted column, and claims should be made there.

### Why the gate hurts

The mechanism is coverage, not signal quality. Both adapters memorise their targets equally
well — val CE reaches 0.0003 for ours and 0.0005 for uniform. The difference is *what* they
memorise. Compaction keeps 25.6% of spans, heavily concentrated on the fact spans it likes,
and it never keeps `auth_header` or `config_flag` at all. Uniform samples the same budget
across the whole trajectory, so it covers facts the compactor systematically drops.

Project B tests the same question on an independent signal. Its KL gate clears its matched
control at +6.77σ, loses to uniform at `floor_weight: 0` (0.510 vs 0.708), and ties both
uniform and random-span selection once non-selected tokens get a 0.1 weight (0.729 each)
([project_b_findings.md](project_b_findings.md)). **Across two verified signals, importance
gating never beats uniform training at this scale**: here it is strictly worse through
coverage, in Project B it is no better than random once its weighting defect is fixed.

## The same mask works for abstention

Compaction-Aware Abstention (arXiv:2608.29934) trains a LoRA on compressor survival masks so a
7B model *refuses* when the evidence was evicted, reporting a 97% cut in hallucination. It
explicitly does not try to recover the evicted content. That is the same free label this
project feeds to consolidation, pointed the other way, so both directions can be run on one
dataset at one scale.

Trained on the train split, evaluated on eval. n = 288, of which 114 evicted and 174 retained.

| | no adapter | abstain LoRA |
|---|---|---|
| refuses when the fact was evicted | 0.000 | **0.728** (+17.5σ) |
| refuses when the fact is still there | 0.000 | 0.109 |
| accuracy on retained facts | 0.684 | **0.770** (+1.81σ) |
| hallucinates on evicted facts | 1.000 | **0.272** |
| abstention margin | 0.000 | **0.619** |

**Hallucination on evicted probes falls by 72.8%**, and the bidirectional check passes: this is
not a refuse-everything model. It declines only 10.9% of probes whose answer is still in
context, and its accuracy on those *rises* from 0.684 to 0.770, though that margin is 1.81σ and
not individually significant.

This reproduces the direction of arXiv:2608.29934 at 0.5B rather than 7B, with 72.8% instead of
their 97%. More useful here is the contrast with everything above it, on one signal and one
run:

| what the mask is asked to do | result |
|---|---|
| recover the evicted fact (consolidation) | **0.000** |
| notice the fact is missing (abstention) | **0.728**, hallucination down 72.8% |

**The compaction mask tells you what was lost. Knowing what was lost is enough to abstain; it
is not enough to recover.** The signal is real and useful — just not for the thing this project
originally proposed to do with it.

## The oracle control is inconclusive: it was undertrained

The positive control gets perfect selection (ground-truth facts, no compactor) plus 20
templated surface forms per fact, with the eval phrasing held out. Trained closed-book.

| arm | all-probe | retained | evicted |
|---|---|---|---|
| cascading, no adapter | 0.413 | 0.684 | 0.000 |
| uniform replay | 0.465 | 0.690 | **0.123** |
| ours | 0.451 | 0.747 | 0.000 |
| **oracle** | **0.587** | **0.948** | **0.035** |
| oracle, closed book | 0.031 | — | 0.031 |

**Do not read the evicted column as a ceiling.** `sleep/oracle.py` inherited the sleep-phase
default of 80 steps, which at batch 4 is **320 of 5,760 examples — 5.6% of a single epoch**,
in 14.4 seconds. The consolidation arms it is meant to bound each ran 25 phases of 80 steps,
2,000 in total. Augmentation cannot help a fact the optimiser never reaches, and the closed-book
score of 0.031 says exactly that: the adapter trained on `question -> answer` pairs still cannot
answer one closed-book, because it saw 94% of them zero times.

What the run does show is worth keeping. Even on 5.6% of an epoch the oracle adapter reaches
**0.948 on retained probes** against a 0.684 baseline, the best all-probe score of any adapter
(0.587 against full context's 0.712), and the lowest mean probe CE in the table. Augmented
`question -> answer` training sharply improves *reading the context*, long before it injects
any knowledge.

Fixed: the step budget now scales with the dataset (3 epochs by default, 1,440 steps per epoch
here) and the script warns when the budget covers less than one epoch. A re-run costs about 13
minutes. Until then the headroom between uniform's 0.123 and the 0.712 ceiling is unmeasured.

## The scoring eval is still confounded

| method | MC acc (all) | MC acc (evicted) | σ vs chance |
|---|---|---|---|
| cascading | 0.590 | 0.202 | −1.11 |
| ours | 0.573 | 0.272 | +0.68 |
| uniform | 0.566 | 0.237 | −0.17 |
| reflection | 0.608 | 0.237 | −0.17 |
| ours + mask | 0.562 | 0.272 | +0.68 |

Chance is 0.241. Nothing separates. Note that uniform recovers 12.3% of evicted facts *by
generation* while ranking at chance — so the ranking eval is failing to detect knowledge that
demonstrably exists.

The reason is a second defect, distinct from the question ambiguity. Distractors are drawn
from `FACT_TEMPLATES[key]["values"]`, a pool of about five values per fact key. With 48
trajectories and 10 keys, **each value is the correct answer for five or six other
trajectories**. An adapter that memorised the corpus assigns low loss to every candidate. The
question fix did not touch this; fixing it needs per-trajectory unique values or a much larger
pool.

**Do not cite the scoring numbers.** They measure the distractor pool, not the weights.

## What survives for the writeup

- **Importance gating never beats uniform training**, on two independent verified signals.
  Here it is strictly worse (0.000 vs 0.123); in Project B it ties random selection once a
  zero-weight defect is fixed. This is the headline.
- **The oracle control is pending a re-run** at a real training budget; the first attempt saw
  5.6% of one epoch and cannot bound anything.
- **The same free mask supports abstention where it fails at recall**: 0.000 recovered versus a
  72.8% cut in hallucination, both measured on one run. A negative and a positive from one
  signal, which is a more complete claim than either alone.
- **The compaction decision is a strong salience signal** (+15.85σ over chance, +8.40σ over a
  positional control) that is nonetheless the wrong thing to train on.
- **The elicitation finding**: the decision must be read off the logits, never generated.
- **Three measurement lessons**, all learned the hard way: a precondition number without a
  control is not a test; a CE reduction is not evidence of acquisition without distractors;
  and a distractor set must not contain answers correct for a different item in the same eval.

See [paper/draft_combined.md](../paper/draft_combined.md).
