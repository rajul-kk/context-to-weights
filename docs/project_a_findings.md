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

This is the same result as Project B, on an independent signal: see
[project_b_findings.md](project_b_findings.md), where a KL gate scoring +2.6σ over its control
still loses to uniform distillation (0.510 vs 0.708). **Two mechanisms, two verified signals,
the same conclusion — uniform coverage beats importance gating at this scale.**

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

- **Uniform coverage beats importance gating**, replicated on two independent signals with a
  live positive control in each. This is the headline.
- **The compaction decision is a strong salience signal** (+15.85σ over chance, +8.40σ over a
  positional control) that is nonetheless the wrong thing to train on.
- **The elicitation finding**: the decision must be read off the logits, never generated.
- **Three measurement lessons**, all learned the hard way: a precondition number without a
  control is not a test; a CE reduction is not evidence of acquisition without distractors;
  and a distractor set must not contain answers correct for a different item in the same eval.

See [paper/draft_combined.md](../paper/draft_combined.md).
