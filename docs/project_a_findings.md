# Project A results

Two questions, answered separately.

1. **Does the compactor's keep/drop decision carry a salience signal?** Yes, +15.85σ.
2. **Does consolidating on that signal beat consolidating on random spans?** **No — it is
   worse.** Across ten synthetic corpora uniform replay recovers 6.5% of evicted facts and
   compaction-supervised consolidation 2.2% (paired t(9)=3.91). HotpotQA does not separate.

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

### Across data seeds

The table above is one synthetic corpus (seed 0). Four more corpora, each regenerated from
scratch (`generate_synthetic --seed N`) and run through cascading, compaction and uniform:

| data seed | n evicted | cascading | compaction | uniform |
|---|---|---|---|---|
| 0 | 114 | 0.000 | 0.000 | **0.123** (14) |
| 1 | 102 | 0.000 | 0.029 (3) | **0.078** (8) |
| 2 | 112 | 0.000 | 0.000 | **0.009** (1) |
| 3 | 98 | 0.000 | 0.020 (2) | **0.061** (6) |
| 4 | 112 | 0.000 | 0.036 (4) | **0.107** (12) |
| mean | | 0.000 | 0.017 | **0.076** |

Five more corpora (data seeds 5 to 9, 2026-11-11) were run as a replication with the same
settings, through `notebooks/a1_seeds.ipynb`. Summaries are in `results/consolidation/`:

| data seed | n evicted | cascading | compaction | uniform |
|---|---|---|---|---|
| 5 | 103 | 0.000 | 0.049 (5) | **0.058** (6) |
| 6 | 104 | 0.000 | 0.010 (1) | **0.019** (2) |
| 7 | 109 | 0.000 | 0.009 (1) | **0.073** (8) |
| 8 | 112 | 0.000 | 0.045 (5) | **0.080** (9) |
| 9 | 111 | 0.000 | 0.018 (2) | **0.045** (5) |
| mean | | 0.000 | 0.026 | **0.055** |

**Uniform beats compaction on all five corpora, significantly.** The margins are 0.123, 0.049,
0.009, 0.041 and 0.071: a paired t of **3.10 on 4 df**, above the 2.78 needed. Cascading
recovers nothing on any corpus. Seed 0 was the most favourable draw and seed 2 the least; at
three seeds the effect looked like it might be seed-0 noise (t(2)=1.81), and two more settled
it. The gain is on evicted facts only: all-probe accuracy is level (uniform −0.021 on average,
t(4)=−0.88), with uniform well below compaction on seed 4 (0.295 vs 0.378).

**The replication agrees.** Uniform beats compaction on all five new corpora too, with a
smaller margin (0.029 against 0.059 on the first five; t(4)=2.90, p=0.044). Over all ten
corpora the paired t is **3.91 on 9 df** (p=0.004), against 2.26 needed. All-probe accuracy is
again level (uniform −0.053, t(4)=−1.07). One corpus takes about 30 minutes on a T4.

## HotpotQA consolidation: no separation

The first consolidation run on natural prose since the question-scoping fix. Same pipeline and
models as the synthetic runs (1.5B compactor, 0.5B target), `configs/kaggle_hotpotqa.yaml`,
32 eval trajectories, 160 probes of which 92 evicted, one seed.

| method | all-probe | evicted |
|---|---|---|
| cascading, no adapter | 0.188 | 0.043 (4/92) |
| ours: compaction-supervised | 0.144 | 0.033 (3/92) |
| uniform replay | 0.194 | 0.054 (5/92) |
| full context (ceiling) | 0.375 | — |

**Nothing separates.** Uniform leads compaction by 2 of 92 (+0.7σ), and neither adapter clears
cascading, which recovers 4 of 92 with no weight update at all: some HotpotQA answers survive in
other paragraphs or the model's prior, so the evicted floor is not zero here as it is on
synthetic. The only visible effect is compaction's all-probe drop (0.144 vs 0.188), the same
sign as on synthetic. The full-context arm evicts almost nothing (5 probes), so its evicted
column is not a ceiling. HotpotQA neither replicates nor contradicts the synthetic result; at
92 evicted probes and a non-zero floor it cannot resolve a gap of the size seen on synthetic.

## LoCoMo (M5): gate not cleared

Registered in `docs/protocol.md` before running: on LoCoMo the compaction arm is trained only if
the Qwen2.5-0.5B compactor keeps answer spans more often than filler (salience lift above 1.0) on
at least one fold. It does not, on either fold, and the 1.5B compactor recorded beside it does
not either. `results/m5_locomo/gate.json`, from `eval/span_report.py` on each fold's eval events.

| fold | compactor | fact keep | filler keep | salience lift | vs chance | positional control lift |
|---|---|---|---|---|---|---|
| 0 | 0.5B | 0.128 | 0.258 | **0.49** | −6.24σ | 0.25 |
| 0 | 1.5B | 0.200 | 0.255 | 0.79 | −2.63σ | 0.24 |
| 1 | 0.5B | 0.116 | 0.259 | **0.45** | −6.83σ | 0.18 |
| 1 | 1.5B | 0.164 | 0.257 | 0.64 | −4.48σ | 0.18 |

**The registered verdict is "gate not cleared"**, so no M5 sleep run was made. On conversational
text the compactor keeps the spans that later answer a question about half as often as the rest:
significantly below chance, though still above a rule that keeps by position alone. The mask that
was a strong salience signal on synthetic data (+15.85σ over chance) points the wrong way here, so
the compaction arm would have nothing to learn that uniform does not.

## Fixing coverage directly does not fix it

Project B's analogous failure (zero gradient on non-selected tokens) was fixed by a small
floor weight, which closed the gap to uniform. We tested whether A's failure is the same
shape: mix a share of the compactor's *dropped* spans back into the training budget, holding
the total budget equal to `compaction`'s.

| method | evicted (n=114) | note |
|---|---|---|
| ours: compaction, 0% mix | 0.000 (0/114) | reproduces the run above exactly |
| compaction, 25% random mix | 0.018 (2/114) | draws uniformly from the dropped pool |
| compaction, 50% random mix | 0.009 (1/114) | |
| compaction, 25% stratified mix | 0.035 (4/114) | round-robins one dropped span per distinct fact key |
| compaction, 50% stratified mix | 0.000 (0/114) | |
| (a) uniform replay | 0.123 (14/114) | reproduces the run above exactly |

**This does not work, and it is not the same failure as Project B's.** Swapping in a random
quarter or half of the *dropped* pool moves evicted recovery from 0.000 to essentially nothing
(1-2 of 114). The reason is the dropped pool's composition: the compactor drops 5,956 filler
spans against only a handful of fact spans per trajectory, so a random draw from it almost
never lands on `auth_header` or `config_flag` specifically — it mostly adds more filler
diversity. Uniform's 0.123 comes from sampling **all** spans, kept and dropped together, every
sleep phase, across 25 phases; that is a much larger number of independent draws at a better
mixing ratio than a one-shot 25-50% swap into an already fact-concentrated budget.

**Targeting the sampling helps a little, and does not fix it either.** `from_compaction_stratified`
groups each event's dropped spans by fact key and round-robins across keys instead of drawing
flat-random, verified on a synthetic 200:1 filler-to-fact skew to hit the rare key on every
trial versus 2-5% for random draws. At 25% mix this roughly doubles recovery over random
mixing (0.035 vs 0.018, 4 vs 2 of 114) — the intended direction, and still an order of
magnitude short of uniform's 0.123. **At 50% mix it drops to zero**, worse than random mixing
at the same budget. One run each, so this could be small-sample noise: at ~11 evicted
instances per fact key, one training run learning or not learning a single key swings the
count by several probes. It could also mean the round-robin, applied per event rather than
across the whole corpus, starves an event's well-covered facts (`db_engine`, `owner`) to make
room for a rare one it happens to see, without the reverse ever correcting for it. Either way,
neither ablation closes the gap, and stratification is not the one-line fix floor-weight was
for Project B.

**So the two projects fail for related but distinct reasons.** B's gate excluded knowledge
from the gradient entirely at the token level within an otherwise-trained example; a small
floor weight restored it fully. A's gate excludes entire fact spans from the training set
across every phase; re-including them, whether at random or targeted by key, recovers at most
4 of 114 facts against uniform's 14. A's failure is not a weighting artifact the way B's was.

Project B's gate, independently, clears its matched control at +6.77σ, and once its own
weighting defect is fixed it is no worse than a random-selection arm with the same masking
(5 seeds, t(4)=0.74) — but no better either
([project_b_findings.md](project_b_findings.md)). **Neither project shows importance gating
beating uniform coverage, but for different reasons**: A's gate is confirmed strictly harmful
because it excludes whole facts from training; B's gate is neutral because, once its own
defect is fixed, selecting by KL rank performs identically to selecting at random. Report
them separately rather than as one mechanism.

## The same mask works for abstention

Compression-Aware Abstention (arXiv:2608.29934) trains a LoRA on compressor survival masks so a
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

**Caveat, found 13 Nov.** Training truncated each prompt from the right at `max_length` (1,024
here), so any example whose retained context pushed it past the limit lost its whole target and
trained on a prompt token instead. Evaluation keeps the end of the prompt and is unaffected. This
run's contexts are not stored locally, so how many examples were cut cannot be checked; the effect
is that those examples taught nothing about answering or refusing, which can only have weakened
the reported effect. `sleep/trainer.py` now
keeps the end, as the scorers do, and U4 retrains with that fix.

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

*Superseded by U4 and U4s below: with a random-mask control the abstention half of this contrast
shrinks to a 21% cut that misses significance on synthetic, and reverses on LoCoMo.*

## Abstention on LoCoMo (U4): does not replicate

Registered in `docs/protocol.md` before running: the same abstention LoRA trained on the
compaction mask against a control trained on a random mask of the same size (198 refusals of 256
training examples in both), Qwen2.5-0.5B, five seeds on each of two folds, ten paired runs. It
holds only if compaction hallucinates less than random (two-sided paired t, p < 0.05) and loses
no more than 0.05 retained accuracy against the model with no adapter. `results/u4_locomo/`.

| mean of 10 runs | no adapter | compaction mask | random mask | paired t (comp − rand) |
|---|---|---|---|---|
| hallucinates on evicted probes | 0.963 | 0.354 | **0.157** | +2.51, p = 0.033 |
| accuracy on retained probes | 0.419 | 0.209 | 0.111 | +2.72, p = 0.024 |
| refuses on retained probes | 0.000 | 0.488 | 0.733 | −2.57, p = 0.030 |
| refuses on evicted probes | 0.000 | 0.640 | 0.838 | −2.47, p = 0.036 |
| abstention margin (evicted − retained) | 0.000 | 0.152 | 0.105 | +1.23, p = 0.249 |

**The registered verdict is "does not replicate"**, and it fails on both counts: the compaction
mask hallucinates *more* than the random control, significantly, and costs 0.21 retained
accuracy, four times the guard. Neither adapter learns to check its context. Both learn to refuse
often, the random one more so, and refusing often is what cuts hallucination on this measure.
The one number that measures selectivity, the abstention margin, is slightly higher for
compaction but not distinguishable from random. Seeds vary widely (compaction's refusal rate on
retained probes runs from 0.06 to 0.95), so 256 examples of which 77% are refusals does not pin
the policy down.

The synthetic result above had no random-mask control, so U4s below repeats the control there.

## Abstention control on synthetic (U4s): a smaller, selective effect that misses the bar

Registered in `docs/protocol.md` before running: the original synthetic corpus and compactor (the
span report reproduces exactly: lift 2.81, +15.85σ over chance), compaction mask against a random
mask of the same size (108 refusals of 288 training examples), five seeds, the U4 bar.
`results/u4s_synthetic/`. Training now keeps the end of over-long prompts (the longest is 1,164
tokens against a limit of 1,024), so this is not a byte-for-byte rerun of the original.

| mean of 5 runs | no adapter | compaction mask | random mask | paired t (comp − rand) |
|---|---|---|---|---|
| hallucinates on evicted probes | 1.000 | **0.786** | 0.977 | −2.65, p = 0.057 |
| accuracy on retained probes | 0.684 | **0.883** | 0.847 | +4.45, p = 0.011 |
| refuses on retained probes | 0.000 | 0.000 | 0.000 | — |
| abstention margin (evicted − retained) | 0.000 | 0.214 | 0.021 | +2.71, p = 0.054 |

**The registered verdict is "does not hold"**: compaction is below random in all five pairs, but
p = 0.057 misses α = 0.05. The accuracy guard passes easily (retained accuracy rises 0.20, for both
arms, from training on answers). What differs from LoCoMo is the shape: here neither adapter
refuses a single retained probe, and the compaction adapter refuses only where the fact was
evicted, while the random one almost never refuses. So on synthetic the mask does carry
information, selectively; on LoCoMo it does not.

The effect is far smaller than the original 72.8%: a 21% mean cut in hallucination, from 3% to
58% across seeds (the seed-0 run, the nearest to the original, cuts 31%). The single uncontrolled
run was at the top of a wide distribution, or owed part of its size to the old truncation.
Report 72.8% only with this beside it.

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

- **Project A's compaction gate loses to uniform**: uniform wins on all ten synthetic
  corpora (0.065 vs 0.022 mean evicted recovery, paired t(9)=3.91), and mixing dropped spans
  back in does not close the gap. On HotpotQA (one seed, 92 evicted) nothing separates, from
  each other or from cascading. This is the headline for A.
- **Project B's gate is neutral, not harmful, once its weighting defect is fixed**: the KL
  ranking performs identically to random selection under the same masking (5 seeds, t(4)=0.74).
  The two projects fail for different reasons; report them separately rather than as one
  mechanism.
- **The oracle control is pending a re-run** at a real training budget; the first attempt saw
  5.6% of one epoch and cannot bound anything.
- **The abstention positive does not survive its controls**: 72.8% less hallucination on one
  uncontrolled synthetic run. With a random-mask control of the same size, on the same corpus
  (U4s, five seeds) the cut is 21% and selective (no retained refusals) but misses significance
  (p = 0.057); on LoCoMo (U4, ten runs) the compaction mask hallucinates more than random
  (0.354 vs 0.157, p = 0.033) and loses 0.21 retained accuracy. At most a weak, corpus-dependent
  signal; do not claim the mask supports abstention.
- **The compaction decision is a strong salience signal** (+15.85σ over chance, +8.40σ over a
  positional control) that is nonetheless the wrong thing to train on. It does not carry over to
  conversation: on LoCoMo both compactors keep answer spans below chance (lift 0.45 to 0.79), so
  M5 stopped at its registered gate.
- **The elicitation finding**: the decision must be read off the logits, never generated.
- **Three measurement lessons**, all learned the hard way: a precondition number without a
  control is not a test; a CE reduction is not evidence of acquisition without distractors;
  and a distractor set must not contain answers correct for a different item in the same eval.

See [paper/draft_combined.md](../paper/draft_combined.md).
