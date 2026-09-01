# Compaction-Supervised Sleep Consolidation

*Workshop draft. Numbers marked TBD are filled from `docs/results.md`.*

## Abstract

Long-running language agents survive limited context windows by compacting: periodically
deciding which spans of their history to keep verbatim and which to summarise away.
Consolidating that history into weights instead is known to work far better than cascading
compaction, but existing recipes need an extra generation pass in which a model reflects on
what it learned, and existing work on *where* and *when* to consolidate reaches for
reinforcement learning. We observe that the compactor has already answered the *what*
question. Its keep/drop decision is a salience label, produced for free on every compaction
event, and it is exactly the supervision a consolidation pass needs. We train a LoRA adapter
with ordinary cross-entropy on the spans a frozen compactor chose to keep, in periodic
"sleep" phases interleaved with the trajectory. On synthetic long software-development
dialogues with facts planted early and probed late, this recovers TBD% of full-context
retention on facts that compaction had already evicted, against TBD% for cascading
compaction with no weight update and TBD% for the same LoRA machinery trained on uniformly
sampled spans. The signal costs nothing: no reward model, no judge, no reflection pass.

## 1. Introduction

An agent that runs for hours hits its context limit repeatedly. The standard response is
compaction — summarise the old part of the transcript, keep the recent part, continue. Done
repeatedly this cascades, and detail degrades with every round.

Recent work shows the alternative is much better. Nightly LoRA consolidation retains 80.4%
±1.3% of knowledge where cascading compaction retains 36.8%±3.0%, between an 11.8%
no-context floor and a 90.1% full-context ceiling [1]. But that recipe consolidates
*reflected* content: an LLM writes a summary of what it learned, and the adapter trains on
that summary. The reflection is an extra generation pass, and its quality bounds the
consolidation.

Meanwhile the question of *what* and *when* to consolidate has been attacked with meta-RL
[2] and compaction itself has been trained with RL [4]. Both are heavy, and in our own
earlier attempt at this problem the RL formulation was where things broke.

Our observation is narrow and, we think, useful: **the compactor's keep/drop decision is
already a supervised label for what to internalise, and it is emitted for free every time
an agent compacts.** A span the compactor chose to preserve verbatim is, by the compactor's
own judgment, load-bearing. Training on exactly those spans turns "what should sleep
consolidation learn" from an RL problem into a cross-entropy problem.

Contributions:

1. Compaction-supervised sleep consolidation: LoRA SFT on compactor-kept spans, with a
   reservoir replay buffer, run in periodic phases across a long trajectory.
2. An auxiliary variant that additionally predicts the keep/drop mask, testing whether
   learning salience helps beyond training on salient content.
3. A controlled comparison against uniform-replay consolidation, reflection consolidation,
   cascading compaction and a full-context ceiling, at fixed example budget and fixed
   compute.

## 2. Related work

**Beyond Inference-Only Deployment [1]** establishes that periodic LoRA consolidation beats
cascading compaction by a wide margin, and supplies the evaluation discipline we adopt:
median rather than mean per-token cross-entropy, which tracks LLM-judged accuracy at
r = +0.99 while mean CE moves the wrong way at r = −0.51. Its consolidation targets are
reflected and synthesised facts — a model writes a summary of what it learned and the
adapter trains on the summary. We consolidate on the compactor's raw keep/drop decision
instead, which removes the synthesis pass entirely and removes reflection quality as a
confound. Their recipe is our baseline (b).

**SCoL [2]** learns *where* in the network to write consolidated knowledge, via meta-RL.
That is the complementary axis to ours: SCoL asks where the write goes, we ask what gets
written, and we answer it with supervised learning rather than RL.

**What to Keep, What to Forget [3]** frames compaction as rate-distortion and characterises
which content survives a given budget. It stops at the compactor and proposes no downstream
consolidation. We take the object it analyses — the keep/drop decision — and use it as
training signal.

**CompactionRL [4]** trains the compactor itself with RL. We hold the compactor frozen and
train the backbone from its decisions; the two are composable in principle, and a better
compactor should on our account yield a better consolidation label.

The distinction we want to be precise about: [1] shows *that* consolidation beats
compaction; [2] and [4] apply RL to adjacent decisions; [3] analyses the compaction
decision without consuming it. None of them uses the compaction decision as supervision.

## 3. Method

### 3.1 Free labels from compaction

Run a trajectory until the context budget trips. Segment the evictable prefix into spans.
A frozen compactor — the same backbone under a fixed prompt — selects at most
`ceil(ρ · n)` spans to keep verbatim, where ρ is the keep fraction, and writes a summary of
the rest. Log `(span_text, kept)` for every span. This costs one forward pass that the
agent was going to make anyway.

### 3.2 Sleep phases

Every K compaction events, train the LoRA adapter for a few hundred steps of teacher-forced
cross-entropy. Each kept span becomes one example: a short session-keyed cue as prompt, the
span text as target, loss on the target only. A reservoir buffer of capacity C holds
previously-kept spans; each phase draws C/2 of them alongside the new ones to blunt
forgetting across phases. The adapter is never reset, so consolidation accumulates.

### 3.3 Mask-prediction auxiliary

A linear head over the final hidden states predicts the keep/drop label per target token,
trained with binary cross-entropy alongside the SFT loss and weighted by λ. Dropped spans
enter the batch for this loss only; their language-modelling labels stay masked out. This
separates "train on salient content" from "learn what salience looks like."

## 4. Experimental setup

**Data.** Synthetic multi-turn software-development dialogues, generated on CPU. Each
trajectory plants six load-bearing facts — datastore choices, header names, rate limits,
queue names, flag names, version pins — in an early window, then continues with topically
plausible filler. Each fact has a held-out probe. 48 train / 16 eval trajectories, 120 turns.
LoCoMo and HotpotQA are the planned extension.

**Models.** Qwen2.5-0.5B-Instruct primary, Qwen2.5-1.5B-Instruct and SmolLM2-360M-Instruct
for cross-scale and cross-family checks. LoRA rank 16, α 32, on all attention and MLP
projections. Single T4.

**Arms.** (ours) compaction-supervised; (a) uniform replay over all spans at matched example
count; (b) reflection consolidation; (c) cascading compaction, no weight update; (d) full
context; and a no-context floor. Arms (a), (b), (c) and ours all see the same
post-compaction context at evaluation, so the adapter is the only difference between them.

**Metrics.** The headline number is accuracy on *evicted* probes — those whose gold answer
is absent from the retained context. Overall retention accuracy, median and mean per-token
CE, prompt token cost and GPU-seconds per sleep phase are also reported.

Baseline (a) is the control that matters. It fixes the machinery, the example budget and the
compute, and varies only whether the compactor chose the spans. Any gap between (a) and ours
is attributable to the compaction signal itself.

## 5. Results

TBD — see `docs/results.md`.

Planned figures: retention accuracy on evicted facts across all six arms (headline);
median vs mean validation CE across sleep phases, showing the divergence [1] reports;
the compaction-ratio sweep, testing whether compaction-supervised consolidation degrades
more slowly than uniform replay as ρ falls.

## 6. Limitations

**The method inherits the compactor's judgment, and small compactors have none.** Salience
lift — the ratio of the fact-span keep rate to the filler-span keep rate — measures whether
the keep/drop decision carries supervision at all. SmolLM2-360M-Instruct scores 0.00x on our
trajectories: it keeps chit-chat and drops every planted fact, which is worse than random.
At that scale there is no signal to be supervised by, and the method reduces to uniform
replay. The compactor must clear a lift of 1.0 before any of this is worth running, and
establishing that threshold across model scales is a precondition we report rather than
assume. If the consolidation target is too small to compact well, the compactor and the
target must be decoupled.

The synthetic generator plants facts with lexical cues that a keyword-based compactor can
exploit; all reported results therefore use the model compactor, and the heuristic backend
is documented as a debugging tool only. Trajectories are short relative to a real agent
session. Consolidation is measured on facts, not on procedures or style. A single seed per
configuration at this budget; variance across seeds is not yet characterised.

## References

[1] Beyond Inference-Only Deployment. arXiv:2605.24657, May 2026.
[2] SCoL: Self-Consolidating Language Models. arXiv:2605.07076, May 2026.
[3] What to Keep, What to Forget. arXiv:2607.08032, Jul 2026.
[4] CompactionRL. arXiv:2607.05378, Jul 2026.
