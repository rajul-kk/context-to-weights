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
"sleep" phases interleaved with the trajectory. Getting a usable decision out of a small
compactor turns out to depend less on its scale than on how the decision is elicited: asked
to generate a ranked list of spans, a 1.5B model answers positionally and carries no signal,
while the same model's keep judgment read directly off its logits separates salient from
filler content at 2.56x against a positional control. How much signal there is depends on
how distinguishable salient content is: 2.56x where the synthetic text marks its own
important lines, 1.61x on HotpotQA prose with the positional control at chance, and 1.10x
when facts and filler are drawn from a single template bank and are stylistically
identical. On synthetic long software-development
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
4. **Salience lift**, a cheap precondition test measuring whether a given compactor's
   keep/drop decision carries any supervision at all, scored against a positional control.
   It is the first thing to run, and it disqualifies most small compactors.
5. The finding that *how the decision is elicited* dominates compactor scale: generated
   index lists carry no signal at 0.5B or 1.5B, while a logit read at 1.5B clears the
   control at 2.56x.

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
A frozen compactor selects `ceil(ρ · n)` spans to keep verbatim, where ρ is the keep
fraction, and writes a summary of the rest. Log `(span_text, kept)` for every span.

The compactor does not *generate* its decision. For each span the frozen backbone is asked
whether deleting that line would make a later question unanswerable, and the keep score is
read straight off the logits as `log p(Yes) − log p(No)`; the top `ceil(ρ · n)` spans by
score are kept. Section 5 gives the reason: asking a model to emit a ranked index list
fails at every scale we tested, while reading the same judgment off the logits works.

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
plausible filler. Each fact has a held-out probe. 48 train / 24 eval trajectories, 120 turns.
An unmarked variant drops the `One thing to lock in:` framing that introduces each fact.
HotpotQA, repackaged so supporting paragraphs sit early and distractors act as filler, is
the external-validity check; a LoCoMo loader exists but its answer-substring labelling is
weaker and is not yet validated.

**Models.** Compaction runs under Qwen2.5-1.5B-Instruct, which clears the positional control
by the widest margin of the models we measured. Consolidation targets Qwen2.5-0.5B-Instruct. The two
decouple cleanly because the event log is plain text, and compaction is a one-off offline
pass, so a larger compactor costs little. SmolLM2-360M-Instruct is the cross-family check.
LoRA rank 16, α 32, on all attention and MLP projections. Single T4.

**Arms.** (ours) compaction-supervised; (a) uniform replay over all spans at matched example
count; (b) reflection consolidation; (c) cascading compaction, no weight update; (d) full
context; and a no-context floor. Arms (a), (b), (c) and ours all see the same
post-compaction context at evaluation, so the adapter is the only difference between them.

**Precondition.** Before any comparison, `salience lift` — the fact-span keep rate over the
filler-span keep rate — must beat the positional control ("keep the first N spans"), which
scores 1.68x on this data because planted facts sit early. A compactor at or below the
control selects no better than chance, and ours and uniform replay then draw from the same
distribution, making the comparison vacuous by construction.

**Precondition.** Before any comparison, *salience lift* — the fact-span keep rate over the
filler-span keep rate — must beat a positional control ("keep the first N spans"), which
scores 1.68x on this data because planted facts sit early. A compactor at or below the
control selects no better than chance; ours and uniform replay then draw from the same
distribution and the comparison is vacuous by construction.

**Metrics.** The headline number is accuracy on *evicted* probes — those whose gold answer
is absent from the retained context. Overall retention accuracy, median and mean per-token
CE, prompt token cost and GPU-seconds per sleep phase are also reported.

Baseline (a) is the control that matters. It fixes the machinery, the example budget and the
compute, and varies only whether the compactor chose the spans. Any gap between (a) and ours
is attributable to the compaction signal itself.

## 5. Does the compaction decision carry signal at all?

The method inherits whatever judgment the compactor has, so we measure that first. Salience
lift is reported against the positional control on 6 eval trajectories, 12 compaction
events, 36 fact spans.

| Compactor | Elicitation | Fact keep | Filler keep | Lift | Control |
|---|---|---|---|---|---|
| SmolLM2-360M | logit scoring | 0.528 | 0.243 | **2.17x** | 1.32x |
| Qwen2.5-0.5B | index list | 0.139 | 0.260 | 0.53x | 1.68x |
| Qwen2.5-0.5B | logit scoring | 0.111 | 0.262 | 0.42x | 1.68x |
| Qwen2.5-1.5B | index list | 0.139 | 0.260 | 0.53x | 1.69x |
| Qwen2.5-1.5B | logit scoring | 0.611 | 0.239 | **2.56x** | 1.68x |
| Qwen2.5-1.5B | logit scoring, unmarked | 0.278 | 0.252 | 1.10x | 1.57x |
| Qwen2.5-1.5B | logit scoring, HotpotQA | 0.400 | 0.248 | **1.61x** | 0.98x |

The control is recomputed per run, so each compactor is judged against the positional
baseline for its own selections rather than a fixed threshold.

Three findings, and the last is the one that constrains the rest.

**Elicitation dominates scale.** Asked to emit a ranked index list, both Qwen sizes reply
with a contiguous prefix `0, 1, 2, ...` on 100% of events, regardless of the text at those
positions. Tripling the parameter count changes nothing, while switching to a logit read at
fixed size takes the same model from 0.53x to 2.56x. Scale is also non-monotonic: SmolLM2-360M
scores 2.17x under logit scoring where Qwen2.5-0.5B manages 0.42x on identical data, so the
usable range is a property of the individual model's behaviour on the probe rather than of
capacity. Because planted facts sit early in
a trajectory, that behaviour *looks* like signal until spans are shuffled before
presentation, at which point it is exactly random selection. The same model, asked the same
question one span at a time with the answer read off the logits, reaches 2.56x.

**A positional control is not optional.** Three earlier measurements of ours — 3.99x, 1.68x,
1.69x — were artifacts, of a silent fallback path, of first-N answers meeting early-planted
facts, and of an index-sorted truncation that reimposed positional bias after shuffling.
Every one erred favourably. The control reproduces the artifact exactly whenever it occurs,
which is what makes it worth reporting alongside every lift figure.

**The signal depends on the conversation flagging its own importance.** Our default
generator introduces each fact with `One thing to lock in:`. An unmarked variant changes
nothing else, and on it the 1.5B scoring compactor falls from 2.56x to 1.10x — chance, below
its control. It had been reading the marker.

This is a different shortcut from the one our heuristic baseline exploits. The heuristic
keys on cue vocabulary shared with the generator's templates and is indifferent to the
marker (4.43x marked, 4.38x unmarked); the model is indifferent to the vocabulary and keys
on the marker. Two backends, two shortcuts, neither visible without a variant that removes
them.

**On natural text the signal is real.** HotpotQA, repackaged so that supporting sentences
scatter through an early window, gives a positional control of 0.98x — position carries no
information — and the same compactor reaches 1.61x. Fact keep 0.400 against a 0.255 chance
rate is about 2.6 sigma at n=60 fact spans.

Taken together the three rows describe a spectrum rather than a threshold. The compaction
decision carries supervision in proportion to how distinguishable salient content is from
its surroundings: strongly when the text marks it, not at all when facts and filler are
drawn from one template bank in one register, and measurably on real prose in between. Our
unmarked variant is the floor of that spectrum rather than a neutral test, and we report it
as such.

Salience lift costs one compaction pass and disqualifies a compactor before any training
runs. We would recommend it, with a positional control and a marker-free variant, as
standard practice for any work that consumes a compaction decision as supervision.

## 6. Consolidation results

TBD — see `docs/results.md`.

Planned figures: retention accuracy on evicted facts across all six arms (headline);
median vs mean validation CE across sleep phases, showing the divergence [1] reports;
the compaction-ratio sweep, testing whether compaction-supervised consolidation degrades
more slowly than uniform replay as ρ falls.

## 7. Limitations

**The method inherits the compactor's judgment, and which models have it is not predictable
from size.** SmolLM2-360M clears the control at 2.17x while Qwen2.5-0.5B fails at 0.42x on
identical data. We can measure whether a given compactor carries signal but we cannot yet
predict it, so salience lift has to be run per compactor rather than assumed from scale.

**The evaluation set is small.** 6 eval trajectories and 36 fact spans. The 2.56x figure is
about 4.9 sigma above chance, but its margin over the positional control rests on roughly
2.7 sigma. This needs widening before the result is load-bearing.

**Blind spots may be systematic.** At 1.5B the compactor never keeps `auth_header`,
`config_flag` or `deploy_target` spans, and in a hand-check the auth-header line scored
lowest of twelve. If that survives a larger evaluation it is more interesting than the
headline: the free signal would have a characteristic shape, and the paper should say which
kinds of knowledge compaction-supervision cannot reach.

**Generator vocabulary leaks into the heuristic baseline.** Eight of the fourteen cue
phrases in our heuristic compactor appear verbatim in the generator's fact templates, so its
4.4x lift measures that overlap rather than judgment. The heuristic is a debugging backend
and none of its numbers are reported. An unmarked variant of the generator, which drops the
`One thing to lock in:` framing, changes the heuristic's lift by 0.05x — confirming the leak
is vocabulary, not the marker. Trajectories are short relative to a real agent
session. Consolidation is measured on facts, not on procedures or style. A single seed per
configuration at this budget; variance across seeds is not yet characterised.

## References

[1] Beyond Inference-Only Deployment. arXiv:2605.24657, May 2026.
[2] SCoL: Self-Consolidating Language Models. arXiv:2605.07076, May 2026.
[3] What to Keep, What to Forget. arXiv:2607.08032, Jul 2026.
[4] CompactionRL. arXiv:2607.05378, Jul 2026.
