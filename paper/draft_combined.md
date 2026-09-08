# Measured, Not Asked: Salience Signals for Weight Consolidation at Small Scale

*Workshop draft. Combines Projects A and B and adds the declarative-attention arm.
Numbers marked TBD are filled from `docs/results*.md`.*

## Abstract

A language agent that wants to move knowledge out of its context and into its weights needs
a salience signal: something that says which parts of the context are worth keeping. Three
such signals are available for free, without a reward model or an LLM judge — the keep/drop
decision a context compactor already makes, the divergence between a model's predictions
with and without a document in context, and the attention declarations a model emits when
asked to route its own reading. We build all three, and find that they divide cleanly along
a line that is not the one we expected. Signals the model is *asked to state* collapse at
small scale: both Qwen2.5-0.5B and 1.5B answer a span-selection request with the contiguous
prefix `0, 1, 2, ...` on 100% of events regardless of content, which under shuffling is
exactly random selection. Signals *measured from the model's behaviour* survive: the same
1.5B model reaches 2.56x salience lift when its keep decision is read off the logits rather
than generated, and on a separate task it routes its own attention to the right region of an
8k-token context at +9σ when the choice is read off the logits versus no better than naming a
fixed slot when it is asked to state it — a gap that widens, not closes, from 0.5B to 1.5B.
We argue that "measure the model, do not ask it" is the governing constraint for
free-supervision methods below the scales where instruction-following is reliable, and we
supply the measurement discipline — a matched control beside every signal-strength figure —
that makes the distinction visible. Neither consolidation method we build beats uniform
training on the same budget; the elicitation finding and the discipline are what survive.

## 1. Introduction

Context is expensive and transient. Weights are cheap to read and permanent. A long-running
agent therefore wants a way to move what matters from one to the other, and the hard part is
not the training — a LoRA pass is routine — but deciding *what* to move.

Prior work reaches for supervision that costs something. Nightly consolidation [1] writes a
reflection with a second generation pass. SCoL [2] learns placement with meta-RL.
CompactionRL [4] trains the compactor itself with RL. Each buys its signal.

We ask what is already free, and find three candidates:

- **The compaction decision.** Every time an agent compacts, it decides which spans survive
  verbatim. That is a salience label, emitted at no extra cost.
- **The context gap.** Run a frozen model twice on the same continuation, once with a skill
  document in context and once without. The per-token KL between the two distributions is
  the document's causal contribution, computed from two forward passes.
- **The attention declaration.** Declarative Attention [5] has a model state which region of
  context it needs to read. That statement is also a salience label.

All three are free. They are not equally usable, and the axis that separates them is the
contribution of this paper.

**Contributions.**

1. Compaction-supervised consolidation: LoRA SFT on compactor-kept spans, with reservoir
   replay, in periodic sleep phases (§3).
2. Context-gap distillation: importance-weighted KL distillation gated on with-versus-without
   context divergence (§4).
3. Declarative attention below the scale at which declaring works: the elicitation threshold,
   and a logit-read substitute that clears it (§5).
4. **A matched control beside every signal-strength figure** — a cheap precondition test for
   whether a candidate signal carries information at all, which caught seven separate artifacts
   in our own pipeline, six inflating a result favourably and one hiding a negative (§6).
5. The governing finding: generated signals fail and measured signals survive, on three
   independent mechanisms, at every scale we can afford (§7).

## 2. Related work

**Beyond Inference-Only Deployment [1]** establishes that periodic LoRA consolidation beats
cascading compaction, and supplies the evaluation discipline we adopt: median rather than
mean per-token cross-entropy, which tracks judged accuracy at r = +0.99 while mean CE moves
the wrong way at r = −0.51. Its targets are reflected facts, written by a second generation
pass. We consolidate on the compactor's raw decision instead, removing that pass and removing
reflection quality as a confound. Their recipe is our baseline.

**SCoL [2]** learns *where* to write consolidated knowledge via meta-RL — the complementary
axis. We ask what gets written and answer with supervised learning.

**What to Keep, What to Forget [3]** frames compaction as rate-distortion and characterises
what survives a budget. It stops at the compactor; we consume its object as training signal.

**CompactionRL [4]** trains the compactor with RL. We hold it frozen and train the backbone
from its decisions; the two compose.

**Declarative Attention [5]** has a 27–31B model declare `<global>`, `<focus>` or `<local>`
inside its chain of thought, and an inference engine skip most of the KV-cache read
accordingly — 52.0% and 31.1% fewer attended tokens at −1.27pp and −2.75pp accuracy,
zero-shot. The paper explicitly leaves training-based extensions open. We take up that
invitation from below: we ask whether the declaration is reliable at the scales a single-GPU
researcher can run, find that it is not, and show what replaces it.

**Skill-to-LoRA [6]** distils each skill into its own adapter by uniform self-distillation,
with no importance gating. It is the direct baseline for §4. **ThinkSwitch [7]** removes
reasoning traces structurally, again ungated. **PEAM [8]** gates on a
parameterization-worthiness score over episodic trajectory pairs — the nearest neighbour to
our gate, on different objects with a different signal. **Titans [9]** defines surprise as a
gradient of an associative-memory loss with respect to the input; ours is a surprise signal
in the same spirit, realised as a divergence between two forward passes rather than a
gradient norm, and applied offline rather than at test time.

## 3. Signal one: the compaction decision

[Method as in Project A: segment the evictable prefix, keep `ceil(ρ·n)` spans, log
`(span_text, kept)`, sleep-phase LoRA SFT with reservoir replay. See `paper/draft.md`.]

## 4. Signal two: the context gap

[Method as in Project B: per-token KL between teacher (document present) and student
reference (document absent), used as both the gate and the loss, with a random-span control
matched on active-token budget. See `paper/draft_b.md`.]

## 5. Signal three: the attention declaration

Declarative Attention assumes the model can say where to look. We test that assumption at
0.5B–1.5B, where a single-GPU agent actually lives.

**Setup.** A trajectory is split into `K = 8` contiguous regions. The model sees all of them,
labelled, plus a question whose answer lies in exactly one. Two elicitations:

- **generate** — the model replies `FOCUS: <k>`, parsed like a tool call, as in [5].
- **read** — for each region we ask whether it contains the answer and take
  `log p(Yes) − log p(No)` off the logits, choosing the argmax. Costs `K` forward passes
  instead of one generation.

**Controls.** Random choice is `1/K`. Modal share exposes a model that always names the same
region. The decisive test is the **content shuffle**: permute which content sits in which
slot and re-elicit. A declaration that tracks content follows the answer to its new slot; one
that tracks position does not. `content_dependence` is the shuffled hit rate above chance;
`slot_stable_rate` is how often the choice does *not* move when content does — near `1/K`
means content-driven, near 1 means position-driven.

**Results.** HotpotQA trajectories with supporting paragraphs scattered across the whole
context, `K = 8`, gold region permuted per probe, 384 probes.

| | 0.5B generate | 0.5B read | 1.5B generate | 1.5B read |
|---|---|---|---|---|
| hit rate (chance 0.125) | 0.206 | 0.333 | 0.154 | 0.372 |
| σ over best-constant (0.148) | +2.8 | +7.7 | +0.3 | +9.1 |
| `content_dependence` | 0.055 | **0.206** | 0.021 | **0.240** |
| `slot_stable_rate` | 0.328 | 0.125 | 0.422 | 0.141 |
| unparsed rate | 0.02 | 0.00 | 0.19 | 0.00 |
| mean attended fraction | 0.145 | 0.135 | 0.291 | 0.137 |

`read` clears every control at both scales: `content_dependence` 0.21–0.24 and
`slot_stable_rate` at `1/K`, so when a region's content moves to a new slot the choice
follows it. `generate` does not track content at either scale. Its raw hit rate clears the
constant control at 0.5B (+2.8σ), but the shuffle unmasks that as position: `content_dependence`
is 0.055, a quarter of `read`'s. At 1.5B the generated declaration is indistinguishable from
naming one fixed region (+0.3σ), the model refuses the `FOCUS:` format on 19% of probes, and
`slot_stable_rate` rises to 0.42. **The larger model is worse at saying where to look, and
identical on reading it.** `read` attends 13.6% of tokens — a larger cut than the paper's
52.0% / 31.1% — but at a 0.33–0.37 hit rate, so the routing is real and roughly 3x chance,
not yet accurate enough to deploy unsupervised. See `docs/declarative.md`.

## 6. Salience lift, and why a control is not optional

For a candidate signal to be worth training on, spans it marks must contain what later
questions need more often than spans it does not. Salience lift is the ratio of the fact-span
keep rate to the filler-span keep rate. It costs one compaction pass.

A raw lift figure is close to meaningless. We report it against a **positional control** —
what "keep the first N spans" would score on the same data — because in any trajectory where
important content appears early, position alone produces a large apparent lift.

We recommend this because it repeatedly caught us. Seven artifacts, each invisible until a
control exposed it:

| Artifact | Apparent effect | Caught by |
|---|---|---|
| Silent fallback to a rule-based compactor | 3.99x | decision provenance counter |
| "Keep at most N" answered `NONE` on every event | 100% empty keeps | empty-keep rate |
| Model answered with the first N spans | 1.68x | positional control |
| `sorted(kept)[:budget]` reimposed position after shuffling | 1.69x | kept-index distribution |
| Supporting paragraphs stacked at the front of a benchmark | 1.76x | positional control (2.44x) |
| Span-mean pooling inverted a real per-token gate | 0.38x of chance | matched-budget control |
| Falling CE on gold answers read as knowledge transfer | 6.29 -> 2.24 | distractor ranking (at chance) |

None was a modelling error; all were construction. We think that is the norm rather than our
misfortune, and that free-supervision work should report a control beside every signal
strength figure as a matter of course.

The last row is the one we most want to generalise, because it is the only one that moved
the result *against* us and so was not caught by disbelief. The context-gap gate (§4) is a
per-token quantity, and we selected on its mean over contiguous spans. Required API
identifiers occur as two or three high-KL tokens inside a dozen near-zero syntax tokens, so
code spans average low; prose restatements of a rule are uniformly moderately surprising, so
they average high. The gate bought prose and skipped code, scoring 0.118 required-token
coverage against a matched-budget random control's 0.307 — a 5.14σ deficit. Scored per token
the same signal clears the same control at +2.62σ. The precondition test existed and was run;
it reported 0.125 next to a `top_frac` of 0.25 and nothing compared the two. **A precondition
number without a control is not a test**, and the ratio of a signal's p90 to its median (here
8x) is a cheap advance warning that pooling it by the mean will destroy it.

The final row generalises furthest, because it concerns a metric the field reports routinely.
Consolidation lowered per-token cross-entropy on evicted gold answers from 6.29 to 2.24, which
we read as knowledge that had entered the weights but could not be decoded. Ranking each gold
answer against distractors drawn from the same fact bank shows that reading was wrong: no arm
scores above the 0.240 chance rate, and the mean margin to the best distractor is negative
everywhere and grows more negative after training. The adapter lowered loss on the gold answer
and its distractors alike, having learned the answer vocabulary without the binding. **A CE
reduction on a target is not evidence of knowledge acquisition unless it is checked against
distractors from the same distribution.** The check costs one extra forward pass per candidate
and it converted a hedged positive into a clean negative.

## 7. Measured beats asked

| Signal | Elicitation | Backbone | Result |
|---|---|---|---|
| compaction decision | generated index list | 0.5B / 1.5B | 0.53x, below control |
| compaction decision | logit read | 360M | 2.17x, clears |
| compaction decision | logit read | 1.5B | 2.56x, clears |
| context gap | measured, no elicitation | 360M | separates identifiers from markdown |
| attention declaration | generated `FOCUS: k` | 0.5B | +0.055 content dep, positional |
| attention declaration | generated `FOCUS: k` | 1.5B | +0.021 content dep, at chance |
| attention declaration | logit read | 0.5B | +0.206 content dep, +7.7σ |
| attention declaration | logit read | 1.5B | +0.240 content dep, +9.1σ |

Two observations.

**Elicitation dominates scale.** Tripling parameters leaves a generated index list at chance;
switching the same model to a logit read takes it from 0.53x to 2.56x. The attention
declaration repeats this exactly: the generated `FOCUS:` at 1.5B carries no content signal
(`content_dependence` 0.021), the same model's logit read carries a strong one (0.240, +9.1σ).
The failure is not that small models lack the judgment — it is that they cannot express it in
a structured format on demand, and asking harder as they scale makes it worse.

**Scale is not monotonic.** SmolLM2-360M clears the control at 2.17x while Qwen2.5-0.5B fails
at 0.42x on identical compaction data with identical code. And on the attention declaration
the 1.5B model is *worse* than the 0.5B at the generated `FOCUS:` — its hit rate falls from
0.206 to 0.154, its unparsed rate rises from 2% to 19%, and it leans harder on naming a fixed
slot. Whether a model carries a usable signal in a given elicitation is a property of its
behaviour on the probe, not of its size, and has to be measured per model rather than assumed.

The practical rule for anyone building free-supervision pipelines below frontier scale:
**derive the signal from what the model does, not from what it says it does.** The context
gap obeys this by construction, which is why it works on a 360M backbone. Compaction
supervision obeys it once the decision is read rather than generated. Declarative attention,
as published, does not — and §5 measures the cost.

## 8. Limitations

Single seed per configuration. Evaluation sets for the compaction lift are small — 36 to 60
fact spans — so the HotpotQA lift of 1.61x sits about 2.6σ above chance and needs widening
before it carries weight; the declarative-attention arm is larger at 384 probes. Our synthetic
generator's unmarked variant is a floor case rather than a neutral test: facts and filler come
from one template bank in one register, so they are near indistinguishable by construction. We
cannot test the 27B+ regime where [5] reports, so our declarative-attention result bounds the
generated declaration from below — it does not work at 1.5B — and does not contradict the
accuracy [5] reports at 27B. The `read` substitute we propose is untested at their scale. All
compute is one T4.

## References

[1] Beyond Inference-Only Deployment. arXiv:2605.24657.
[2] SCoL: Self-Consolidating Language Models. arXiv:2605.07076.
[3] What to Keep, What to Forget. arXiv:2607.08032.
[4] CompactionRL. arXiv:2607.05378.
[5] Language Models Can Control Their Own Attention. arXiv:2609.02737.
[6] Skill-to-LoRA. arXiv:2606.16769.
[7] ThinkSwitch. arXiv:2606.01080.
[8] PEAM. arXiv:2605.27762.
[9] Titans: Learning to Memorize at Test Time. arXiv:2501.00663.
