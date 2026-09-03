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
than generated, and 2.17x for a 360M model. We argue that "measure the model, do not ask
it" is the governing constraint for free-supervision methods below the scales where
instruction-following is reliable, and we supply the measurement discipline — salience lift
against a positional control — that makes the distinction visible.

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
4. **Salience lift with a positional control** — a cheap precondition test for whether a
   candidate signal carries information at all, which caught five separate artifacts in our
   own pipeline, every one of which inflated results favourably (§6).
5. The governing finding: generated signals fail and measured signals survive, at every
   scale we can afford (§7).

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
that tracks position does not. `content_dependence` is the shuffled hit rate above chance.

**Results.** TBD — see `docs/results_declare.md`.

## 6. Salience lift, and why a control is not optional

For a candidate signal to be worth training on, spans it marks must contain what later
questions need more often than spans it does not. Salience lift is the ratio of the fact-span
keep rate to the filler-span keep rate. It costs one compaction pass.

A raw lift figure is close to meaningless. We report it against a **positional control** —
what "keep the first N spans" would score on the same data — because in any trajectory where
important content appears early, position alone produces a large apparent lift.

We recommend this because it repeatedly caught us. Five artifacts, each inflating the result
in the favourable direction, each invisible until a control exposed it:

| Artifact | Apparent effect | Caught by |
|---|---|---|
| Silent fallback to a rule-based compactor | 3.99x | decision provenance counter |
| "Keep at most N" answered `NONE` on every event | 100% empty keeps | empty-keep rate |
| Model answered with the first N spans | 1.68x | positional control |
| `sorted(kept)[:budget]` reimposed position after shuffling | 1.69x | kept-index distribution |
| Supporting paragraphs stacked at the front of a benchmark | 1.76x | positional control (2.44x) |

None was a modelling error; all were construction. We think that is the norm rather than our
misfortune, and that free-supervision work should report a control beside every signal
strength figure as a matter of course.

## 7. Measured beats asked

| Signal | Elicitation | Backbone | Result |
|---|---|---|---|
| compaction decision | generated index list | 0.5B / 1.5B | 0.53x, below control |
| compaction decision | logit read | 360M | 2.17x, clears |
| compaction decision | logit read | 1.5B | 2.56x, clears |
| context gap | measured, no elicitation | 360M | separates identifiers from markdown |
| attention declaration | generated `FOCUS: k` | 0.5B / 1.5B | TBD |
| attention declaration | logit read | 0.5B / 1.5B | TBD |

Two observations.

**Elicitation dominates scale.** Tripling parameters leaves a generated index list at chance;
switching the same model to a logit read takes it from 0.53x to 2.56x. The failure is not that
small models lack the judgment — it is that they cannot express it in a structured format on
demand.

**Scale is not monotonic.** SmolLM2-360M clears the control at 2.17x while Qwen2.5-0.5B fails
at 0.42x on identical data with identical code. Whether a compactor carries signal is a
property of that model's behaviour on the probe, not of its size, and has to be measured per
model rather than assumed.

The practical rule for anyone building free-supervision pipelines below frontier scale:
**derive the signal from what the model does, not from what it says it does.** The context
gap obeys this by construction, which is why it works on a 360M backbone. Compaction
supervision obeys it once the decision is read rather than generated. Declarative attention,
as published, does not — and §5 measures the cost.

## 8. Limitations

Single seed per configuration. Evaluation sets are small — 36 to 60 fact spans — so the
HotpotQA lift of 1.61x sits about 2.6σ above chance and needs widening before it carries
weight. Our synthetic generator's unmarked variant is a floor case rather than a neutral test:
facts and filler come from one template bank in one register, so they are near
indistinguishable by construction. We cannot test the 27B+ regime where [5] reports, so our
declarative-attention result speaks only to small scale and does not contradict theirs. All
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
