# Measured, Not Asked: Salience Signals for Weight Consolidation at Small Scale

*Workshop draft. Combines Projects A and B and adds the declarative-attention arm. Source
results: `docs/project_a_findings.md`, `docs/project_b_findings.md`, `docs/declarative.md`.*

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
Two different measurements of that same model, querying its logits and instrumenting where it
attends, both recover the signal, with the logit read ahead or level at 0.5B, 1.5B and 7B, so "measure, do not
ask" picks out a family of methods rather than a single one.
We argue that "measure the model, do not ask it" is the governing constraint for
free-supervision methods below the scales where instruction-following is reliable, and we
supply the measurement discipline — a matched control beside every signal-strength figure —
that makes the distinction visible. On the compaction signal, uniform replay recovers more
evicted facts than gating on a +15.9σ salience signal on all five synthetic corpora (7.6% vs
1.7% mean, paired t(4)=3.10), and mixing a share of
the dropped spans back into training does not close the gap: **uniform coverage beats
importance gating**, with a positive control showing the setup could have registered a win.
On an independent context-gap signal, a gate that clears its matched control at +6.77σ loses
to full-weight uniform training at a zero floor weight, and once that weighting defect is
fixed the KL ranking performs identically to random selection under the same masking (5
seeds, t(4)=0.74) — neutral, not beneficial. The two signals fail for different reasons; we
report them separately rather than as one mechanism.

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
   whether a candidate signal carries information at all, which caught eight separate artifacts
   in our own pipeline, six inflating a result favourably and two hiding a possible positive (§6).
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

Segment the evictable prefix, keep `ceil(ρ·n)` spans, log `(span_text, kept)`, and run
sleep-phase LoRA SFT on the kept spans with reservoir replay. The label is free: the compactor
had to make that decision anyway.

**The signal is strong.** On 48 synthetic trajectories with 97 compaction events, fact spans
are kept at 0.663 against a filler rate of 0.236 — a 2.81x lift, **+15.85σ** over chance and
**+8.40σ** over a positional control, with zero heuristic fallbacks or empty keeps.

**Consolidating on it is worse than not selecting at all.** Backbone Qwen2.5-0.5B, 25 sleep
phases, 288 probes of which 114 evicted:

| method | all-probe | evicted |
|---|---|---|
| (c) cascading, no adapter | 0.413 | 0.000 |
| **ours: compaction-supervised** | 0.451 | **0.000** |
| compaction, 25% random mix | — | 0.018 |
| compaction, 50% random mix | — | 0.009 |
| compaction, 25% stratified mix | — | 0.035 |
| compaction, 50% stratified mix | — | 0.000 |
| **(a) uniform replay** | 0.465 | **0.123** |
| (b) reflection | 0.389 | 0.000 |
| (d) full context (ceiling) | 0.712 | — |

Uniform recovers 14 of 114 evicted facts, **+4.0σ** above zero; the compaction-gated arm
recovers none, a **3.1σ** gap in favour of not using the signal. Across five independently
generated corpora uniform wins every time, 0.076 against 0.017 mean evicted recovery, paired
t(4) = 3.10 (critical 2.78); seed 0 was the most favourable draw. All-probe accuracy is level
(t(4) = −0.88), so the gain is specific to evicted facts. On HotpotQA (one seed, 92 evicted
probes) nothing separates: cascading with no adapter recovers 4, compaction 3, uniform 5,
because some answers survive in other paragraphs or the model's prior. That run neither
replicates nor contradicts the synthetic result. Both adapters memorise their
targets equally well (validation CE 0.0005 and 0.0003), so the difference is coverage, not
optimisation: the compactor keeps `db_engine`, `owner` and `version_pin` on 100% of events but
`auth_header` on 0% and `config_flag` on 3.4%, while uniform samples the same budget across
everything. These numbers reproduced exactly across two independent sessions.

Mixing a random 25% or 50% of the compactor's *dropped* spans back into the training budget,
holding the total budget fixed, moves recovery from 0.000 to essentially nothing (2 and 1 of
114). The dropped pool is 5,956 filler spans against a handful of fact spans per trajectory,
so a random draw from it rarely lands on the specific facts the compactor excludes. Grouping
dropped spans by fact key and round-robining across keys instead of drawing flat-random
roughly doubles recovery at 25% mix (4 of 114) but falls to zero at 50% — one run each, and
still an order of magnitude short of uniform. Uniform's 0.123 comes from sampling every span,
kept and dropped, across 25 phases, not from one proportional mixing pass, targeted or not.
This is a different failure from Project B's (below): A excludes whole fact spans from the
training set across every phase, and no remixing strategy tested reintroduces them at
anything close to uniform's rate.

**The same mask succeeds at the opposite task.** Following arXiv:2608.29934, a LoRA trained on
the same free label to *refuse* when the evidence was evicted reaches 0.728 abstention on
evicted probes against 0.000 for the base model (+17.5σ), cutting hallucination by **72.8%**,
while refusing only 10.9% of probes whose answer is still present and raising retained accuracy
from 0.684 to 0.770. The compaction mask tells a model what was lost; that is enough to abstain
and not enough to recover.

## 4. Signal two: the context gap

**Scoring.** For a skill document `d`, demonstration query `q` and response `r`, form
`T = chat(system, d ++ q)` and `S = chat(system, q)`, run the frozen backbone on `T ++ r` and
`S ++ r`, and score each response token by

```
kl_i = KL( p_teacher(· | T, r_<i) || p_student(· | S, r_<i) )
```

Tokens are grouped into sentence and code-line spans via the tokenizer's offset mapping; a
span scores the mean of its tokens.

**Gating and loss.** Keep the top ρ fraction of spans (weight 1) and give the rest a floor
weight (0 by default); token-level gating is an ablation. The teacher is the same model with
its LoRA adapter disabled and the document in context, the student the adapter-enabled model
without it, and the loss is the weighted KL `sum_i w_i KL_i / sum_i w_i`. One model in memory,
two forward passes per step. One adapter per skill category, so cross-skill interference is
measurable.

**Setup.** Eight invented tool-use APIs in three categories, each a `SKILL.md`, nine
demonstrations and twelve held-out tasks; the APIs are fictional so the base model cannot know
them. Qwen2.5-1.5B-Instruct, LoRA rank 16, one T4. Arms: full document in prompt, S2L-style
uniform distillation, no document, a random-span control at matched active-token budget, and
the KL gate. A task passes when every required string appears in the output.

**Gate verification.** Required-token coverage — the fraction of each demo's required API
identifiers inside the gate's selection — against a matched-budget random control: **0.597 vs
0.392, +6.77σ**. An earlier cached score file under an older tokenizer gave -5.14σ (§6).

**Result.** `uniform` here trains on every token at full weight, with no subsetting; `random`
uses the same 25%-selected, floor-weighted masking as the gate but with no signal behind the
choice. With `floor_weight` at 0 the gated arm reaches **0.510**, about 2.9σ below `uniform`
and below `random` at 0.583: the prefix that conditions each selected identifier gets no
gradient. A 0.1 floor lifts it above 0.65 on all five seeds tested. The comparison that
isolates the KL signal is `kl_top` against `random`, which share identical masking: paired
across five seeds the mean difference is +0.021 (t(4)=0.74) and the sign is inconsistent
seed to seed — the ranking adds nothing over random selection. Both masked arms nominally
beat full-weight `uniform` (`random`: +0.035, t(4)=3.90, significant; `kl_top`: +0.056,
t(4)=2.33, not quite), a separate, narrower finding about masked versus full-weight training
at a fixed step budget, not about selection quality. The internalised skill keeps partial
function under a mismatched retrieved document (0.292 vs 0.000 for the prompted skill).

## 5. Signal three: the attention declaration

Declarative Attention assumes the model can say where to look. We test that assumption at
0.5B–1.5B, where a single-GPU agent actually lives.

**Setup.** A trajectory is split into `K = 8` contiguous regions. The model sees all of them,
labelled, plus a question whose answer lies in exactly one. Two elicitations:

- **generate** — the model replies `FOCUS: <k>`, parsed like a tool call, as in [5].
- **read** — for each region we ask whether it contains the answer and take
  `log p(Yes) − log p(No)` off the logits, choosing the argmax. Costs `K` batched region
  prompts instead of one generation.
- **attention** — no elicitation at all: we run the model on the whole context plus the
  question and read the attention mass the final query position places on each region.
  **attention_late** restricts this to the second half of the layers. This is the probing
  route taken by Sentinel and the retrieval-head line, run here on the same model, same
  layouts and same probes as the other two.

**Controls.** Random choice is `1/K`. Modal share exposes a model that always names the same
region. The decisive test is the **content shuffle**: permute which content sits in which
slot and re-elicit. A declaration that tracks content follows the answer to its new slot; one
that tracks position does not. `content_dependence` is the shuffled hit rate above chance;
`slot_stable_rate` is how often the choice does *not* move when content does — near `1/K`
means content-driven, near 1 means position-driven.

**Results.** HotpotQA trajectories with supporting paragraphs scattered across the whole
context, `K = 8`, gold region permuted per probe, 384 probes, **three seeds** (mean +/- sd).

| model | elicitation | hit (chance 0.125) | σ over constant | `content_dependence` | `slot_stable_rate` | unparsed |
|---|---|---|---|---|---|---|
| 0.5B | generate | 0.193 +/- 0.012 | 2.2 | 0.085 +/- 0.021 | 0.281 | 0.019 |
| 0.5B | attention | 0.151 +/- 0.003 | 0.2 | 0.014 +/- 0.012 | 0.882 | 0.000 |
| 0.5B | attention_late | 0.234 +/- 0.011 | 4.0 | 0.113 +/- 0.029 | 0.356 | 0.000 |
| 0.5B | **read** | **0.332 +/- 0.004** | **7.7** | **0.203 +/- 0.003** | 0.136 | 0.000 |
| 1.5B | generate | 0.150 +/- 0.008 | 0.1 | 0.030 +/- 0.023 | 0.413 | 0.170 |
| 1.5B | attention | 0.286 +/- 0.023 | 6.0 | 0.163 +/- 0.010 | 0.447 | 0.000 |
| 1.5B | attention_late | 0.374 +/- 0.017 | 9.2 | 0.258 +/- 0.028 | 0.200 | 0.000 |
| 1.5B | **read** | **0.378 +/- 0.005** | **9.3** | **0.241 +/- 0.005** | 0.136 | 0.000 |

**The generated declaration fails at both scales and degrades with size.** Its raw hit rate
clears the constant control at 0.5B (+2.2σ), but the shuffle unmasks that as position:
`content_dependence` is 0.085, well under half of `read`'s. At 1.5B it is indistinguishable
from naming one fixed region (+0.1σ), the model refuses the `FOCUS:` format on 17% of probes,
and `slot_stable_rate` rises to 0.41.

**`read` wins clearly at 0.5B; at 1.5B it is a tie, not a crossover.** At 0.5B the self-query
is clearly ahead of the best attention variant — hit 0.332 vs 0.234, content dependence 0.203
vs 0.113, both significant on a paired t across the three seeds (t(2) = 24.9 and 4.9 against a
critical value of 4.30). At 1.5B, `read` and `attention_late` are 0.378 vs 0.374 on hit rate
(t(2) = 0.30) and 0.241 vs 0.258 on content dependence (t(2) = −0.85) — neither margin clears
significance in either direction. A single-seed run of this ablation had reported late-layer
attention nominally ahead at 1.5B; three seeds show that was layout noise, not a reversal.
Cost is comparable: `read` issues `K` batched region prompts, `attention` one pass over the
whole context, covering roughly the same number of tokens.

**Layer choice decides whether probing works at all.** Averaged over every layer, attention is
useless at 0.5B: content dependence 0.014, naming the same slot on 88% of probes. Restricted
to the late half it clears its control at both scales. Early layers carry position rather than
content and dominate the average — one Qwen2.5-1.5B layer-0 query·key product reaches 152,967
against a late-layer typical peak near 300. A probing baseline reported without this ablation
would understate itself by more than an order of magnitude on `content_dependence`.

**Scale separates asking from measuring, up to a point.** From 0.5B to 1.5B, content
dependence moves `generate` 0.085 → 0.030 (**−0.055**), `read` 0.203 → 0.241 (+0.038),
`attention` 0.014 → 0.163 (+0.149) and `attention_late` 0.113 → 0.258 (+0.145): every measured
signal improves except the generated one, which closes the 0.5B gap between `read` and
`attention_late` to a tie by 1.5B. A third scale, Qwen2.5-7B-Instruct in nf4 (three seeds,
128 probes), does not continue that trend: `attention_late`'s content dependence falls back to
0.174 while `read` holds at 0.247, and `read` leads on hit rate 0.388 vs 0.333 (t(2) = 6.06,
significant) and on content dependence (t(2) = 3.21, not significant). It rules out the
extrapolation that attention probing keeps closing the gap past 1.5B. See
`docs/declarative.md`.

## 6. Salience lift, and why a control is not optional

For a candidate signal to be worth training on, spans it marks must contain what later
questions need more often than spans it does not. Salience lift is the ratio of the fact-span
keep rate to the filler-span keep rate. It costs one compaction pass.

A raw lift figure is close to meaningless. We report it against a **positional control** —
what "keep the first N spans" would score on the same data — because in any trajectory where
important content appears early, position alone produces a large apparent lift.

We recommend this because it repeatedly caught us. Eight artifacts, each invisible until a
control exposed it:

| Artifact | Apparent effect | Caught by |
|---|---|---|
| Silent fallback to a rule-based compactor | 3.99x | decision provenance counter |
| "Keep at most N" answered `NONE` on every event | 100% empty keeps | empty-keep rate |
| Model answered with the first N spans | 1.68x | positional control |
| `sorted(kept)[:budget]` reimposed position after shuffling | 1.69x | kept-index distribution |
| Supporting paragraphs stacked at the front of a benchmark | 1.76x | positional control (2.44x) |
| Cached scores from an older tokenizer inverted the gate | -5.14σ | re-scoring in the current environment (+6.40σ) |
| Falling CE on gold answers read as knowledge transfer | 6.29 -> 2.24 | distractor ranking (at chance) |
| Eval questions that did not identify which item was meant | 50% ceiling read as 100% | duplicate-answer audit |

None was a modelling error; all were construction. We think that is the norm rather than our
misfortune, and that free-supervision work should report a control beside every signal
strength figure as a matter of course.

The tokenizer row is the one we most want to generalise, because it moved the result
*against* us, so disbelief did not catch it, and because we misdiagnosed it once. The
context-gap gate (§4) is per-token and we selected on its mean over spans. Scored from a
cached file produced under an older tokenizer, which split identifiers such as `vx_stash`
into five pieces, code spans averaged low and prose restatements high: 0.118 required-token
coverage against a matched control's 0.307, -5.14σ. We attributed this to mean-pooling.
Re-scoring in the current environment, with no code change, gives 0.586 against 0.392,
**+6.40σ**. The first gate check had no control at all: it reported 0.125 next to a
`top_frac` of 0.25 and nothing compared the two. Seeing the stale number, we then marked the
downstream comparison confounded; re-running it reproduced the original result exactly. A
retraction needs the same evidence as a claim. **A precondition number without a control is
not a test, and a cached score file is only valid in the environment that produced it.**

The last two rows concern metrics rather than signals, and they compound. Consolidation
lowered per-token cross-entropy on evicted gold answers from 6.29 to 2.24, which we first read
as knowledge that had entered the weights but could not be decoded. Ranking each gold answer
against distractors from the same fact bank put every arm at the 0.240 chance rate, which we
then read as knowledge that was never acquired. Both readings were premature. Auditing the
benchmark showed that 98.3% of its questions had more than one correct answer across
trajectories -- the generator scoped questions by a service drawn from a pool of eight, so
"Which header carries the auth token?" was asked of 48 conversations with four different
answers. A model with perfect recall of every trajectory could not have exceeded 50%, and the
ranking distractors were themselves other trajectories' correct answers. **A CE reduction on a
target is not evidence of knowledge acquisition unless it is checked against distractors from
the same distribution; and a distractor set must not contain answers that are correct for a
different item in the same eval.** The audit -- count the distinct answers each question
string receives, and compute what a perfect memoriser could score -- is what made Project A's
result measurable: after scoping every question to its trajectory, uniform replay recovers
0.123 of evicted facts and compaction-gated consolidation 0.000 on the first corpus (§3). It is the cheaper check
we should have run before the experiment rather than after it.

## 7. Measured beats asked

| Signal | Elicitation | Backbone | Result |
|---|---|---|---|
| compaction decision | generated index list | 0.5B / 1.5B | 0.53x, below control |
| compaction decision | logit read | 360M | 2.17x, clears |
| compaction decision | logit read | 1.5B | 2.56x, clears |
| context gap | measured, no elicitation | 360M | separates identifiers from markdown |
| attention declaration | generated `FOCUS: k` | 0.5B | +0.085 content dep, positional |
| attention declaration | generated `FOCUS: k` | 1.5B | +0.030 content dep, at chance |
| attention declaration | logit read | 0.5B | +0.203 content dep, +7.7σ |
| attention declaration | logit read | 1.5B | +0.241 content dep, +9.3σ |
| attention declaration | logit read | 7B nf4 | +0.247 content dep, +5.0σ |
| attention declaration | attention probe, late layers | 0.5B | +0.113 content dep, +4.0σ |
| attention declaration | attention probe, late layers | 1.5B | +0.258 content dep, +9.2σ |
| attention declaration | attention probe, late layers | 7B nf4 | +0.174 content dep, +3.8σ |

Three observations.

**Elicitation dominates scale.** Tripling parameters leaves a generated index list at chance;
switching the same model to a logit read takes it from 0.53x to 2.56x. The attention
declaration repeats this exactly: the generated `FOCUS:` at 1.5B carries no content signal
(`content_dependence` 0.030), while the same model's logit read (0.241, +9.3σ) and its
late-layer attention (0.258, +9.2σ) both carry a strong one. The failure is not
that small models lack the judgment — it is that they cannot express it in a structured format
on demand, and asking harder as they scale makes it worse.

**Measured is one claim; *which* measurement is another.** The two measured routes — asking
the model a semantic question and reading its logits, or instrumenting where it actually
attends — are not interchangeable: the self-query leads at 0.5B, ties at 1.5B and leads
again at 7B (§5), and an all-layer attention average is near-useless at 0.5B while a
late-layer one is not. The self-query is the safer default at every scale we tested. "Measure the model"
is therefore the start of a design decision, not the end of one.

**Scale is not monotonic.** SmolLM2-360M clears the control at 2.17x while Qwen2.5-0.5B fails
at 0.42x on identical compaction data with identical code. And on the attention declaration
the 1.5B model is *worse* than the 0.5B at the generated `FOCUS:` — its hit rate falls from
0.193 to 0.150, its unparsed rate rises from 2% to 17%, and it leans harder on naming a fixed
slot. Whether a model carries a usable signal in a given elicitation is a property of its
behaviour on the probe, not of its size, and has to be measured per model rather than assumed.

The practical rule for anyone building free-supervision pipelines below frontier scale:
**derive the signal from what the model does, not from what it says it does.** The context
gap obeys this by construction, which is why it works on a 360M backbone. Compaction
supervision obeys it once the decision is read rather than generated. Declarative attention,
as published, does not — and §5 measures the cost.

## 8. Limitations

Project A has five data seeds for the main comparison and one for every ablation; Project
B has five seeds for the gate comparison and one for each sweep. The HotpotQA compaction lift (1.34-1.44x, +3.75σ and +4.77σ at 288-386 fact spans) is a precondition measurement; HotpotQA consolidation is one seed at 92 evicted probes with a non-zero no-adapter floor, and does not separate the arms. The headline consolidation result is synthetic. The declarative-attention arm is larger at 384 probes (128 at 7B). Our synthetic
generator's unmarked variant is a floor case rather than a neutral test: facts and filler come
from one template bank in one register, so they are near indistinguishable by construction. We
cannot test the 27B+ regime where [5] reports, so our declarative-attention result bounds the
generated declaration from below — it does not work at 1.5B, and we did not run it at 7B —
and does not contradict the accuracy [5] reports at 27B. The `read` substitute
we propose is untested at their scale. All compute is one T4 except the 7B point, which needed
4-bit quantisation to fit.

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
