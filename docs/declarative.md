# Declarative attention below the scale where declaring works

Declarative Attention (arXiv:2609.02737) has a model state which region of its context it
needs to read — `<global>`, `<focus>`, `<local>` — and an inference engine skip most of the
KV-cache read accordingly. Reported on Gemma-4-31B and Qwen-3.6-27B: 52.0% and 31.1% fewer
attended tokens at 1.27pp and 2.75pp accuracy cost, zero-shot. The paper leaves
training-based extensions explicitly open.

This arm asks a prior question. The method assumes the model can *say* where to attend. We
already know that assumption fails for a structurally identical request: asked to select
spans by index, both Qwen2.5-0.5B and 1.5B answer `0, 1, 2, ...` on 100% of events regardless
of content (see [protocol.md](protocol.md)). If declarations fail the same way, the reported
token savings do not transfer to the scales a single-GPU researcher can run.

## Setup

A trajectory is split into `K` token-balanced regions, labelled inline. The model sees all of
them plus a question whose answer lies in exactly one. Two elicitations:

- **generate** — reply `FOCUS: <k>`, parsed like a tool call, as in the paper.
- **read** — for each region, ask whether it contains the answer and take
  `log p(Yes) - log p(No)` off the logits; choose the argmax. Costs `K` forward passes rather
  than one generation.

```bash
python declare/run.py --config configs/declare.yaml --modes generate read
```

## Controls

A hit rate on its own says nothing, for the same reason a salience lift on its own says
nothing.

Regions are **permuted per probe by default**, so the gold region is uniform across slots by
construction. Without that, gold clusters early — supporting content lands near the front of
a trajectory — and "always name region 0" scores 0.41 against a 1/K floor of 0.125, which is
enough to swallow any real effect. `--no-balance` reproduces the natural layout if you want
to measure the position prior itself.

| Control | What it catches |
|---|---|
| `random_control` = 1/K | the floor |
| `best_constant_control` | always naming one fixed region. Collapses to ~1/K once gold is balanced; report it anyway, since it is the number that exposed the unbalanced version |
| `slot_stable_rate` | permute which content sits in which slot and re-elicit. Near 1.0 means the model names a slot, not content |
| `content_dependence` | shuffled hit rate above chance — does the declaration follow the answer to its new slot? |
| `modal_share` | how often the same region is named |

`run.py` warns when a mode fails to beat the best constant policy, or when the declaration
barely moves under shuffling.

## Result

Qwen2.5-0.5B and 1.5B, HotpotQA trajectories with supporting paragraphs scattered across the
whole context so position carries no prior, `K = 8`, gold region permuted per probe, **384
probes**. Chance is 0.125, best-constant control 0.148.

Four elicitations on identical layouts: `generate` and `read` as above, plus `attention` and
`attention_late` (see the ablation section below). **Three seeds** (0, 1, 2), mean +/- sd
across seeds; the seeds reshuffle the layout permutation, not the probe set.

| model | mode | hit | σ(const) | content dep | slot stable | unparsed |
|---|---|---|---|---|---|---|
| 0.5B | generate | 0.193 +/- 0.012 | 2.2 +/- 0.4 | 0.085 +/- 0.021 | 0.281 +/- 0.053 | 0.019 |
| 0.5B | attention | 0.151 +/- 0.003 | 0.2 +/- 0.4 | 0.014 +/- 0.012 | 0.882 +/- 0.003 | 0.000 |
| 0.5B | attention_late | 0.234 +/- 0.011 | 4.0 +/- 0.8 | 0.113 +/- 0.029 | 0.356 +/- 0.045 | 0.000 |
| 0.5B | **read** | **0.332 +/- 0.004** | **7.7 +/- 0.5** | **0.203 +/- 0.003** | 0.136 +/- 0.017 | 0.000 |
| 1.5B | generate | 0.150 +/- 0.008 | 0.1 +/- 0.9 | 0.030 +/- 0.023 | 0.413 +/- 0.029 | 0.170 |
| 1.5B | attention | 0.286 +/- 0.023 | 6.0 +/- 0.7 | 0.163 +/- 0.010 | 0.447 +/- 0.005 | 0.000 |
| 1.5B | attention_late | 0.374 +/- 0.017 | 9.2 +/- 0.5 | 0.258 +/- 0.028 | 0.200 +/- 0.016 | 0.000 |
| 1.5B | **read** | **0.378 +/- 0.005** | **9.3 +/- 0.3** | **0.241 +/- 0.005** | 0.136 +/- 0.005 | 0.000 |

best-constant control 0.148 +/- 0.009, chance 0.125.

**`read` works at both scales, and it replicates across seeds tightly.** +7.7σ and +9.3σ
over the constant control, standard deviation under 0.005 on `content_dependence` at both
scales. When a region's content is shuffled to a new slot the choice follows it. This is a
content-tracking signal, essentially flat from 0.5B to 1.5B (0.203 to 0.241).

**`generate` does not track content at either scale.** At 0.5B the raw hit rate clears the
constant control (+2.8σ) — but `content_dependence` is only 0.055, a quarter of `read`'s, and
`slot_stable_rate` is 3x chance. The hit rate is mostly position. At 1.5B the generated
declaration is statistically indistinguishable from always naming one region (+0.3σ), the
model refuses the `FOCUS:` format on 19% of probes, and `slot_stable_rate` climbs to 0.42.

**Scale makes the generated declaration worse.** Every `generate` number degrades from 0.5B
to 1.5B: hit rate 0.206 → 0.154, σ over constant +2.8 → +0.3, unparsed 2% → 19%,
`slot_stable_rate` 0.33 → 0.42. The `read` numbers are flat to slightly better. The larger
model is worse at *saying* where to look and identical at *reading* it.

**The content-shuffle control is load-bearing.** At 0.5B, raw hit rate alone reports
`generate` as clearing the control at +2.8σ. Only the shuffle reveals the signal is
positional. This is the same lesson as the compaction positional control (§6) and the KL-gate
matched-budget control — the third independent instance in this project of a raw
signal-strength number that a matched control overturns.

**Token saving.** `read` attends 13.6% of the context — a larger cut than the paper's
52.0% / 31.1% — but at a 0.33–0.37 hit rate, roughly 3x chance. The routing signal is real
and cheap to extract at this scale; it is not accurate enough to act on without a check.

## Comparability

Every mode is evaluated on **identical layouts**. An earlier version drew fresh permutations
per mode, so `generate` and `read` saw different gold placements and their hit rates were not
comparable — visible in the run only because the two gold distributions printed differently.
Layouts are now materialised once and shared.

## Attention-probing ablation

The literature describes asking-the-model and probing-attention as the two routes to the same
answer - Declarative Attention [5] asks, Sentinel (arXiv:2505.23277) and the retrieval-head
line probe - but on different models and different data. This runs them head to head: same
layouts, same probes, same model.

`attention` reads the attention mass the final query position puts on each region and takes
the argmax. `attention_late` restricts that to the second half of the layers. The probe
registers a custom attention implementation that keeps SDPA for the forward pass and
materialises only the final query row, so cost is `O(heads x seq)` per layer rather than
`O(heads x seq^2)`; `run.py` refuses to start unless it reproduces SDPA on a padded batch.

### Result

Rows `attention` and `attention_late` in the table above. Margins below are a **paired t
across the three seeds**, not a binomial sigma, since the seeds share a probe set and only the
layout permutation is resampled; the t(2) critical value is 4.30.

| model | metric | read | attention_late | diff | t(2) | significant |
|---|---|---|---|---|---|---|
| 0.5B | hit rate | 0.332 | 0.234 | +0.099 | +24.9 | **yes** |
| 0.5B | content dependence | 0.203 | 0.113 | +0.090 | +4.91 | **yes** |
| 1.5B | hit rate | 0.378 | 0.374 | +0.003 | +0.30 | no |
| 1.5B | content dependence | 0.241 | 0.258 | -0.016 | -0.85 | no |

**Read wins clearly at 0.5B. At 1.5B it is a tie, not a crossover.** The single-seed run this
section originally reported had late-layer attention nominally ahead at 1.5B (0.391 vs 0.372);
with two more seeds the mean moves to 0.374 vs 0.378, a difference of 0.003 in hit rate, not
distinguishable from zero (t=0.30). Content dependence still points attention's way at 1.5B
(-0.016) but the paired t (-0.85) is far short of significant. Report 1.5B as level, not as
attention ahead.

**Layer choice decides whether attention probing works at all.** Averaged over every layer it
is useless at 0.5B - content dependence 0.003, naming the same slot on 88% of probes and the
same region on 94%. Restricted to the late half it clears its control at both scales. Early
layers, where a single Qwen2.5-1.5B layer-0 q-k product reaches 152,967, dominate the average
and carry position rather than content.

**Scale separates asking from measuring.** Content dependence from 0.5B to 1.5B, 3-seed means:

| elicitation | 0.5B | 1.5B | change |
|---|---|---|---|
| generate | 0.085 | 0.030 | **-0.055** |
| attention | 0.014 | 0.163 | +0.149 |
| attention_late | 0.113 | 0.258 | +0.145 |
| read | 0.203 | 0.241 | +0.038 |

Every measured signal improves with scale; the generated declaration is the only one that
degrades. Attention probing rises about four times faster than the self-query in absolute
terms, which is what closes the 0.5B gap by 1.5B — but closing a gap this size lands on a
tie, confirmed only in the negative (read's 0.5B lead is real; the 1.5B tie is also real, not
a reversal in progress). A third scale would show whether the trend continues past a tie into
an actual lead for attention, or plateaus.

Cost is comparable: `read` issues K region prompts in one batched forward, `attention` one
pass over the whole context, and both cover roughly the same number of tokens.

## Pilot runs, superseded

Earlier CPU runs on SmolLM2-360M (n = 12 and n = 24) showed `generate` naming a fixed slot
(`slot_stable_rate` up to 0.958) and `read` at chance. The direction matched the result above,
but the numbers are confounded: SmolLM2-360M has an 8192-token window and the rendered
contexts run to 8966 tokens, so the longest trajectories were silently truncated — under
right-side truncation, cutting the question off the end. `sleep/lm.py` now truncates
left-side and `declare/run.py` refuses to run when the context overflows the window. The
Qwen2.5 models have a 32768 window and are unaffected.

## Open

- A third scale (7B-4bit) would settle whether attention probing overtakes `read` past 1.5B,
  or plateaus at a tie. Three seeds at two scales rules out the 1.5B crossover being real; it
  does not rule out a real crossover further out.
- `read` at 27B+, to see whether it matches or beats the generated declaration [5] reports
  there. We cannot run it.
- The routing is 3x chance, not deployable unsupervised. A cheap verifier on the chosen
  region would close that, at the cost of the method's simplicity.
