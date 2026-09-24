# Project B results

1. **Does the with/without-context KL carry a usable importance signal?** Yes. The span gate
   selects required content at **+6.77σ** over a matched-budget control.
2. **Does gating distillation on it beat uniform distillation?** **No, and it doesn't beat
   random selection either.** At `floor_weight: 0` it loses to both (0.510 vs 0.583 and
   0.708). Fixing the floor to 0.1 lifts it above 0.65 on every seed, but paired against
   `random` — the arm with identical masking and no signal at all — the KL ranking adds
   nothing (5 seeds, t(4)=0.74). The gate selects the right tokens (+6.77σ at selection) and
   that selection does not translate into better training.

## The run

`Qwen2.5-1.5B-Instruct`, 8 toy skills in 3 categories, 300 steps per group,
`gate.granularity: span`, `top_frac: 0.25`, `floor_weight: 0.0`. Batch job on a Kaggle T4,
2026-09-24, source `c37bcd7`, fresh scoring and no restored artifacts.

| method | in-group pass (n=96) |
|---|---|
| (a) full skill text in prompt (ceiling) | 0.708 |
| (b) S2L-style uniform distillation | **0.708** |
| (d) random-span control | 0.583 |
| ours: KL-gated distillation | **0.510** |
| (c) no skill (floor) | 0.000 |

| comparison | difference | margin |
|---|---|---|
| ours vs uniform | -0.198 | about -2.9σ |
| ours vs random-span control | -0.073 | about -1.0σ, not significant |
| uniform vs random-span control | +0.125 | about +1.8σ |

**Uniform distillation matches the full-prompt ceiling** at 69% fewer runtime tokens
(201 -> 62), a clean replication of S2L. An internalised skill also keeps partial function
under a mismatched retrieved document (**0.292 vs 0.000** for the prompted skill).

These numbers reproduce an earlier run exactly (0.510 / 0.708 / 0.583), so that run also used
a correctly scored gate.

## The gate works; training on it does not

`inspect_gate.py` measures **required-token coverage** — the fraction of each demo's
`required` API identifiers inside the gate's selection — against a **matched-budget random
control** over 20 permutations.

| score file | coverage | control | margin | verdict |
|---|---|---|---|---|
| **batch run** (2026-09-24, transformers 5.0) | **0.597** | 0.392 | **+6.77σ** | clears control |
| earlier Kaggle scoring | 0.586 | 0.392 | +6.40σ | clears control |
| stale local file (older tokenizer, 3,871 tokens) | 0.118 | 0.307 | -5.14σ | retracted |

The stale file split identifiers such as `vx_stash` into five pieces, which drowned code spans
under span-mean pooling. That number is retracted; it never reflected the gate.

So selection is not the problem: the gate covers required content at 1.5x its control, and
the gated adapter still trains worse than one fed random spans.

## Floor-weight ablation

At `floor_weight: 0` every non-selected token gets zero gradient, so the low-surprise prefix
that conditions each selected identifier is never trained. Same gate (+6.77σ), same 300
steps, non-selected tokens given a small weight instead (batch job, 2026-09-24 to 2026-09-25,
5 seeds).

**`uniform` and `random` are not the same kind of arm.** In `distill/gate.py`, `uniform` gives
every token weight 1.0 — full-weight training on everything, no subsetting at all.
`random`, like `kl_top`, gives weight 1.0 to a 25% subset and the floor weight to the rest;
the only difference between `random` and `kl_top` is *which* 25% is chosen. So `kl_top` vs
`random` isolates whether the KL ranking matters; `random` or `kl_top` vs `uniform` compares
masked training against unmasked training, a different question entirely.

| floor weight | seed | KL-gated | random-span | uniform |
|---|---|---|---|---|
| 0 | 0 | 0.510 | 0.583 | 0.708 |
| 0.1 | 0 | 0.729 | 0.729 | 0.708 |
| 0.1 | 1 | 0.823 | 0.750 | 0.729 |
| 0.1 | 2 | 0.802 | 0.719 | 0.698 |
| 0.1 | 3 | 0.813 | 0.792 | 0.729 |
| 0.1 | 4 | 0.656 | 0.729 | 0.677 |
| 0.3 | 0 | 0.708 | 0.656 | 0.708 |

In-group pass, n=96 each. Paired across all five floor-0.1 seeds:

| comparison | mean diff | t(4) | crit | significant |
|---|---|---|---|---|
| kl_top - random (isolates the KL ranking) | +0.021 | 0.74 | 2.78 | no |
| kl_top - uniform (masked vs unmasked) | +0.056 | 2.33 | 2.78 | no |
| random - uniform (masked vs unmasked) | +0.035 | 3.90 | 2.78 | **yes** |

**The zero floor was still the defect** — floor 0.1 never scores below 0.656, against 0.510
at floor 0. That holds.

**The KL ranking itself adds nothing over random selection.** Seeds 0-3 put the gate ahead of
random by 0 to 0.08; seed 4 reverses it, gate below random by 0.07. Paired across all five,
+0.021, t(4)=0.74 — not remotely significant, and the sign is not even consistent. The
"nominal lead" read from three seeds does not survive a fourth and fifth. Selecting by KL rank
is indistinguishable from selecting at random, at this scale and step budget.

**A masked 25% at floor 0.1 tends to beat unmasked full-weight training, and this part is
real.** `random - uniform` is positive on all five seeds (0.021 to 0.062) with small variance,
significant at t(4)=3.90. `kl_top - uniform` points the same way but seed 4's reversal keeps
it short of significance (t=2.33). This is not a claim about selection quality — `random`
picks its 25% with no signal at all — it is a claim that training on a smaller, full-weight
subset plus a low-weight remainder outperforms training on everything at full weight, at 300
steps. Plausibly a fixed-step-budget effect: masked training concentrates gradient on fewer
tokens per step. Untested whether it survives a longer step budget or a learning-rate sweep
for the uniform arm; report it as a real but narrow finding, not a general claim about masking.

**Recorded prediction, scored, twice.** We predicted gated distillation would land between
random and uniform. At floor 0 it landed below both. At floor 0.1 the correct comparison
(against random) shows no gap at all — the prediction of an intermediate position was wrong
both times; there is no ordering between kl_top and random to be intermediate to.

## Methodological record

- **The first gate check had no control.** `gate_report.json` recorded
  `required_in_top_frac: 0.125` beside `top_frac: 0.25` and nothing compared them.
  `inspect_gate.py` now reports a σ margin against a matched-budget control.
- **A cached score file is only valid in the environment that produced it.** The same gate
  read -5.14σ from a stale local file and +6.77σ when re-scored, with no code change.
- **We briefly withdrew a correct result.** Seeing the stale inspection number, we assumed the
  downstream run had trained on it too and marked Project B's comparison confounded. Re-running
  reproduced it exactly. A retraction needs the same evidence as a claim.

## Next

- Whether masked-at-floor-0.1 beats unmasked `uniform` in general, or only at 300 steps: a
  longer step budget or an LR sweep for `uniform` would tell whether that's a real training
  effect or an artifact of a step count picked for `kl_top`, not for it.
- Token-granularity arm, run and pending write-up: see below.

See [paper/draft_combined.md](../paper/draft_combined.md).
