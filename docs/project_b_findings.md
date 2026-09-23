# Project B results

1. **Does the with/without-context KL carry a usable importance signal?** Yes. The span gate
   selects required content at **+6.77σ** over a matched-budget control.
2. **Does gating distillation on it beat uniform distillation?** **No.** With that verified
   gate, KL-gated distillation reaches 0.510 against uniform's 0.708, and does not beat a
   random-span control either.

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
the gated adapter still trains worse than one fed random spans. **Coverage of the right tokens
is not what limits distillation here.** The leading explanation is `floor_weight: 0.0`: every
non-selected token gets zero gradient, so the low-surprise prefix that conditions each
high-surprise identifier is never trained, and the path to the knowledge breaks even though
the knowledge was selected. Random spans break that path less systematically, and uniform
does not break it at all. This is the same shape as Project A's failure
([project_a_findings.md](project_a_findings.md)).

**Recorded prediction, now scored.** Before this run we predicted gated distillation would
land between the random control and uniform. It landed below the random control. The
coverage account alone does not explain that; the gradient-path account does.

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

- `floor_weight` ablation (e.g. 0.1 and 0.3) at the same gate: the direct test of the
  gradient-path explanation. If a small floor lifts gated distillation to uniform, gating
  selects correctly and only needs conditioning context.
- Token-granularity arm: `scripts/run_skills.py --granularity token --stages
  score,distill,eval,report`, wired into [b1_skills.ipynb](../notebooks/b1_skills.ipynb) via
  `RUN_TOKEN_ARM`.

See [paper/draft_combined.md](../paper/draft_combined.md).
