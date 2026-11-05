# KV quantization gate

## Question

Does the `read` relevance signal from Project C survive when keys and values are stored at low
precision? Mixed-precision KV methods assign bits by an importance signal (MiKV, ZipCache and
related work). A signal comparison against a matched random assignment only makes sense if
routing degrades somewhere between full precision and 2 bits.

## Rule

Per model and scheme, with the baseline paired by seed (two-sided paired t at 5%):

| Outcome | Reading | Next step |
|---|---|---|
| no significant drop down to 2 bits | nothing for a signal to guide | stop |
| significant drop already at 8 bits | quantizer too harsh or routing too fragile | fix the quantizer once, then stop |
| holds at some width, drops at a lower one | cliff, room for guided assignment | design the assignment experiment |

The baseline must reproduce the committed `results/declare` hit rate for the same model and
seed to within 0.01, and clear 2 sigma over the best constant control.

## Design

- Runner: `declare/run.py --modes read --kv-bits B --kv-scheme S`.
- Quantization is simulated inside attention (`declare/elicit.py`), so it measures accuracy loss
  only.
- Schemes: `token` (per token, groups of 32 over the head dimension) and `channel_keys` (keys per
  channel over groups of 32 tokens, trailing group kept at full precision, padding excluded from
  each group's range, values per token).
- Bits: 8, 6, 4, 3, 2 against a full-precision baseline.
- Models: Qwen2.5-0.5B and 1.5B. Probes: the 384 behind Table 2, layouts pinned by
  `results/declare/gold/gold384_s<seed>.json`.
- Notebook: `notebooks/e1_kvquant.ipynb`, one seed per session. Report:
  `python -m eval.kvquant_report --run-root <runs>`.

## Pilot

A CPU pilot at Qwen2.5-0.5B used only 16 probes and the per-token scheme. Hit rate was 0.25
at full precision, 0.31 at 8 bits and 0.06 at 4, 3 and 2 bits against 0.125 chance. The sample
and the baseline are too weak to conclude anything, which is why the full 384-probe grid with a
second scheme is the gate.

## Scoop check

Importance-guided mixed-precision KV quantization is established. A matched-random control for
bit assignment and a multi-signal comparison were not found in a shallow search of abstracts, so
any follow-up claims only that comparison. Read the SemKV and MixKVQ ablations in full before
spending compute on it.
