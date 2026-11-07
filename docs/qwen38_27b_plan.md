# Qwen3.8 27B plan

## Why this model

It tops the sub-40B open-weight band on Artificial Analysis and sits at the scale where the
paper we position against reports that a stated region declaration works (27B and 31B). Our
routing table stops at 7B. One more point on the same 384 probes shows whether `generate`
closes the gap to `read` at 27B or not.

## What the model card changes

| Fact | Consequence |
|---|---|
| 64 layers: 48 Gated DeltaNet, 16 full attention (24 query heads, 4 KV heads, head dim 256) | the attention-probing modes assume attention in every layer and are not run |
| thinking is on by default | `enable_thinking: false` through `model.chat_template_kwargs`, or the next token after the prompt starts a `<think>` block and the Yes/No read sits on the wrong position |
| vision-language checkpoint | `model.loader` chooses causal or image-text, picked by the preflight from the config class |
| about 14 to 16 GB at nf4 | does not fit one T4, so `model.device_map: auto` with a per-GPU `max_memory` over two T4s |
| bf16 weights, T4 has no bf16 | fp16 compute, with a finite-logits check since fp16 can overflow on some models |
| 262k native context | the 7k-token `generate` prompt fits |

## Runs

- Modes: `read` and `generate`, 384 probes, seeds 0 to 2, layouts pinned to
  `results/declare/gold/gold384_s<seed>.json`.
- Layouts: regions are built with the model's own tokenizer, so under the Qwen3.8 tokenizer
  a probe or two lands in a different region than under Qwen2.5 (one in 384 at seed 0, with
  region 2 at 37 against 38). The 27B therefore keeps its own gold reference
  `gold384_qwen38_s<seed>.json`, written on the first run. `read` and `generate` share layouts
  within the run; the match to the other scales is exact except for those probes.
- Size: the first session measured 125 s per probe for both modes, 13.4 hours for 384. Each
  session runs the first 96 layouts (`--take 96`, about 3.5 hours), so a seed is a prefix and
  the test is directional rather than the 384-probe power of the other scales.
- Notebook: `notebooks/c3_qwen38_27b.ipynb`, config `configs/kaggle_declare_27b.yaml`.
- Stages: preflight, a timing run that stops if the projection exceeds `MAX_HOURS`, the grid,
  then a report merged with the committed 0.5B, 1.5B and 7B summaries.

## Results (seeds 0 to 2, first 96 layouts)

Summaries and gold references are in `results/declare_27b/`.

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| `read` hit rate | 0.365 | 0.365 | 0.365 | 0.365 |
| `generate` hit rate | 0.396 | 0.354 | 0.396 | 0.382 |
| `read` content dependence | 0.229 | 0.250 | 0.240 | 0.240 |
| `generate` content dependence | 0.250 | 0.219 | 0.208 | 0.226 |

Chance is 0.125 and the best constant control 0.156 to 0.198. The paired difference in hit rate
(`read` minus `generate`) is -0.017 with a standard deviation of 0.024 across seeds (t(2) =
-1.25), and the content dependence difference is +0.014 (t(2) = 0.80). On the probes where
exactly one mode was right, `read` won 33 and `generate` won 38 over the three seeds. The modes
are tied at 27B, against a `read` lead of 0.19 at 7B.

The seeds reshuffle only the layout of the same 96 probes, so they are not independent probe
samples, and `read` scores each region on its own, which is why its hit rate is identical across
seeds. The t(2) values understate the uncertainty from probe sampling, and the result is
directional. The run was 4-bit with thinking disabled and took 3.2 hours per seed.

## Outcomes

| Result | Reading |
|---|---|
| `read` still wins significantly | measured beats asked holds through 27B, which conflicts with the positioning paper and needs a careful explanation |
| `generate` ties or wins | stated routing emerges between 7B and 27B, which refines "measured, not asked" into a scale statement |
| `read` collapses | check the thinking and fp16 guards before believing it |

## Unknowns

- Nothing about loading, kernels or speed has been tested. The DeltaNet layers need either
  flash-linear-attention or a slow torch fallback, and a T4 may not run the fast kernels.
- The wall-clock budget is unmeasured. The timing stage exists to measure it before the grid.
- `check_probe_parity` reads the config attention implementation from the top-level config,
  which may differ for a nested vision-language config. A failure there shows up in the
  preflight, not later.
- The hybrid cache means KV quantization applies to only 16 of 64 layers. The KV gate
  (`kvquant_gate.md`) is not planned at this scale until the routing run says whether the model
  is worth more compute.
- A G1-style confidence test at 27B would likely saturate on the current task families and
  needs harder tasks first.
