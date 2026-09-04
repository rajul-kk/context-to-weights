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

## First measurements

SmolLM2-360M-Instruct, HotpotQA trajectories, `K = 8`. Small n; recorded because the failure
mode is unambiguous, not because the numbers are conclusive.

**Unbalanced layout, n = 12** — gold confined to regions 0-3, best constant 0.333:

| | generate | read |
|---|---|---|
| hit rate | 0.333 | 0.333 |
| best constant | 0.333 | 0.333 |
| slot stable rate | **1.000** | 0.000 |
| modal share | 0.917 | 0.333 |

Neither mode beats the constant policy, and the two diagnostics separate them cleanly.
`generate` answered `FOCUS: 0` on essentially every probe and every shuffle — the same
degenerate behaviour as the index-list compactor, reproduced in a second task with a
different prompt format. `read` varies its scores per region and changes its choice under
shuffling, so it is responding to content; it is simply not accurate at 360M.

**Balanced layout, n = 24** — TBD, see `artifacts/runs_declare/smollm_bal`.

## What would make this a result

- n in the hundreds, not six
- 0.5B, 1.5B, and 7B-4bit, to locate the threshold where generated declarations start working
- `read` clearing `best_constant_control` at some scale
- attended-token fraction compared against the paper's 52.0% / 31.1%, so the saving is
  comparable

If generated declarations turn out to work fine at 1.5B, this arm becomes a negative result
and the combined paper leans on the compaction and context-gap signals instead.
