# Results

Backbone `HuggingFaceTB/SmolLM2-360M-Instruct`, compactor `heuristic`, keep_frac 0.25, context budget 512.

## Retention

| method | acc | acc(evicted) | n_ev | medCE(ev) | medCE | meanCE | tokens |
|---|---|---|---|---|---|---|---|
| no context (floor) | 0.000 | 0.000 | 18 | 3.549 | 3.549 | 4.671 | 44.000 |
| (c) cascading, no update | 0.944 | nan | 0 | nan | 0.008 | 1.443 | 480.667 |
| (a) uniform replay | 1.000 | nan | 0 | nan | 0.010 | 1.468 | 480.667 |
| (b) reflection | 0.944 | nan | 0 | nan | 0.007 | 1.541 | 480.667 |
| ours: compaction-supervised | 0.889 | nan | 0 | nan | 0.008 | 1.379 | 480.667 |
| ours + mask head | 1.000 | nan | 0 | nan | 0.008 | 1.420 | 480.667 |
| (d) full context (ceiling) | 0.667 | nan | 0 | nan | 1.486 | 4.570 | 989.667 |

> **Configuration is degenerate.** The compactor evicted no probed fact, so `acc(evicted)` is undefined and this table cannot separate the methods. Lower `compaction.keep_frac` or lengthen trajectories and rerun.

> **Ceiling inverted.** The full-context arm scores below cascading, so it is not acting as a ceiling. The backbone is too small to use the long context; scale up before reading (d) as an upper bound.

## Consolidation cost

| method | phases | total GPU-s | GPU-s/phase |
|---|---|---|---|
| ours: compaction-supervised | 1 | 458.2 | 458.2 |
| ours + mask head | 1 | 349.1 | 349.1 |
| (a) uniform replay | 1 | 472.5 | 472.5 |
| (b) reflection | 1 | 1182.3 | 1182.3 |

## Figures

![headline](../artifacts/runs/report/figures/headline_retention.png)

![ce](../artifacts/runs/report/figures/ce_curves.png)
