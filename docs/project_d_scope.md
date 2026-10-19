# Project D scope: iterated self-training with measured signals

## Question

In a closed loop where a model trains on its own samples, does the choice of selection signal
matter, and does the loop improve or degrade accuracy, calibration and diversity over rounds?
Arms A to C found that free signals detect salience but do not beat random or uniform selection.
D tests that claim where a verifier exists and the loop is iterated.

## Hypotheses

| | Claim | Falsified if |
|---|---|---|
| H1 | Verifier filtering improves pass@1 for 3+ rounds, then plateaus | no gain over round 0 at t(4) 5% |
| H2 | Unfiltered self-training lowers diversity (distinct-n, pass@k) and calibration (ECE) each round | diversity and ECE stay within seed noise |
| H3 | Among verified-correct samples, selecting by measured confidence ties random selection | confidence selection beats random at t(4) 5% |
| H4 | A measured correctness signal (logit P(true)) ranks samples better than a stated one (asked yes/no, verbalised confidence) | AUROC of stated is not lower |
| H5 | An external teacher raises the ceiling above any self-only arm | teacher arm does not exceed best self arm |

H3 and H4 carry over the Measured-not-asked and Salient-not-trainable findings.

## Setup

- **Tasks:** verifiable and programmatic. Reuse the B toy skills (held-out pass rate already
  defined) plus modular arithmetic and string transforms with exact-match checkers. Held-out
  split never enters a loop.
- **Models:** Qwen2.5-0.5B and 1.5B Instruct. 7B nf4 only if the small scales show an effect.
- **Loop per round:** sample k=8 per prompt at T=0.8, label with the verifier, select by arm,
  LoRA SFT on the selection (adapter carried across rounds, as in `sleep/trainer.py`),
  evaluate. 5 rounds, 5 seeds.

## Arms

| Arm | Selection |
|---|---|
| verifier | all verified-correct samples |
| verifier+conf | verified-correct, top half by mean token log-prob |
| verifier+random | verified-correct, random half (size matched) |
| raw | all samples, no filter (collapse control) |
| self-judge | model asked "is this correct", keep yes |
| self-read | keep by logit P(yes) above the median |
| teacher | 7B samples that pass the verifier |

## Measurements per round

- held-out pass@1 and pass@8
- ECE and AUROC of token-confidence, verbalised confidence and logit P(yes) against the verifier
- distinct-2 and mean pairwise edit distance of samples (collapse)
- selection size, so arms stay matched

## Reuse and new code

- Reuse: `sleep/lm.py`, `sleep/trainer.py`, `common/stats.py` (`paired_t`), `common/io.py`
  overrides, `eval/declare_report.py` pattern for per-run summaries and a regenerated verdict.
- Built for G1: `selftrain/tasks.py` (five verifiable families), `selftrain/signals.py` (AUROC, ECE,
  confidence parsing, gate verdict), `selftrain/gate.py` (sampler and three readers),
  `selftrain/gate_report.py`, `configs/selftrain.yaml`, `tests/test_selftrain.py`.
- Still to build after G1: selectors, loop, `notebooks/d1`, the multi-round report.
- Run G1: `python selftrain/gate.py --n 400 --seeds 0 1 2` then
  `python -m selftrain.gate_report --run-root artifacts/selftrain`. Set `USE_TF=0` on Windows.
- Repo rules hold: no comments, tests for every verifier and selector, summaries committed to
  `results/selftrain/`, paper numbers tripwired from data.

## Gate before building the loop

Three stages, each cheap, each with a stop rule. Fail one and D does not proceed.

| Gate | Needs training | Measures | Pass | Cost |
|---|---|---|---|---|
| G1 | no | AUROC against the verifier of logit P(yes), token log-prob and verbalised confidence, on 400+ samples per task at 0.5B and 1.5B | measured beats stated by 0.05 AUROC with non-overlapping seed ranges, and measured AUROC above 0.6 | minutes on CPU at 0.5B |
| G2 | one LoRA round | round-0 pass@1 sits in 20 to 60 percent, and one verifier-filtered round raises held-out pass@1 | gain above seed noise at t(2) 5% on 3 seeds | about an hour on a T4 |
| G3 | one LoRA round per arm | conf-selected vs random-selected vs raw, size matched | any arm differs from random by more than the G2 gain's seed sd | about 3 hours on a T4 |

Reading the outcomes:
- G1 fails: no usable measured signal at this scale, so H3 and H4 are dead and D reduces to a replication of known self-training results. Stop.
- G2 fails: no headroom or no learnable gain, so adjust task difficulty once, then stop.
- G3 ties everywhere: the Salient-not-trainable finding extends to a verified loop. That is a reportable negative result, and the 5-round grid is unnecessary.
- G3 separates: build the full loop.

G1 needs no GPU, so it runs first and decides whether G2 is worth a Kaggle slot.

## Milestones

1. Tasks, verifiers and selectors with CPU tests (no GPU).
2. Smoke run at SmolLM2-360M, 2 rounds, 1 seed, to check the loop end to end.
3. 0.5B full grid: 7 arms x 5 seeds x 5 rounds on Kaggle (2 concurrent sessions).
4. 1.5B grid for the arms that matter after step 3.
5. Report, then a paper section or a standalone short paper.

## Cost (rough, unmeasured)

Sampling dominates. A 0.5B round of a few hundred prompts at k=8 should take minutes on a T4,
so the 0.5B grid is on the order of tens of GPU-hours and 1.5B roughly 3x that. Measure in
step 2 before committing.

## Risks

- Tasks too easy (ceiling) or too hard (no verified samples). Calibrate difficulty so round-0
  pass@1 is 20 to 60 percent.
- Small-model effects may not transfer to 7B+, which the paper must state.
- Seven arms with five seeds invites multiplicity. Pre-register H1 to H5 and apply the same
  Bonferroni caveat as before.
- Overlap with STaR, ReST-EM, self-rewarding and model-collapse work. The contribution is the
  matched-selection comparison and the measured-versus-stated ranking, not the loop itself, so
  run a scoop check before step 3.

## Decision needed

Whether D is a new standalone paper or a section of the current one. Recommendation: standalone,
since the current paper is already 12 pages and its claim is complete.
