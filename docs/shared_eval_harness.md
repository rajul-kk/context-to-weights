# Shared evaluation harness for Myrios and skill-library

Status: design, version 0.1.0 of the `myrios_eval` package. Scope is item 5 of the 5 Nov 2026
design plans (Plan B): give the planned library-size sweep in `skill-library` (10 to 200 skills,
in-context vs cumulative LoRA vs isolated adapters, 1B and 3B, 20 seeds) the same metric
definitions, statistics and result file as Myrios, without changing any existing Myrios number.

Paths written `skill-library/...` refer to `D:\Work\skill-library` at commit `906d77a` (branch
`feat/library-size-sweep`). That repo was read, not edited.

## 1. Duplication inventory

Each row is one quantity that both repos compute, or that Myrios computes more than once.
"Same" means the arithmetic agrees on non-empty input; "differs" lists what does not.

| quantity | Myrios | skill-library | agreement |
|---|---|---|---|
| success / pass rate over items | `eval/skill_eval.py:65-80` (`summarize.rate`, nan on empty); `eval/metrics.py:36-56` (`retention_accuracy`, 0.0 on empty); `eval/abstention.py:24`; `eval/scoring_retention.py:49`; `selftrain/gate.py:99` | `skill-library/analysis/metrics.py:37-39` (`Snapshot.success_rate`, mean of per-skill rates, 0.0 on empty); per-skill rate is mean over repeats at `:97` | same mean; empty-input value differs (nan vs 0.0) |
| mean prompt tokens | `eval/skill_eval.py:78-79`, `eval/metrics.py:51`; token count is `len(tokenizer.encode(chat_text(...), add_special_tokens=False))` at `eval/skill_eval.py:59` and `eval/retention.py:405`, i.e. the chat-templated prompt including the system message | `skill-library/analysis/metrics.py:41-43` and `:163-168`; count from `skill-library/library/retrieval.py:16-25` (chars/4 heuristic on CPU, `add_special_tokens=False` HF count on GPU) over the raw prompt without chat template, summed over the retries inside one task attempt (`skill-library/agent/loop.py:71`) | mean is the same; **what is counted differs** (template, retries, heuristic) |
| aggregate over seeds (mean, sd) | three copies: `eval/declare_report.py:32-47` (`group`), `eval/kvquant_report.py:33-34` (`mean_sd`), `paper/figures/make_figures.py:34-35` (`mean_sd`); all sample sd (n-1), 0.0 for one seed | `skill-library/baselines/runner.py:182-205` pools snapshots of all seeds; `final()` at `:55-65` and again in `skill-library/analysis/collect.py:250-257`; no sd anywhere | **differs**: see 1.1 |
| paired t across seeds or corpora | `common/stats.py:27-34` (`paired_t`), critical values `:3-8, 21-24`, verdict `:11-18` | none | n/a |
| binomial standard error | `eval/report.py:103-106` (`_se`) | none | n/a |
| multiple-comparison correction | `eval/significance.py:69-78` (Holm) | none | n/a |
| retention of old skills, forgetting | none (Myrios measures retention of evicted facts, `eval/metrics.py:58-66`) | `skill-library/analysis/metrics.py:45-48` (`earliest_k`), `:111-129` (forgetting curve), `:171-196` (normalised curve) | n/a, but the sweep needs a scalar forgetting number and has none |
| stored parameters | not reported | `skill-library/sleep/backend.py:44`, `skill-library/sleep/lora.py:183-188` (`numel` of trainable params on GPU), hard-coded `LORA_PARAMS_R16 = 4_718_592` in `skill-library/baselines/methods.py:13` on CPU; per-seed mean cast to int at `skill-library/baselines/runner.py:77` | n/a |
| token saving vs in-context | `eval/skill_report.py:188-197` | prose only (README "roughly in half") | n/a |
| FLOPs | none (wall-clock GPU seconds at `eval/report.py:69-82`) | none | missing in both |
| bootstrap CIs | none | none | missing in both |
| headline text table | `eval/report.py:85-100`, `eval/skill_report.py:171-185` | `skill-library/analysis/collect.py:289-302` and a copy at `skill-library/scripts/reproduce.py:45-59` | formatting only, out of scope |

### 1.1 A definitional difference found while building the inventory

`skill-library` defines a method's final score as the mean over snapshots taken at the
**global** last night across all seeds (`skill-library/baselines/runner.py:59-60`). A seed whose run
ended one night earlier is silently dropped. In `skill-library/results/cpu/results.json`:

| method | seeds in run | seeds in `final_success_rate` | `final_success_rate` | mean of `per_seed_final` |
|---|---|---|---|---|
| in_context | 20 | 20 | 0.853 | 0.853 |
| seq_no_replay | 20 | 15 | 0.098 | 0.107 |
| isolated | 20 | 19 | 0.733 | 0.740 |
| single_shot | 20 | 20 | 0.707 | 0.707 |
| ours | 20 | 19 | 0.733 | 0.740 |

Myrios always reduces each seed to one number and then aggregates across seeds. The shared
harness follows Myrios (`final_per_seed`), and offers the old rule as `final_global_last_night`
only so that a golden test can reproduce the published `skill-library` numbers. With 200-skill
libraries, seeds will finish at different nights far more often, so the sweep should use the
per-seed rule. This changes `skill-library`'s headline numbers by up to 0.009 and is the owner's
call; nothing here changes them.

### 1.2 Other things the owner should check

- `LORA_PARAMS_R16 = 4_718_592` equals four 1536x1536 projections over 24 layers at rank 16.
  Qwen2.5-1.5B has 28 layers and grouped-query attention (k and v are 1536x256), so the
  constant does not match the GPU model. `lora_params` in the harness computes it from shapes.
- On CPU the token column is a chars/4 estimate. The sweep's matched-token comparison needs a
  real tokenizer count, recorded with which tokenizer produced it.

## 2. Architecture

A pure-stdlib Python package `myrios_eval` living inside Myrios at
`packages/myrios-eval/src/myrios_eval`, with its own `pyproject.toml`, installable on its own:

```
pip install -e D:/Work/Myrios/packages/myrios-eval
pip install "myrios-eval @ git+file:///D:/Work/Myrios@<commit>#subdirectory=packages/myrios-eval"
```

| module | contents |
|---|---|
| `metrics` | `success_rate`, `mean_prompt_tokens`, `token_saving`, `final_per_seed`, `final_global_last_night`, `earliest_k`, `backward_transfer`, `forgetting` |
| `stats` | `mean_sd`, `paired_t`, `critical`, `verdict_for`, `binomial_se`, `holm`, `bootstrap_ci`, `paired_bootstrap_ci`, `aggregate_seeds` |
| `cost` | `lora_params`, `inference_flops`, `lora_train_flops`, `full_train_flops` |
| `schema` | `SCHEMA_VERSION`, `make_result`, `make_arm`, `validate`, `write_result`, `read_result` |
| `adapters` | `from_myrios_skill_summary`, `from_skill_library_results` (read dicts, import nothing from either repo) |

Myrios's existing modules are **not** changed to import the package. The package sits next to
them, and golden tests pin each shared function to the existing Myrios function and to saved
numbers, so the two cannot drift apart unnoticed (section 6).

## 3. Decisions

### 3.1 Where the shared code lives

| option | verdict | reason |
|---|---|---|
| **pip-installable subpackage inside Myrios, own `pyproject.toml` in a subdirectory** | chosen | one source of truth next to the code it must agree with; golden tests run in Myrios CI against Myrios artifacts; installs alone, so it does not drag Myrios's `sleep`, `baselines` or `eval` packages into skill-library |
| install the whole Myrios repo into skill-library | rejected | both repos have top-level `sleep` and `baselines` packages; installing one into the other's environment shadows modules |
| `common/eval` inside Myrios's `common` package | rejected | importing it from skill-library would put Myrios's `common` on the path; the name is generic and collides easily |
| standalone new repo | rejected | a third repo for about 400 lines; the golden data lives in Myrios, so tests would need a cross-repo checkout |
| git submodule in skill-library | rejected | submodules complicate the Kaggle source zips both repos use, and pin a whole repo when one folder is needed |
| copied module | rejected as the default, kept as fallback | copies drift; the fallback for offline Kaggle runs is to copy the single `myrios_eval/` folder (stdlib only) and record its `__version__` in each result file, which `validate` checks |

### 3.2 Refactor existing Myrios code or not

Not in this change. The hard constraint is no behaviour change to Myrios results. The shared
functions are proven equal to the old ones by tests, which would allow a later one-line delegation
(for example `common/stats.py` re-exporting `paired_t`), but that would add an import dependency to
every Kaggle notebook source zip (`scripts/package_source.py` lists the directories it ships) for
no change in output. Left for the Myrios owner; the golden tests already guarantee agreement.

### 3.3 Dependencies

Standard library only (`statistics`, `random`, `json`, `math`). Myrios's tests run without torch,
skill-library's CPU pipeline has no numpy in `requirements.txt`, and Kaggle images vary. Rejected:
numpy/scipy for bootstrap and t (would make results depend on library versions; the t critical
table is already hard-coded in Myrios and is copied verbatim).

## 4. Metric definitions

All rates are fractions in [0, 1]. Empty input returns nan for every mean-type metric, so an
empty arm can never pass as a zero. This is the one intended difference from old code that
returned 0.0 (`eval/metrics.py:46`, `skill-library/analysis/metrics.py:39`); golden tests cover
non-empty input only.

| metric | definition |
|---|---|
| `success_rate` | mean of per-item outcomes (bool or float in [0,1]). For skill-library, an item is one skill and its outcome is the mean over eval repeats, as now. |
| `prompt_tokens` | mean tokens of the model input per generation call. The result file must say how it was counted: `token_counter` (`hf:<tokenizer id>` or `heuristic:chars/4`) and `prompt_tokens_scope` (`chat_template` or `raw_prompt`; plus `per_call` or `per_task_attempt`). Myrios is `chat_template`/`per_call`; skill-library today is `raw_prompt`/`per_task_attempt`. Numbers with different scopes must not be compared. |
| `token_saving` | `1 - tokens(arm) / tokens(in-context arm)`, as `eval/skill_report.py:196`. |
| final score per seed | `final_per_seed`: for each seed, the value at that seed's own last snapshot; then aggregate across seeds. |
| `earliest_k` | mean success over the k skills learned earliest (sort by skills learned since, descending, then tier), k=3, as `skill-library/analysis/metrics.py:45-48`. |
| `backward_transfer` | Lopez-Paz and Ranzato (2017): for each skill, final score minus score at the first snapshot after it was learned, averaged over skills learned before the final snapshot. Negative means forgetting. |
| `forgetting` | Chaudhry et al. (2018): for each such skill, best earlier score minus final score, averaged. Non-negative. |
| `stored_params` | adapter parameters that must be kept to serve every skill: for one shared adapter, its size; for isolated adapters, the sum. `lora_params(shapes, rank)` = sum of `rank * (d_in + d_out)` over adapted matrices. |
| `inference_flops` | `2 * N * tokens` (forward pass, N = model parameters), the usual dense-transformer estimate; the attention term quadratic in length is ignored and the file says so via `flops_method`. |
| `lora_train_flops` | `4 * N * tokens + 6 * N_lora * tokens`: forward (2N) and the activation-gradient half of the backward pass (2N) through the frozen base, plus full training cost on adapter weights. `full_train_flops` = `6 * N * tokens`. |
| seed aggregate | `aggregate_seeds(values)`: `mean`, `sd` (sample, n-1; 0.0 for one seed, as Myrios), `n`, and a percentile bootstrap 95% CI over seeds (10,000 resamples, fixed RNG seed 0, `random.Random`). |
| arm comparison | `paired_t` over seeds (identical to `common/stats.py`), with `critical(n)` and `verdict`; plus `paired_bootstrap_ci` on the per-seed differences. Paired t stays the headline because the Myrios paper reports it; the bootstrap CI is added because the sweep has 20 seeds, where it is informative. |
| several comparisons | `holm`, identical to `eval/significance.py:69-78`. |

## 5. Result schema (version 1.0)

One JSON file per experiment. Both repos write it; neither has to give up its own files.

```json
{
  "schema": "myrios_eval.result",
  "schema_version": "1.0",
  "harness_version": "0.1.0",
  "experiment": "skill-library/library-size-sweep",
  "repo": "skill-library",
  "commit": "906d77a",
  "created": "2026-11-05T12:00:00+00:00",
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "token_counter": "hf:Qwen/Qwen2.5-1.5B-Instruct",
  "prompt_tokens_scope": "raw_prompt/per_task_attempt",
  "flops_method": "2N per token inference, 4N+6N_lora per token LoRA training, attention term ignored",
  "arms": [
    {
      "arm": "ours",
      "label": "Cumulative LoRA + replay",
      "conditions": {"library_size": 50, "replay_cap": 12},
      "per_seed": [
        {"seed": 0, "n_items": 50, "metrics": {"success_rate": 0.74, "prompt_tokens": 412.0,
         "stored_params": 4718592, "backward_transfer": -0.05, "forgetting": 0.06,
         "train_flops": 1.2e15, "inference_flops": 1.2e12}}
      ],
      "aggregate": {"success_rate": {"mean": 0.74, "sd": 0.01, "n": 20,
                    "ci_low": 0.735, "ci_high": 0.745, "ci": "percentile bootstrap 95%, 10000 resamples, seed 0"}}
    }
  ],
  "comparisons": [
    {"a": "ours", "b": "isolated", "metric": "success_rate", "conditions": {"library_size": 50},
     "mean_diff": 0.0, "t": 0.0, "dof": 19, "critical": 2.093, "verdict": "indistinguishable",
     "ci_low": -0.01, "ci_high": 0.01}
  ],
  "extra": {}
}
```

Rules enforced by `validate`: required top-level keys present; `schema_version` major equals
the package's; every arm has a unique (`arm`, `conditions`) pair; every `per_seed` entry has an
integer `seed` that is unique within the arm and a `metrics` dict of numbers (nan written as
`null`); `aggregate` keys are a subset of the metric names present; `prompt_tokens` requires
`token_counter` and `prompt_tokens_scope`; any `*_flops` metric requires `flops_method`.
Repo-specific detail (curves, records, snapshots) goes under `extra` or stays in the repo's own
files.

## 6. Golden tests (equivalence proof)

Located in `tests/test_shared_eval_golden.py`. They read only saved numbers:

1. `paper/figures/data.json` (committed): shared `paired_t` reproduces t(4) = 3.10 and
   t(2) = 1.81 (consolidation), t(4) = 0.74, 3.90, 2.33 (distillation) and every routing t in the
   paper, and returns exactly (`==`) what `common.stats.paired_t` returns on every pair. Shared
   `mean_sd` reproduces every "mean ± sd" routing cell.
2. `results/declare/` (committed): shared `aggregate_seeds` gives exactly the `mean` and `_sd`
   of `eval/declare_report.py:group` for all 12 groups and every metric; shared `mean_sd` equals
   `eval/kvquant_report.py:mean_sd`.
3. `tests/fixtures/golden_skill_eval.json` (committed, extracted from the local
   `artifacts/runs_skill/report` CPU run: per-record `passed`, `in_group`, `prompt_tokens`, plus
   the saved summaries): shared `success_rate` and `mean_prompt_tokens` reproduce every saved
   `pass_rate`, `pass_rate_in_group`, `pass_rate_out_group` and `mean_prompt_tokens`, and equal
   `eval.skill_eval.summarize`. When the local artifacts exist the test also checks the fixture
   against them.
4. `skill-library/results/cpu/results.json` (read-only, skipped when absent; path from
   `SKILL_LIBRARY_ROOT`, default `../skill-library`): the adapter with
   `final_global_last_night` reproduces `final_success_rate`, `final_earliest_3`,
   `final_stored_params` and mean tokens for all five methods, and `final_per_seed` reproduces the
   mean of `per_seed_final`.
5. Unit tests for the new metrics (`backward_transfer`, `forgetting`, bootstrap, FLOPs, LoRA
   parameter counts, schema validation) in `tests/test_shared_eval.py`.

Anything that could not be proven equal was not refactored; per section 3.2 nothing was.

## 7. Versioning

- Package version `myrios_eval.__version__` (semver). Patch: bug fixes that do not change any
  golden number. Minor: new functions or optional schema fields. Major: a changed definition or
  required field.
- `SCHEMA_VERSION` is separate (`major.minor`). Readers accept the same major.
- Every result file records both, and the producing repo's commit.
- Release by tagging the Myrios commit as `myrios-eval-v<version>` (not done in this change; the
  Myrios owner tags when merging). skill-library pins a tag or commit, never a branch.

## 8. Migration plan for skill-library (for its owner to apply)

No change to skill-library is made here. Suggested steps, smallest first:

1. Add to `requirements.txt`:
   `myrios-eval @ git+file:///D:/Work/Myrios@<tag>#subdirectory=packages/myrios-eval`
   (or `-e ../Myrios/packages/myrios-eval` locally). For Kaggle, copy the `myrios_eval/` folder
   into the uploaded source zip.
2. After `write_results` in `skill-library/baselines/runner.py`, also write the shared file:
   ```python
   from myrios_eval.adapters import from_skill_library_results
   from myrios_eval.schema import write_result
   doc = from_skill_library_results(payload, experiment="skill-library/library-size-sweep",
                                    commit=commit, conditions={"library_size": n},
                                    token_counter="heuristic:chars/4",
                                    prompt_tokens_scope="raw_prompt/per_task_attempt")
   write_result(out / "result.myrios_eval.json", doc)
   ```
   The adapter uses the per-seed final rule (section 1.1), computes `backward_transfer`,
   `forgetting` and `earliest_3` per seed, and aggregates with bootstrap CIs.
3. For the sweep, one `make_arm` per (method, library size, model) and `paired_t` /
   `paired_bootstrap_ci` between methods at each size. Add `train_flops` from
   `lora_train_flops(N, tokens_trained, lora_params(...))` and `inference_flops` per attempt so
   matched-token and matched-FLOP comparisons read off the same file.
4. Decide whether to switch the README/paper headline to the per-seed final rule (changes
   0.098 to 0.107 and 0.733 to 0.740). If not, keep both and say which one the table uses.
5. Switch the GPU token count to the chat-templated prompt per call if numbers are to be put next
   to Myrios's; otherwise keep `raw_prompt/per_task_attempt` and do not compare token columns
   across repos.

## 9. Risks

- **Silent definition drift.** Mitigated by golden tests that compare against the old functions,
  not only against saved numbers.
- **Empty-input change** (nan instead of 0.0). Only affects arms with no items, which no saved
  result has; documented above.
- **Bootstrap with few seeds.** With 3 to 5 seeds a percentile bootstrap CI is too narrow; the
  Myrios paper therefore keeps paired t as the headline, and so should the sweep at small n.
- **FLOP estimates are approximations** (no attention term, no generation-length detail). The
  method string travels with every number.
- **Two token scopes.** Comparing Myrios and skill-library token columns directly would be
  wrong until step 5 of the migration is applied; `validate` forces the scope to be recorded.
- **Myrios is actively developed.** The package touches no existing module; the only shared
  file edited is `pyproject.toml` (pytest path, ruff), so merge conflicts are unlikely.

## 10. Citing "salient is not trainable" in skill-library

See [shared_eval_citation_note.md](shared_eval_citation_note.md).
