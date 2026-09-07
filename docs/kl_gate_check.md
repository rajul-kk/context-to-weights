# KL gate sanity check

> **Superseded.** This check ran on a two-skill debug slice with SmolLM2-360M and reported
> `required_in_top_frac = 0.50` with no control to compare it against. On the full eight-skill
> set with Qwen2.5-1.5B the same measurement is 0.118 against a matched-budget random control
> of 0.307 — 5.14σ *below* chance. The interpretation section below rationalises a number that
> should have stopped the run. Kept as written, because that is the finding. See
> [project_b_findings.md](project_b_findings.md).

The dual-forward-pass importance signal is the riskiest part of Project B, so it is verified
by hand before anything trains on it. This is the check, run on SmolLM2-360M-Instruct over
two toy skills (18 demonstrations, 1050 response tokens).

```bash
python kl_gate/score.py --config configs/skill_cpu_debug.yaml \
  --skills veltrix-cache pallas-http --out artifacts/runs_skill/scores_debug.jsonl
python kl_gate/inspect_gate.py --scores artifacts/runs_skill/scores_debug.jsonl
```

## Distribution

| | |
|---|---|
| median token KL | 0.840 |
| mean token KL | 2.078 |
| p90 token KL | 6.303 |
| required tokens inside top-25% of spans | 0.50 |

The distribution is heavily right-skewed, which is what a useful gate needs: most tokens are
predictable without the skill doc, a small tail is not.

## Highest-KL spans

```
3.334  [pallas-http]    'The rule is that deadline_ms defaults to nothing and must be set explicitly...'
3.216  [pallas-http]    'deadline_ms defaults to nothing and must be set explicitly; retries require idem_key.'
2.810  [veltrix-cache]  'ttl_s is mandatory and must not exceed 86400; Veltrix rejects a write without it.'
2.715  [pallas-http]    'pallas_probe("internal", body=body, deadline_ms=250)'
```

## Lowest-KL spans

```
0.165  '```python'
0.154  '```\n\nNote:'
0.126  '```\n\nNote:'
```

## Token level

```
[veltrix-cache write] 'st':8.25  'V':7.50  'rejects':7.45  'x':7.20  'write':5.90  'without':5.35
[veltrix-cache read]  'rejects':7.94  'V':7.17  'fetch':7.00  'x':6.99  's':6.01
[veltrix-cache rule]  'X':8.86  'rejects':7.53  'elt':6.97  'write':6.49  'TT':6.37
```

The subword pieces of `vx_stash`, `vx_fetch` and `VX_TTL_MISSING` are the highest-scoring
tokens in their demonstrations, alongside the words carrying the API's rule. Markdown
scaffolding scores an order of magnitude lower.

## Interpretation

The gate fires on API identifiers, error codes and rule clauses, and stays quiet on
formatting and boilerplate. This is the behaviour the method assumes.

`required_in_top_frac = 0.50` looks low until you look at what misses. The required strings
for write/read tasks include the first positional argument value (`"batch"`, `"internal"`),
which appears in the *query*. The student model can already predict it without the skill
doc, so its KL is correctly low. That value is not skill knowledge, and the gate is right to
skip it. The API function names and error codes — the parts that are skill knowledge — land
in the top band consistently.

This distinction is worth stating in the paper: the gate scores *what the skill document
adds*, not *what the answer contains*.
