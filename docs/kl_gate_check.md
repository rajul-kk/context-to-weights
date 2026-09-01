# KL gate sanity check

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
| median token KL | 1.016 |
| mean token KL | 2.135 |
| p90 token KL | 6.055 |
| required tokens inside top-25% of spans | 0.50 |

The distribution is heavily right-skewed, which is what a useful gate needs: most tokens are
predictable without the skill doc, a small tail is not.

## Highest-KL spans

```
3.354  [pallas-http]    'The rule is that deadline_ms defaults to nothing and must be set explicitly...'
3.280  [pallas-http]    'deadline_ms defaults to nothing and must be set explicitly; retries require idem_key.'
2.803  [veltrix-cache]  'ttl_s is mandatory and must not exceed 86400; Veltrix rejects a write without it.'
2.697  [pallas-http]    'pallas_probe("internal", body=body, deadline_ms=250)'
2.630  [veltrix-cache]  'It raises `VX_TTL_MISSING`.'
```

## Lowest-KL spans

```
0.233  '```python'
0.204  '```\n\nNote:'
0.185  '```\n\nNote:'
```

## Token level

```
[veltrix-cache write] 'st':7.96  'rejects':7.62  'V':7.16  'x':6.98  't':5.76  'without':5.46
[veltrix-cache read]  'rejects':8.21  'V':7.05  'fetch':6.93  'x':6.52  's':6.03
[veltrix-cache rule]  'rejects':7.95  'X':7.91  'elt':6.74  'TT':6.24  'write':6.06
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
