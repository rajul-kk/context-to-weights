# cinder-metrics

Emit and query Cinder time-series metrics.

## Interface

- `cinder_point(series, value, labels, resolution_s)` — write path.
- `cinder_range(series, value, labels)` — read path.

## Rules

- labels must be low-cardinality; Cinder drops a series once it exceeds 200 label values.
- Both calls raise `CINDER_CARDINALITY` when the rule above is violated.
- `series` is always the first positional argument. Everything else is keyword-only.

## Example

```python
cinder_point("batch", value=value, labels=["read"], resolution_s=30)
```
