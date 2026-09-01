# veltrix-cache

Write and read entries in the Veltrix regional cache.

## Interface

- `vx_stash(bucket, key, payload, ttl_s)` — write path.
- `vx_fetch(bucket, key, payload)` — read path.

## Rules

- ttl_s is mandatory and must not exceed 86400; Veltrix rejects a write without it.
- Both calls raise `VX_TTL_MISSING` when the rule above is violated.
- `bucket` is always the first positional argument. Everything else is keyword-only.

## Example

```python
vx_stash("batch", key=key, payload=payload, ttl_s=3600)
```
