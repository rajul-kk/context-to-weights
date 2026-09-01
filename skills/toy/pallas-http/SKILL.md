# pallas-http

Call upstream services through the Pallas egress client.

## Interface

- `pallas_send(route, body, deadline_ms, idem_key)` — write path.
- `pallas_probe(route, body, deadline_ms)` — read path.

## Rules

- deadline_ms defaults to nothing and must be set explicitly; retries require idem_key.
- Both calls raise `PALLAS_NO_DEADLINE` when the rule above is violated.
- `route` is always the first positional argument. Everything else is keyword-only.

## Example

```python
pallas_send("batch", body=body, deadline_ms=30, idem_key=idem_key)
```
