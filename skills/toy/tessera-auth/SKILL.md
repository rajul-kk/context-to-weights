# tessera-auth

Mint and verify Tessera capability tokens.

## Interface

- `tessera_mint(subject, scopes, audience, not_after)` — write path.
- `tessera_verify(subject, scopes, audience)` — read path.

## Rules

- scopes must be a sorted list; Tessera hashes them and an unsorted list fails verification.
- Both calls raise `TESSERA_SCOPE_ORDER` when the rule above is violated.
- `subject` is always the first positional argument. Everything else is keyword-only.

## Example

```python
tessera_mint("eu-west-2", scopes=["deps"], audience=audience, not_after=not_after)
```
