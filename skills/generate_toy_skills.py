import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, set_seed, write_json, write_jsonl
from skills.specs import SKILL_SPECS

SKILL_MD = """# {name}

{summary}

## Interface

- `{fn_write}({write_sig})` — write path.
- `{fn_read}({read_sig})` — read path.

## Rules

- {rule}
- Both calls raise `{error}` when the rule above is violated.
- `{arg0}` is always the first positional argument. Everything else is keyword-only.

## Example

```python
{fn_write}("{arg0_value}", {kw_example})
```
"""

WRITE_TASK = "Write the {name} call that stores {obj} for {arg0_value}."
READ_TASK = "Write the {name} call that reads back {obj} from {arg0_value}."
RULE_TASK = "What does {name} raise when the {arg0} rule is violated, and what is the rule?"

WRITE_ANSWER = """Use the write path with every keyword argument set explicitly.

```python
{fn_write}("{arg0_value}", {kw_example})
```

{rule_sentence}"""

READ_ANSWER = """Use the read path against the same first argument.

```python
{fn_read}("{arg0_value}", {read_kw})
```

{rule_sentence}"""

RULE_ANSWER = "It raises `{error}`. The rule is that {rule}"

ARG0_VALUES = ["primary", "eu-west-2", "tier-one", "shared", "internal", "edge", "batch"]


def kw_pairs(spec, rng, n=None):
    args = spec["args"][1:]
    if n:
        args = args[:n]
    out = []
    for a in args:
        if a.endswith("_s") or a.endswith("_ms"):
            out.append(f"{a}={rng.choice([30, 120, 900, 3600, 250, 1500])}")
        elif a in ("scopes", "labels", "inputs"):
            out.append(f'{a}=["{rng.choice(["read", "write", "admin", "src", "deps"])}"]')
        elif a in ("state", "sandbox"):
            out.append(f'{a}="{rng.choice(["strict", "on", "off", "loose"])}"')
        else:
            out.append(f'{a}={a}')
    return ", ".join(out)


def build_skill(spec, rng, n_demos, n_tasks):
    arg0 = spec["args"][0]
    arg0_value = rng.choice(ARG0_VALUES)
    write_sig = ", ".join(spec["args"])
    read_sig = ", ".join([spec["args"][0]] + spec["args"][1:3])
    rule_sentence = "Note: " + spec["rule"]
    doc = SKILL_MD.format(
        name=spec["name"],
        summary=spec["summary"],
        fn_write=spec["fn_write"],
        fn_read=spec["fn_read"],
        write_sig=write_sig,
        read_sig=read_sig,
        rule=spec["rule"],
        error=spec["error"],
        arg0=arg0,
        arg0_value=arg0_value,
        kw_example=kw_pairs(spec, rng),
    )

    def make_item(i):
        kind = ["write", "read", "rule"][i % 3]
        value = rng.choice(ARG0_VALUES)
        obj = rng.choice(spec["objects"])
        if kind == "write":
            q = WRITE_TASK.format(name=spec["name"], obj=obj, arg0_value=value)
            a = WRITE_ANSWER.format(fn_write=spec["fn_write"], arg0_value=value,
                                    kw_example=kw_pairs(spec, rng), rule_sentence=rule_sentence)
            required = [spec["fn_write"], value]
        elif kind == "read":
            q = READ_TASK.format(name=spec["name"], obj=obj, arg0_value=value)
            a = READ_ANSWER.format(fn_read=spec["fn_read"], arg0_value=value,
                                   read_kw=kw_pairs(spec, rng, n=2), rule_sentence=rule_sentence)
            required = [spec["fn_read"], value]
        else:
            q = RULE_TASK.format(name=spec["name"], arg0=arg0)
            a = RULE_ANSWER.format(error=spec["error"], rule=spec["rule"])
            required = [spec["error"]]
        return {"query": q, "response": a, "required": required, "kind": kind}

    demos = [make_item(i) for i in range(n_demos)]
    tasks = [make_item(i + n_demos) for i in range(n_tasks)]
    return doc, demos, tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="skills/toy")
    ap.add_argument("--n-demos", type=int, default=9)
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    root = ensure_dir(args.out)
    index = []

    for spec in SKILL_SPECS:
        doc, demos, tasks = build_skill(spec, rng, args.n_demos, args.n_tasks)
        d = ensure_dir(root / spec["name"])
        (d / "SKILL.md").write_text(doc, encoding="utf-8")
        write_jsonl(d / "demos.jsonl", demos)
        write_jsonl(d / "tasks.jsonl", tasks)
        index.append({"name": spec["name"], "category": spec["category"],
                      "n_demos": len(demos), "n_tasks": len(tasks)})

    write_json(root / "index.json", index)
    print(f"{len(index)} skills -> {root}")


if __name__ == "__main__":
    main()
