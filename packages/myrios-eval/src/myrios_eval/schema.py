import json
import math
from datetime import datetime, timezone
from pathlib import Path

from myrios_eval.stats import aggregate_seeds

SCHEMA = "myrios_eval.result"
SCHEMA_VERSION = "1.0"
HARNESS_VERSION = "0.1.0"
REQUIRED = ("schema", "schema_version", "harness_version", "experiment", "repo", "commit",
            "created", "arms")


def make_arm(arm, per_seed, label=None, conditions=None, aggregate=None):
    per_seed = sorted(per_seed, key=lambda r: r["seed"])
    if aggregate is None:
        names = sorted({k for r in per_seed for k in r["metrics"]})
        aggregate = {k: aggregate_seeds([r["metrics"].get(k) for r in per_seed]) for k in names}
    return {"arm": arm, "label": label or arm, "conditions": dict(conditions or {}),
            "per_seed": per_seed, "aggregate": aggregate}


def make_result(experiment, repo, commit, arms, comparisons=None, model=None, token_counter=None,
                prompt_tokens_scope=None, flops_method=None, extra=None, created=None):
    doc = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "harness_version": HARNESS_VERSION,
        "experiment": experiment,
        "repo": repo,
        "commit": commit,
        "created": created or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "token_counter": token_counter,
        "prompt_tokens_scope": prompt_tokens_scope,
        "flops_method": flops_method,
        "arms": list(arms),
        "comparisons": list(comparisons or []),
        "extra": dict(extra or {}),
    }
    validate(doc)
    return doc


def _is_number(v):
    return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))


def validate(doc):
    errors = []
    for key in REQUIRED:
        if key not in doc:
            errors.append(f"missing top-level key {key!r}")
    if errors:
        raise ValueError("; ".join(errors))
    if doc["schema"] != SCHEMA:
        errors.append(f"schema is {doc['schema']!r}, expected {SCHEMA!r}")
    if str(doc["schema_version"]).split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        errors.append(f"schema_version {doc['schema_version']} has a different major version")
    seen = set()
    metric_names = set()
    for arm in doc["arms"]:
        key = (arm.get("arm"), json.dumps(arm.get("conditions", {}), sort_keys=True))
        if key in seen:
            errors.append(f"duplicate arm {key}")
        seen.add(key)
        seeds = []
        names = set()
        for row in arm.get("per_seed", []):
            if not isinstance(row.get("seed"), int) or isinstance(row.get("seed"), bool):
                errors.append(f"arm {arm.get('arm')}: seed must be an integer")
            seeds.append(row.get("seed"))
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                errors.append(f"arm {arm.get('arm')}: per_seed entry has no metrics dict")
                continue
            for name, v in metrics.items():
                if not _is_number(v):
                    errors.append(f"arm {arm.get('arm')}: metric {name} is not a number")
            names |= set(metrics)
        if len(seeds) != len(set(seeds)):
            errors.append(f"arm {arm.get('arm')}: repeated seed")
        extra = set(arm.get("aggregate", {})) - names
        if extra and arm.get("per_seed"):
            errors.append(f"arm {arm.get('arm')}: aggregate has metrics with no per-seed value "
                          f"{sorted(extra)}")
        metric_names |= names | set(arm.get("aggregate", {}))
    if "prompt_tokens" in metric_names and not (doc.get("token_counter")
                                                and doc.get("prompt_tokens_scope")):
        errors.append("prompt_tokens needs token_counter and prompt_tokens_scope")
    if any(n.endswith("_flops") for n in metric_names) and not doc.get("flops_method"):
        errors.append("a *_flops metric needs flops_method")
    if errors:
        raise ValueError("; ".join(errors))
    return True


def _clean(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def write_result(path, doc):
    validate(doc)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(doc), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path


def read_result(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    validate(doc)
    return doc
