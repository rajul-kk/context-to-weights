from myrios_eval.metrics import (backward_transfer, earliest_k, final_global_last_night,
                                 final_per_seed, forgetting, history_from_snapshots,
                                 mean_prompt_tokens, success_rate)
from myrios_eval.schema import make_arm, make_result
from myrios_eval.stats import compare_arms


def myrios_skill_metrics(records):
    in_group = [r["passed"] for r in records if r["in_group"]]
    out_group = [r["passed"] for r in records if not r["in_group"]]
    return {
        "success_rate": success_rate(r["passed"] for r in records),
        "success_rate_in_group": success_rate(in_group),
        "success_rate_out_group": success_rate(out_group),
        "prompt_tokens": mean_prompt_tokens(r["prompt_tokens"] for r in records),
    }


def from_myrios_skill_summary(summaries_by_seed, arm, conditions=None):
    per_seed = []
    for seed, s in sorted(summaries_by_seed.items()):
        per_seed.append({"seed": int(seed), "n_items": s["n"], "metrics": {
            "success_rate": s["pass_rate"],
            "success_rate_in_group": s["pass_rate_in_group"],
            "success_rate_out_group": s["pass_rate_out_group"],
            "prompt_tokens": s["mean_prompt_tokens"],
        }})
    return make_arm(arm, per_seed, label=arm, conditions=conditions)


def _snapshot_success(snap):
    skills = snap.get("skills") or []
    if skills:
        return success_rate(s["success_rate"] for s in skills)
    return snap["success_rate"]


def _snapshot_tokens(snap):
    skills = snap.get("skills") or []
    if skills:
        return mean_prompt_tokens(s["prompt_tokens"] for s in skills)
    return snap["mean_tokens"]


def _snapshot_earliest(snap, k=3):
    return earliest_k(snap.get("skills") or [], k)


def split_episodes_by_seed(method):
    groups = []
    for e in method.get("episodes") or []:
        if e["episode"] == 0 or not groups:
            groups.append([])
        groups[-1].append(e)
    seeds = []
    for snap in method["snapshots"]:
        if snap["seed"] not in seeds:
            seeds.append(snap["seed"])
    if len(groups) != len(seeds):
        return {}
    return dict(zip(seeds, groups))


def skill_library_legacy_finals(method):
    snaps = method["snapshots"]
    episodes = split_episodes_by_seed(method)
    run_tokens = [mean_prompt_tokens(e["prompt_tokens"] for e in g) for g in episodes.values()]
    return {
        "mean_prompt_tokens": sum(run_tokens) / len(run_tokens) if run_tokens else float("nan"),
        "final_success_rate": final_global_last_night(snaps, _snapshot_success),
        "final_earliest_3": final_global_last_night(snaps, _snapshot_earliest),
        "final_stored_params": int(final_global_last_night(snaps, lambda s: s["stored_params"])),
        "final_mean_tokens": final_global_last_night(snaps, _snapshot_tokens),
    }


def skill_library_per_seed(method):
    snaps = method["snapshots"]
    success = final_per_seed(snaps, _snapshot_success)
    earliest = final_per_seed(snaps, _snapshot_earliest)
    params = final_per_seed(snaps, lambda s: s["stored_params"])
    tokens = final_per_seed(snaps, _snapshot_tokens)
    sizes = final_per_seed(snaps, lambda s: s["library_size"])
    episodes = split_episodes_by_seed(method)
    by_seed = {}
    for snap in snaps:
        by_seed.setdefault(snap["seed"], []).append(snap)
    rows = []
    for seed in sorted(success):
        history = history_from_snapshots(by_seed[seed])
        run_tokens = (mean_prompt_tokens(e["prompt_tokens"] for e in episodes[seed])
                      if seed in episodes else None)
        rows.append({"seed": int(seed), "n_items": sizes[seed], "metrics": {
            "success_rate": success[seed],
            "earliest_3": earliest[seed],
            "prompt_tokens": tokens[seed],
            "run_prompt_tokens": run_tokens,
            "stored_params": params[seed],
            "backward_transfer": backward_transfer(history),
            "forgetting": forgetting(history),
        }})
    return rows


def from_skill_library_results(payload, experiment, commit, conditions=None, model=None,
                               token_counter="heuristic:chars/4",
                               prompt_tokens_scope="raw_prompt/per_task_attempt",
                               compare=(("ours", "isolated"), ("ours", "in_context"),
                                        ("ours", "seq_no_replay")),
                               created=None):
    arms = []
    per_seed_by_arm = {}
    legacy = {}
    for key, method in payload["methods"].items():
        rows = skill_library_per_seed(method)
        per_seed_by_arm[key] = {r["seed"]: r["metrics"] for r in rows}
        arms.append(make_arm(key, rows, label=method.get("label"), conditions=conditions))
        legacy[key] = skill_library_legacy_finals(method)
    comparisons = []
    for a, b in compare:
        if a not in per_seed_by_arm or b not in per_seed_by_arm:
            continue
        for metric in ("success_rate", "earliest_3"):
            row = compare_arms({s: m[metric] for s, m in per_seed_by_arm[a].items()},
                               {s: m[metric] for s, m in per_seed_by_arm[b].items()})
            row.update({"a": a, "b": b, "metric": metric, "conditions": dict(conditions or {})})
            comparisons.append(row)
    return make_result(experiment, "skill-library", commit, arms, comparisons=comparisons,
                       model=model, token_counter=token_counter,
                       prompt_tokens_scope=prompt_tokens_scope,
                       extra={"run_id": payload.get("run_id"), "backend": payload.get("backend"),
                              "legacy_global_last_night": legacy},
                       created=created)
