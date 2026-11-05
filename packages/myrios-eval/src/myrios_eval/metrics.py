import math


def _outcome(v):
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    return float(v)


def success_rate(outcomes):
    values = [_outcome(v) for v in outcomes]
    return sum(values) / len(values) if values else float("nan")


def mean_prompt_tokens(counts):
    counts = list(counts)
    return sum(counts) / len(counts) if counts else float("nan")


def token_saving(arm_tokens, in_context_tokens):
    if not in_context_tokens or in_context_tokens != in_context_tokens:
        return float("nan")
    return 1.0 - arm_tokens / in_context_tokens


def final_per_seed(snapshots, value, seed_key="seed", time_key="night"):
    last = {}
    for snap in snapshots:
        s = snap[seed_key]
        if s not in last or snap[time_key] >= last[s][time_key]:
            last[s] = snap
    return {s: value(last[s]) for s in sorted(last)}


def final_global_last_night(snapshots, value, time_key="night"):
    if not snapshots:
        return float("nan")
    end = max(s[time_key] for s in snapshots)
    finals = [value(s) for s in snapshots if s[time_key] == end]
    return sum(finals) / len(finals)


def earliest_k(skills, k=3, since_key="skills_since_learned", tier_key="tier",
               score_key="success_rate"):
    oldest = sorted(skills, key=lambda s: (-s[since_key], s[tier_key]))[:k]
    return sum(s[score_key] for s in oldest) / len(oldest) if oldest else float("nan")


def _learned_trajectories(history):
    out = {}
    for item, points in history.items():
        pts = sorted(points, key=lambda p: p[0])
        if pts:
            out[item] = pts
    return out


def backward_transfer(history):
    traj = _learned_trajectories(history)
    if not traj:
        return float("nan")
    end = max(p[-1][0] for p in traj.values())
    deltas = [pts[-1][1] - pts[0][1] for pts in traj.values()
              if len(pts) > 1 and pts[-1][0] == end]
    return sum(deltas) / len(deltas) if deltas else float("nan")


def forgetting(history):
    traj = _learned_trajectories(history)
    if not traj:
        return float("nan")
    end = max(p[-1][0] for p in traj.values())
    drops = [max(v for _, v in pts[:-1]) - pts[-1][1] for pts in traj.values()
             if len(pts) > 1 and pts[-1][0] == end]
    return sum(drops) / len(drops) if drops else float("nan")


def history_from_snapshots(snapshots, item_key="skill", time_key="night", score_key="success_rate",
                           items_key="skills"):
    history = {}
    for snap in snapshots:
        for item in snap[items_key]:
            history.setdefault(item[item_key], []).append((snap[time_key], item[score_key]))
    return history


def is_nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))
