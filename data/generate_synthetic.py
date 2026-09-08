import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, set_seed, write_jsonl
from common.schema import Fact, Probe, Trajectory, Turn
from data.banks import (
    CLOSERS,
    FACT_TEMPLATES,
    FILLER_ASSISTANT,
    FILLER_USER,
    PROJECT_NAMES,
    SERVICES,
)


def _fill(text, project, service, value=None):
    return text.format(project=project, service=service, value=value)


def build_trajectory(traj_id, rng, n_facts, n_turns, early_window, unmarked=False,
                     project=None):
    project = project or rng.choice(PROJECT_NAMES)
    services = rng.sample(SERVICES, k=min(len(SERVICES), max(3, n_facts)))
    templates = rng.sample(FACT_TEMPLATES, k=min(len(FACT_TEMPLATES), n_facts))

    fact_slots = sorted(rng.sample(range(0, early_window), k=len(templates)))
    turns = []
    facts = []
    probes = []
    slot_set = set(fact_slots)
    fact_iter = iter(range(len(templates)))

    idx = 0
    unit = 0
    while len(turns) < n_turns:
        service = rng.choice(services)
        pair = rng.randrange(len(FILLER_USER))
        if unit in slot_set:
            f_i = next(fact_iter)
            tpl = templates[f_i]
            value = rng.choice(tpl["values"])
            svc = services[f_i % len(services)]
            statement = _fill(tpl["statement"], project, svc, value)
            question = _fill(tpl["question"], project, svc)
            user_line = _fill(FILLER_USER[pair], project, svc)
            turns.append(Turn(idx=idx, role="user", content=user_line, tags=["filler"]))
            idx += 1
            body = _fill(FILLER_ASSISTANT[pair], project, svc)
            turns.append(
                Turn(
                    idx=idx,
                    role="assistant",
                    content=(f"{body} {statement}" if unmarked
                             else f"{body} One thing to lock in: {statement}"),
                    tags=["fact", tpl["key"], value],
                )
            )
            facts.append(Fact(key=tpl["key"], statement=statement, value=value, turn_idx=idx))
            probes.append(Probe(fact_key=tpl["key"], question=question, answer=value, aliases=[]))
            idx += 1
        else:
            turns.append(
                Turn(idx=idx, role="user", content=_fill(FILLER_USER[pair], project, service), tags=["filler"])
            )
            idx += 1
            turns.append(
                Turn(
                    idx=idx,
                    role="assistant",
                    content=_fill(FILLER_ASSISTANT[pair], project, service),
                    tags=["filler"],
                )
            )
            idx += 1
            if rng.random() < 0.25 and len(turns) < n_turns:
                turns.append(Turn(idx=idx, role="user", content=rng.choice(CLOSERS), tags=["filler"]))
                idx += 1
        unit += 1

    turns = turns[:n_turns]
    kept_keys = {t.tags[1] for t in turns if "fact" in t.tags}
    facts = [f for f in facts if f.key in kept_keys]
    probes = [p for p in probes if p.fact_key in kept_keys]

    return Trajectory(
        traj_id=traj_id,
        domain="software-dev",
        turns=turns,
        facts=facts,
        probes=probes,
        meta={"project": project, "n_turns": len(turns), "n_facts": len(facts)},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/data/synthetic")
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--n-facts", type=int, default=6)
    ap.add_argument("--n-turns", type=int, default=120)
    ap.add_argument("--early-window", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unmarked", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out = ensure_dir(args.out)

    for split, n in (("train", args.n_train), ("eval", args.n_eval)):
        rows = []
        for i in range(n):
            traj = build_trajectory(
                traj_id=f"{split}-{i:04d}",
                rng=rng,
                n_facts=args.n_facts,
                n_turns=args.n_turns,
                early_window=args.early_window,
                unmarked=args.unmarked,
                project=f"{PROJECT_NAMES[i % len(PROJECT_NAMES)]}-{i:02d}",
            )
            rows.append(traj.to_dict())
        write_jsonl(out / f"{split}.jsonl", rows)
        print(f"{split}: {len(rows)} trajectories -> {out / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
