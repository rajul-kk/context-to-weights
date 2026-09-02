import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.io import ensure_dir, set_seed, write_jsonl
from common.schema import Fact, Probe, Trajectory, Turn

ASK = [
    "Pull up what we have on {title}.",
    "What do the notes say about {title}?",
    "Anything on {title} in the research folder?",
    "Give me the background on {title}.",
]


def _sentences(para):
    return [s.strip() for s in para if s and s.strip()]


def build_trajectory(traj_id, items, rng):
    turns = []
    facts = []
    probes = []
    idx = 0

    supporting = []
    distractors = []
    for item in items:
        titles = set(item["supporting_titles"])
        for title, sents in item["context"]:
            clean = _sentences(sents)
            if title in titles:
                markers = [clean[i] for i in item["supporting_sents"].get(title, [])
                           if 0 <= i < len(clean)] or clean
                supporting.append((title, clean, markers, item["answer"]))
            else:
                distractors.append((title, clean, None, None))

    rng.shuffle(distractors)
    ordered = supporting + distractors

    for title, sents, markers, answer in ordered:
        if not sents:
            continue
        turns.append(Turn(idx=idx, role="user", content=rng.choice(ASK).format(title=title),
                          tags=["filler"]))
        idx += 1
        body = " ".join(sents)
        tags = ["fact", title] + markers if markers else ["filler"]
        turns.append(Turn(idx=idx, role="assistant", content=f"On {title}: {body}", tags=tags))
        if markers:
            facts.append(Fact(key=title, statement=" ".join(markers), value=answer or "",
                              turn_idx=idx))
        idx += 1

    for item in items:
        probes.append(Probe(fact_key=item["supporting_titles"][0], question=item["question"],
                            answer=item["answer"], aliases=[]))

    return Trajectory(
        traj_id=traj_id, domain="hotpotqa", turns=turns, facts=facts, probes=probes,
        meta={"n_turns": len(turns), "n_probes": len(probes), "source": "hotpotqa-distractor"},
    )


def load_rows(split, limit):
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
    rows = []
    for ex in ds:
        ctx = ex["context"]
        pairs = list(zip(ctx["title"], ctx["sentences"])) if isinstance(ctx, dict) else ctx
        sf = ex["supporting_facts"]
        sf_pairs = list(zip(sf["title"], sf["sent_id"])) if isinstance(sf, dict) else list(sf)
        sent_map = {}
        for title, sent_id in sf_pairs:
            sent_map.setdefault(title, []).append(int(sent_id))
        rows.append({"question": ex["question"], "answer": ex["answer"], "context": pairs,
                     "supporting_titles": sorted(sent_map), "supporting_sents": sent_map})
        if len(rows) >= limit:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/data/hotpotqa")
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--per-trajectory", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    out = ensure_dir(args.out)

    need = (args.n_train + args.n_eval) * args.per_trajectory
    rows = load_rows("validation", need)
    if len(rows) < need:
        raise SystemExit(f"only {len(rows)} examples available, need {need}")

    cursor = 0
    for split, n in (("train", args.n_train), ("eval", args.n_eval)):
        out_rows = []
        for i in range(n):
            items = rows[cursor: cursor + args.per_trajectory]
            cursor += args.per_trajectory
            out_rows.append(build_trajectory(f"hotpot-{split}-{i:04d}", items, rng).to_dict())
        write_jsonl(out / f"{split}.jsonl", out_rows)
        turns = sum(r["meta"]["n_turns"] for r in out_rows) / max(1, len(out_rows))
        print(f"{split}: {len(out_rows)} trajectories, {turns:.0f} turns avg "
              f"-> {out / (split + '.jsonl')}")


if __name__ == "__main__":
    main()
