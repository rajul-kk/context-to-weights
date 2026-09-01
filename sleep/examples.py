import random
from dataclasses import asdict, dataclass
from typing import List

CONSOLIDATION_CUE = "Session {traj_id} retained note:\n"
REFLECTION_CUE = "Session {traj_id} reflection:\n"


@dataclass
class SleepExample:
    prompt: str
    target: str
    kept: bool
    source: str
    traj_id: str
    fact_key: str = None

    def to_dict(self):
        return asdict(self)


def from_compaction(event, include_dropped=False):
    out = []
    for span in event.spans:
        if not span.kept and not include_dropped:
            continue
        out.append(
            SleepExample(
                prompt=CONSOLIDATION_CUE.format(traj_id=event.traj_id),
                target=span.text,
                kept=span.kept,
                source="compaction",
                traj_id=event.traj_id,
                fact_key=span.carries_fact,
            )
        )
    return out


def from_uniform(event, rng, n=None):
    spans = list(event.spans)
    n = n or sum(1 for s in spans if s.kept)
    picks = rng.sample(spans, k=min(n, len(spans)))
    return [
        SleepExample(
            prompt=CONSOLIDATION_CUE.format(traj_id=event.traj_id),
            target=s.text,
            kept=True,
            source="uniform",
            traj_id=event.traj_id,
            fact_key=s.carries_fact,
        )
        for s in picks
    ]


def from_reflection(event, reflection_text):
    return [
        SleepExample(
            prompt=REFLECTION_CUE.format(traj_id=event.traj_id),
            target=reflection_text,
            kept=True,
            source="reflection",
            traj_id=event.traj_id,
            fact_key=None,
        )
    ]


class ReplayBuffer:
    def __init__(self, capacity=40, seed=0):
        self.capacity = capacity
        self.items: List[SleepExample] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def add_all(self, examples):
        for ex in examples:
            self.add(ex)

    def add(self, ex):
        self.seen += 1
        if len(self.items) < self.capacity:
            self.items.append(ex)
            return
        j = self.rng.randrange(self.seen)
        if j < self.capacity:
            self.items[j] = ex

    def sample(self, n):
        if not self.items:
            return []
        return self.rng.sample(self.items, k=min(n, len(self.items)))

    def state_dict(self):
        return {"capacity": self.capacity, "seen": self.seen,
                "items": [i.to_dict() for i in self.items]}

    def load_state_dict(self, state):
        self.capacity = state["capacity"]
        self.seen = state["seen"]
        self.items = [SleepExample(**i) for i in state["items"]]
