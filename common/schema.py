from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    idx: int
    role: str
    content: str
    tags: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Fact:
    key: str
    statement: str
    value: str
    turn_idx: int

    def to_dict(self):
        return asdict(self)


@dataclass
class Probe:
    fact_key: str
    question: str
    answer: str
    aliases: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Trajectory:
    traj_id: str
    domain: str
    turns: List[Turn]
    facts: List[Fact]
    probes: List[Probe]
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "traj_id": self.traj_id,
            "domain": self.domain,
            "turns": [t.to_dict() for t in self.turns],
            "facts": [f.to_dict() for f in self.facts],
            "probes": [p.to_dict() for p in self.probes],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d):
        return Trajectory(
            traj_id=d["traj_id"],
            domain=d["domain"],
            turns=[Turn(**t) for t in d["turns"]],
            facts=[Fact(**f) for f in d["facts"]],
            probes=[Probe(**p) for p in d["probes"]],
            meta=d.get("meta", {}),
        )


@dataclass
class Span:
    span_id: str
    turn_idx: int
    text: str
    kept: bool
    score: Optional[float] = None
    carries_fact: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class CompactionEvent:
    event_id: int
    traj_id: str
    turn_range: List[int]
    spans: List[Span]
    summary: str
    tokens_before: int
    tokens_after: int
    decided_by: str = "unknown"

    @property
    def compaction_ratio(self):
        if self.tokens_before == 0:
            return 0.0
        return 1.0 - (self.tokens_after / self.tokens_before)

    def to_dict(self):
        return {
            "event_id": self.event_id,
            "traj_id": self.traj_id,
            "turn_range": self.turn_range,
            "spans": [s.to_dict() for s in self.spans],
            "summary": self.summary,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "decided_by": self.decided_by,
            "compaction_ratio": self.compaction_ratio,
        }

    @staticmethod
    def from_dict(d):
        return CompactionEvent(
            event_id=d["event_id"],
            traj_id=d["traj_id"],
            turn_range=d["turn_range"],
            spans=[Span(**s) for s in d["spans"]],
            summary=d["summary"],
            tokens_before=d["tokens_before"],
            tokens_after=d["tokens_after"],
            decided_by=d.get("decided_by", "unknown"),
        )
