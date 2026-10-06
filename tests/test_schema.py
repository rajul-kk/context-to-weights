import pytest

from common.schema import CompactionEvent, Fact, Probe, Span, Trajectory, Turn


def trajectory():
    return Trajectory(
        traj_id="t1",
        domain="infra",
        turns=[Turn(0, "user", "hello", ["fact", "db_engine", "postgres"]), Turn(1, "assistant", "ok")],
        facts=[Fact("db_engine", "We use postgres.", "postgres", 0)],
        probes=[Probe("db_engine", "Which engine?", "postgres", ["pg"])],
        meta={"seed": 3},
    )


def test_trajectory_round_trip():
    t = trajectory()
    assert Trajectory.from_dict(t.to_dict()).to_dict() == t.to_dict()


def test_trajectory_meta_is_optional_on_load():
    d = trajectory().to_dict()
    del d["meta"]
    assert Trajectory.from_dict(d).meta == {}


def event(before=100, after=40):
    return CompactionEvent(
        event_id=1, traj_id="t1", turn_range=[0, 5],
        spans=[Span("s0", 0, "kept text", True, 0.9, "db_engine"), Span("s1", 1, "dropped", False)],
        summary="sum", tokens_before=before, tokens_after=after, decided_by="model")


def test_compaction_event_round_trip():
    e = event()
    again = CompactionEvent.from_dict(e.to_dict())
    assert again.to_dict() == e.to_dict()
    assert again.spans[0].carries_fact == "db_engine"
    assert again.decided_by == "model"


def test_compaction_ratio():
    assert event(100, 40).compaction_ratio == pytest.approx(0.6)
    assert event(0, 0).compaction_ratio == 0.0


def test_decided_by_defaults_when_missing():
    d = event().to_dict()
    del d["decided_by"]
    assert CompactionEvent.from_dict(d).decided_by == "unknown"
