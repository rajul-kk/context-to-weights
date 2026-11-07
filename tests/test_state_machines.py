import subprocess
import sys
from pathlib import Path

import pytest

from common.io import choice_list, read_json, write_json
from declare.plan import plan_attention
from sleep.checkpoint import (PHASE_STAMP, ResumeError, mark_best, phase_dir, resume_run,
                              save_phase)

ROOT = Path(__file__).resolve().parents[1]


class StubModel:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_model.bin").write_text("weights")


def test_plan_picks_probe_for_any_attention_mode():
    assert plan_attention(["read", "attention_late"]).kind == "probe"


def test_plan_picks_quant_only_with_bits():
    plan = plan_attention(["read"], 4, "channel_keys")
    assert (plan.kind, plan.kv_bits, plan.recorded_scheme) == ("quant", 4, "channel_keys")


def test_plan_default_is_repeat_and_records_no_scheme():
    plan = plan_attention(["generate", "read"])
    assert (plan.kind, plan.kv_bits, plan.recorded_scheme) == ("repeat", 0, "none")


def test_plan_rejects_bits_with_attention_modes():
    with pytest.raises(SystemExit, match="cannot be combined"):
        plan_attention(["attention"], 4)


def test_plan_rejects_bad_bits_and_schemes():
    with pytest.raises(SystemExit):
        plan_attention(["read"], -1)
    with pytest.raises(SystemExit):
        plan_attention(["read"], 4, "nonsense")


def test_choice_list_accepts_known_and_skips_blanks():
    assert choice_list("data,,sleep", ("data", "sleep"), "stage") == ["data", "sleep"]


def test_choice_list_rejects_typos():
    with pytest.raises(SystemExit, match="sleeep"):
        choice_list("data,sleeep", ("data", "sleep"), "stage")


@pytest.mark.parametrize("script", ["run_all.py", "run_skills.py"])
def test_pipeline_scripts_reject_unknown_stages(script):
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--stages", "bogus",
                           "--dry-run"], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "unknown stage" in proc.stderr


def test_write_json_leaves_no_temporary_file(tmp_path):
    target = tmp_path / "a" / "state.json"
    write_json(target, {"phase": 1})
    assert read_json(target) == {"phase": 1}
    assert not list(target.parent.glob("*.tmp"))


def test_resume_of_an_empty_run_dir_is_fresh(tmp_path):
    assert resume_run(tmp_path, False) is None


def test_save_then_resume_round_trips(tmp_path):
    save_phase(tmp_path, 3, StubModel(), {"phase": 3, "cursor": 12})
    resume = resume_run(tmp_path, False)
    assert resume.state["cursor"] == 12
    assert (Path(resume.adapter) / "adapter_model.bin").exists()
    assert resume.mask_head is None


def test_resume_refuses_state_without_adapter(tmp_path):
    save_phase(tmp_path, 2, StubModel(), {"phase": 2, "cursor": 8})
    for f in (tmp_path / "latest" / "adapter").iterdir():
        f.unlink()
    (tmp_path / "latest" / "adapter").rmdir()
    with pytest.raises(ResumeError, match="interrupted"):
        resume_run(tmp_path, False)


def test_resume_refuses_adapter_without_state(tmp_path):
    save_phase(tmp_path, 2, StubModel(), {"phase": 2, "cursor": 8})
    (tmp_path / "latest" / "state.json").unlink()
    with pytest.raises(ResumeError, match="no state.json"):
        resume_run(tmp_path, False)


def test_resume_refuses_adapter_and_state_from_different_phases(tmp_path):
    save_phase(tmp_path, 2, StubModel(), {"phase": 2, "cursor": 8})
    write_json(tmp_path / "latest" / "adapter" / PHASE_STAMP, {"phase": 3})
    with pytest.raises(ResumeError, match="phase 3"):
        resume_run(tmp_path, False)


def test_resume_refuses_a_mask_head_mismatch(tmp_path):
    save_phase(tmp_path, 2, StubModel(), {"phase": 2, "cursor": 8})
    with pytest.raises(ResumeError, match="mask_head"):
        resume_run(tmp_path, True)


def test_resume_refuses_a_missing_mask_head_file(tmp_path):
    torch = pytest.importorskip("torch")
    save_phase(tmp_path, 2, StubModel(), {"phase": 2, "cursor": 8},
               mask_head=torch.nn.Linear(2, 1))
    (tmp_path / "latest" / "mask_head.pt").unlink()
    with pytest.raises(ResumeError, match="missing"):
        resume_run(tmp_path, True)


def test_mask_head_round_trips(tmp_path):
    torch = pytest.importorskip("torch")
    head = torch.nn.Linear(2, 1)
    save_phase(tmp_path, 1, StubModel(), {"phase": 1, "cursor": 4}, mask_head=head)
    resume = resume_run(tmp_path, True)
    assert Path(resume.mask_head).exists()


def test_saving_without_a_mask_head_removes_a_stale_one(tmp_path):
    torch = pytest.importorskip("torch")
    save_phase(tmp_path, 1, StubModel(), {"phase": 1, "cursor": 4},
               mask_head=torch.nn.Linear(2, 1))
    save_phase(tmp_path, 2, StubModel(), {"phase": 2, "cursor": 8})
    assert not (tmp_path / "latest" / "mask_head.pt").exists()
    assert resume_run(tmp_path, False).state["phase"] == 2


def test_mark_best_copies_the_phase(tmp_path):
    save_phase(tmp_path, 2, StubModel(), {"phase": 2, "cursor": 8})
    mark_best(tmp_path, 2)
    assert read_json(tmp_path / "best" / "best.json") == {"phase": 2}
    assert phase_dir(tmp_path, 2).exists()


def test_sleep_loop_rejects_an_unknown_mix_mode(tmp_path):
    pytest.importorskip("peft")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "sleep" / "loop.py"), "--events", str(tmp_path / "e.jsonl"),
         "--run-dir", str(tmp_path / "run"), "--set", "sleep.mix_mode=bogus"],
        cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "mix_mode" in proc.stderr
