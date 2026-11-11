from eval import report


def _row(label, acc=0.3, ev=0.1, n_ev=40):
    return {"label": label, "retention_accuracy": acc, "evicted_accuracy": ev, "n_evicted": n_ev, "n": 200}


def test_no_warnings_when_cascading_evicts_and_full_is_a_ceiling():
    assert report.retention_warnings([_row("cascading"), _row("full", acc=0.6)]) == []


def test_degenerate_and_inverted_ceiling_are_both_reported_in_order():
    rows = [_row("cascading", n_ev=0), _row("full", acc=0.2)]
    titles = [t for t, _ in report.retention_warnings(rows)]
    assert titles == ["Configuration is degenerate.", "Ceiling inverted."]
    lines = report.retention_section("TABLE", report.retention_warnings(rows))
    assert lines[:4] == ["## Retention", "", "TABLE", ""]
    assert lines[4].startswith("> **Configuration is degenerate.**")


def test_reading_needs_cascading_and_an_adapter():
    assert report.reading_section([_row("cascading")]) == []
    assert report.reading_section([_row("ours")]) == []


def test_oracle_reading_depends_on_clearing_the_baseline_by_more_than_noise():
    casc, low, high = _row("cascading", ev=0.10), _row("oracle", ev=0.11), _row("oracle", ev=0.60)
    assert report._oracle_reading(low, casc)[0].startswith("**The oracle control does not clear")
    assert report._oracle_reading(high, casc)[0].startswith("**The oracle control clears")


def test_ours_vs_uniform_verdict():
    tie = report._ours_vs_uniform(_row("ours", ev=0.20), _row("uniform", ev=0.21))
    assert "**within noise**" in tie[0]
    ahead = report._ours_vs_uniform(_row("ours", ev=0.60, n_ev=4000), _row("uniform", ev=0.20, n_ev=4000))
    assert "**ours ahead**" in ahead[0]


def test_empty_inputs_give_empty_sections():
    assert report.cost_section({}) == []
    assert report.abstention_section([]) == []
    assert report.scores_section([]) == []
    assert report.header_section([], "no/such/run/root") == []


NAN = float("nan")


def test_oracle_reading_with_a_nan_side_is_not_read_as_clearing():
    assert report._oracle_reading(_row("oracle", ev=0.6), _row("cascading", ev=NAN))[0].startswith(
        "**The oracle control cannot be read**")
    assert report._oracle_reading(_row("oracle", ev=NAN), _row("cascading", ev=0.1))[0].startswith(
        "**The oracle control cannot be read**")


def test_ours_vs_uniform_with_a_nan_side_has_no_winner():
    for ours, uni in ((NAN, 0.2), (0.2, NAN)):
        line = report._ours_vs_uniform(_row("ours", ev=ours), _row("uniform", ev=uni))[0]
        assert "**not comparable**" in line and "ahead" not in line


def test_reading_skips_nan_adapters_when_picking_the_best():
    rows = [_row("cascading", acc=0.5), _row("ours", acc=NAN), _row("uniform", acc=0.4)]
    text = "\n".join(report.reading_section(rows))
    assert "The best adapter is (a) uniform replay at 0.400" in text


def test_reading_without_anything_to_say_is_empty():
    rows = [_row("cascading", acc=0.3), _row("reflection", acc=0.4)]
    assert report.reading_section(rows) == []


def test_scores_with_only_cascading_do_not_claim_every_adapter_degrades():
    casc = {"label": "cascading", "mc_accuracy": 0.5, "evicted_sigma_over_chance": NAN}
    assert not any("Every adapter" in line for line in report.scores_section([casc]))
    worse = dict(casc, label="ours", mc_accuracy=0.4)
    assert any("Every adapter" in line for line in report.scores_section([casc, worse]))


def test_adapters_are_the_sleep_runs():
    assert set(report.ADAPTERS) == set(report.SLEEP_RUNS)
