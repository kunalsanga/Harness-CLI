"""Phase 10.5 — ConvergenceGovernor / BudgetGovernor unit tests.

The governors are deterministic evidence trackers: they never decide anything
by themselves, they report stagnation/over-budget conditions from recorded
tool executions so the runtime (the only authority) can act.
"""

from __future__ import annotations

from harness_core.runtime.governance import BudgetGovernor, ConvergenceGovernor

# ── ConvergenceGovernor ────────────────────────────────────────────────────


def test_no_stagnation_on_distinct_successful_commands():
    gov = ConvergenceGovernor()
    for cmd in ("pytest -q", "npm test", "cargo test"):
        gov.record_tool("run_command", {"command": cmd}, succeeded=True, exit_code=0)
    report = gov.report()
    assert report.stalled is False
    assert report.evidence == []


def test_repeated_identical_failure_is_detected():
    gov = ConvergenceGovernor()
    # Two identical failures warn (diagnose, don't blind-retry).
    for _ in range(2):
        gov.record_tool(
            "run_command",
            {"command": "node test.js"},
            succeeded=False,
            exit_code=1,
            stderr="AssertionError: expected 42",
        )
    report = gov.report()
    assert report.stalled is False  # warn threshold only
    assert any(e.kind == "repeated_failure" for e in report.evidence)


def test_third_identical_failure_stalls_and_recommends_switch():
    gov = ConvergenceGovernor()
    for _ in range(3):
        gov.record_tool(
            "run_command",
            {"command": "node test.js"},
            succeeded=False,
            exit_code=1,
            stderr="AssertionError: expected 42",
        )
    report = gov.report()
    assert report.stalled is True
    assert report.recommended_switch == "TESTER -> DEBUGGER"
    assert report.escalate_model is True
    assert report.escalate_reason
    assert "node test.js" in report.repeated_command


def test_repeated_command_with_success_does_not_stall():
    gov = ConvergenceGovernor()
    # Same command run 3x but only 1 failure streak counts when it eventually
    # succeeds (progress). A successful run clears the failure streak.
    gov.record_tool("run_command", {"command": "node test.js"}, succeeded=False, exit_code=1, stderr="boom")
    gov.record_tool("run_command", {"command": "node test.js"}, succeeded=True, exit_code=0)
    gov.record_tool("run_command", {"command": "node test.js"}, succeeded=True, exit_code=0)
    report = gov.report()
    assert report.stalled is False


def test_identical_patch_is_stagnation_evidence():
    gov = ConvergenceGovernor()
    for _ in range(3):
        gov.record_tool(
            "write_file",
            {"path": "src/app.js", "content": "same content"},
            succeeded=True,
        )
    report = gov.report()
    assert report.stalled is True
    assert any(e.kind == "repeated_patch" for e in report.evidence)


def test_no_progress_streak_stalls():
    gov = ConvergenceGovernor(max_no_progress_events=2)
    gov.record_no_progress()
    gov.record_no_progress()
    report = gov.report()
    assert report.stalled is True
    assert any(e.kind == "no_progress" for e in report.evidence)


def test_reset_clears_state():
    gov = ConvergenceGovernor()
    for _ in range(3):
        gov.record_tool("run_command", {"command": "node test.js"}, succeeded=False, exit_code=1, stderr="x")
    assert gov.report().stalled is True
    gov.reset()
    assert gov.report().stalled is False


def test_report_to_dict_is_structured():
    gov = ConvergenceGovernor()
    for _ in range(3):
        gov.record_tool("run_command", {"command": "node test.js"}, succeeded=False, exit_code=1, stderr="x")
    d = gov.report().to_dict()
    assert d["stalled"] is True
    assert d["recommended_switch"] == "TESTER -> DEBUGGER"
    assert all("count" in e for e in d["evidence"])


# ── BudgetGovernor ─────────────────────────────────────────────────────────


def test_budget_stage_tracking():
    bg = BudgetGovernor(default_tool_cap=3, default_token_cap=100)
    bg.set_stage("implementation")
    bg.record_tool()
    bg.record_tool()
    bg.record_tool()
    bg.record_tokens(120)
    assert bg.stage_over_budget("implementation") is True
    assert "implementation" in bg.over_budget_stages()


def test_budget_other_stages_unaffected():
    bg = BudgetGovernor(default_tool_cap=2, default_token_cap=100)
    bg.set_stage("debugging")
    bg.record_tool()
    bg.record_tool()
    bg.record_tool()  # over
    assert bg.stage_over_budget("debugging") is True
    assert bg.stage_over_budget("planning") is False
    assert bg.stage_over_budget("verification") is False


def test_budget_usage_summary():
    bg = BudgetGovernor(default_tool_cap=5, default_token_cap=1000)
    bg.set_stage("verification")
    bg.record_tool()
    bg.record_tokens(200)
    usage = bg.usage()
    assert usage["total_tools"] == 1
    assert usage["total_tokens"] == 200
    assert usage["current_stage"] == "verification"
    assert usage["stages"]["verification"]["tool_calls"] == 1
    assert usage["stages"]["verification"]["over_budget"] is False


def test_budget_reset():
    bg = BudgetGovernor(default_tool_cap=2, default_token_cap=100)
    bg.set_stage("exploration")
    bg.record_tool()
    bg.record_tool()
    bg.record_tool()
    assert bg.stage_over_budget("exploration") is True
    bg.reset()
    assert bg.stage_over_budget("exploration") is False
    assert bg.usage()["total_tools"] == 0
