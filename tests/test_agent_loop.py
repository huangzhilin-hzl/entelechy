# SPDX-License-Identifier: Apache-2.0
"""Real CLI state transitions with explicitly synthetic GPU observations."""

import json
from pathlib import Path

import pytest
from helpers import admit, init_example, records

from entelechy import guide, interaction
from entelechy.artifacts import read_json, write_json
from entelechy.cli import main, parser
from entelechy.compiler import Schedule, verify

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "session"
    init_example(root)
    return root


def check_actions(result):
    for command in result["next_actions"]:
        assert command["argv"][0] == "entelechy"
        parser().parse_args(command["argv"][1:])


@pytest.mark.parametrize("prefix", [[], ["loop"], ["evolve"]])
def test_empty_command_teaches_same_help(prefix, capsys):
    assert main(prefix) == 0
    text = capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main([*prefix, "--help"])
    assert error.value.code == 0
    assert capsys.readouterr().out == text


@pytest.mark.parametrize("topic", guide.SPEC_TOPICS)
def test_installed_specs_are_queryable(topic, capsys):
    assert main(["spec", topic]) == 0
    assert json.loads(capsys.readouterr().out)["topic"] == topic


def test_documented_ir_examples_remain_executable():
    for operator, schedules in guide.spec("ir")["schedule_examples"].items():
        for raw in schedules:
            assert verify(Schedule.from_dict(raw), operator=operator).valid


def test_submit_failure_preserves_input_and_exposes_compiler_path(session):
    draft = session / "draft.ir.json"
    raw = read_json(draft)
    raw["schedule"]["mapping"] = "persistent_tensorcore"
    write_json(draft, raw)
    result = interaction.submit(session, draft)
    assert result["status"] == "rejected"
    assert result["findings"][0]["root_cause_proven"] is False
    assert read_json(Path(result["submitted_source"])) == raw
    assert any(a["action"] == "compiler_investigation" for a in result["next_actions"])
    assert interaction.loop_status(session)["budget"]["submitted"] == 0
    check_actions(result)


def test_complete_loop_freezes_before_holdout(session, tmp_path, monkeypatch):
    candidate_id = admit(session)
    assert interaction.submit(session, session / "draft.ir.json")["status"] == "duplicate"
    with pytest.raises(ValueError, match="evaluate every"):
        interaction.freeze(session)
    with pytest.raises(ValueError, match="frozen routing"):
        interaction.holdout(session)

    def evaluate(root, trace_input, candidate, split, **kwargs):
        if split == "holdout":
            assert (root / "portfolio.freeze.json").exists()
            assert read_json(root / "experiment.json")["phase"] == "holdout_running"
        return records(trace_input, candidate, split)

    monkeypatch.setattr(interaction, "evaluate_candidate", evaluate)
    result = interaction.evaluate(session, candidate_id)
    assert result["status"] == "evaluated"
    check_actions(result)
    with pytest.raises(ValueError, match="already evaluated"):
        interaction.evaluate(session, candidate_id)
    check_actions(interaction.freeze(session))
    with pytest.raises(ValueError, match="after routing"):
        admit(session)
    result = interaction.holdout(session)
    assert result["status"] == "qualified"
    assert result["report"]["deployment_qualified"] is False
    check_actions(result)
    with pytest.raises(ValueError, match="single-use"):
        interaction.holdout(session)
    assert main(["loop", "export", str(session), "--output", str(tmp_path / "export")]) == 0
    assert (tmp_path / "export" / "LICENSE").is_file()


def test_unavailable_means_fix_environment_and_retry(session, monkeypatch):
    candidate_id = admit(session)
    seen = []

    def evaluate(root, trace_input, candidate, split, **kwargs):
        seen.append(kwargs["attempt"])
        return records(
            trace_input, candidate, split, status="unavailable" if len(seen) == 1 else "ok"
        )

    monkeypatch.setattr(interaction, "evaluate_candidate", evaluate)
    result = interaction.evaluate(session, candidate_id)
    assert result["status"] == "unavailable"
    assert not any(a["action"] == "compiler_investigation" for a in result["next_actions"])
    assert any(a["action"] == "doctor" for a in result["next_actions"])
    with pytest.raises(ValueError, match="without measured"):
        interaction.freeze(session)
    assert interaction.evaluate(session, candidate_id)["status"] == "evaluated"
    assert seen == [1, 2]


def test_slow_candidate_exposes_evidence_without_inventing_bottleneck(session, monkeypatch):
    candidate_id = admit(session)
    monkeypatch.setattr(
        interaction,
        "evaluate_candidate",
        lambda root, c, k, split, **kw: records(c, k, split, latency=3.0),
    )
    result = interaction.evaluate(session, candidate_id)
    assert result["summary"]["qualified"] is False
    finding = result["findings"][0]
    assert finding["code"] == "PERFORMANCE_TARGET_NOT_MET"
    assert finding["root_cause_proven"] is False
    assert interaction.loop_status(session)["last_feedback"]["event_id"] == result["event_id"]
    check_actions(result)


def test_device_is_inherited_and_fixed_after_measurement(session, monkeypatch):
    first = admit(session)
    draft = session / "draft.ir.json"
    raw = read_json(draft)
    raw["schedule"]["items_per_thread"] = 2
    write_json(draft, raw)
    second = admit(session)
    seen = []

    def evaluate(root, trace_input, candidate, split, **kwargs):
        seen.append(kwargs["device"])
        return records(
            trace_input, candidate, split, status="unavailable" if len(seen) == 1 else "ok"
        )

    monkeypatch.setattr(interaction, "evaluate_candidate", evaluate)
    interaction.evaluate(session, first)
    result = interaction.evaluate(session, first, device=3)
    pending = next(a for a in result["next_actions"] if a["action"] == "evaluate")
    assert parser().parse_args(pending["argv"][1:]).device == 3
    with pytest.raises(ValueError, match="device cannot change"):
        interaction.evaluate(session, second, device=0)
    interaction.evaluate(session, second)
    interaction.freeze(session)
    with pytest.raises(ValueError, match="device cannot change"):
        interaction.holdout(session, device=0)
    interaction.holdout(session)
    assert seen == [0, 3, 3, 3]


def test_holdout_regression_closes_the_session(session, monkeypatch):
    candidate_id = admit(session)
    monkeypatch.setattr(
        interaction,
        "evaluate_candidate",
        lambda root, c, k, split, **kw: records(
            c, k, split, latency=1.0 if split == "train" else 3.0
        ),
    )
    interaction.evaluate(session, candidate_id)
    interaction.freeze(session)
    assert interaction.holdout(session)["status"] == "rejected"
    with pytest.raises(ValueError):
        interaction.evaluate(session, candidate_id)
    with pytest.raises(ValueError):
        interaction.freeze(session)


def test_identity_drift_and_unadmitted_candidate_are_refused(session, monkeypatch):
    with pytest.raises(ValueError, match="not admitted"):
        interaction.evaluate(session, "unknown")
    monkeypatch.setattr(interaction, "evaluation_fingerprint", lambda: "changed")
    with pytest.raises(ValueError, match="implementation changed"):
        interaction.loop_status(session)


def test_source_and_routing_tampering_cannot_reach_holdout(session, monkeypatch):
    candidate_id = admit(session)
    monkeypatch.setattr(
        interaction, "evaluate_candidate", lambda root, c, k, split, **kw: records(c, k, split)
    )
    interaction.evaluate(session, candidate_id)
    interaction.freeze(session)
    routing = read_json(session / "portfolio.json")
    routing["routing"]["all"] = "baseline"
    write_json(session / "portfolio.json", routing)
    with pytest.raises(ValueError, match="routing changed"):
        interaction.holdout(session)


def test_session_has_one_writer(session):
    import fcntl

    with (session / ".loop.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another CLI operation"):
            admit(session)


def test_command_errors_are_machine_readable(session, capsys):
    assert main(["loop", "holdout", str(session)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "error"
    assert "frozen routing" in result["error"]["message"]


@pytest.mark.parametrize("command", ["check", "analyze", "loop submit"])
def test_removed_schedule_option_is_not_accepted(command, session):
    with pytest.raises(SystemExit) as error:
        parser().parse_args([*command.split(), str(session), "--schedule", "unused.json"])
    assert error.value.code == 2


def test_bare_schedule_is_not_a_cli_program(session):
    draft = session / "draft.ir.json"
    write_json(draft, read_json(draft)["schedule"])
    result = interaction.submit(session, draft)
    assert result["status"] == "rejected"
    assert "schedule and bindings" in result["findings"][0]["message"]
    check_actions(result)


def test_rejected_ir_bytes_are_part_of_the_evidence(session):
    draft = session / "draft.ir.json"
    write_json(draft, {"unsupported": "program"})
    result = interaction.submit(session, draft)
    Path(result["submitted_source"]).write_text("{}\n")
    with pytest.raises(ValueError, match="rejected IR changed"):
        interaction.loop_status(session)
