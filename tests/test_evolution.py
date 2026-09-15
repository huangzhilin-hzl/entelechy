# SPDX-License-Identifier: Apache-2.0
"""Compiler revision gates; synthetic data tests mechanics, never performance."""

import shutil
from pathlib import Path

import pytest
from helpers import admit, init_example, records

import entelechy
from entelechy import evolution, interaction, search
from entelechy.artifacts import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(
        Path(entelechy.__file__).resolve().parent,
        checkout / "entelechy",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(ROOT / "examples", checkout / "examples")
    (checkout / "tests").mkdir()
    (checkout / "tests" / "test_minimal.py").write_text("def test_minimal():\n    assert True\n")
    for name in ("LICENSE", "pyproject.toml"):
        shutil.copyfile(ROOT / name, checkout / name)
    monkeypatch.setattr(evolution, "_checkout", lambda: checkout)
    session = tmp_path / "before"
    init_example(session)
    candidate_id = admit(session)
    event = interaction.loop_status(session)["last_feedback"]
    proposal = tmp_path / "proposal"
    evolution.begin(session, Path(event["evidence_path"]), "Test a lowering hypothesis.", proposal)
    return checkout, session, candidate_id, proposal


def changed_compiler(checkout):
    path = checkout / "entelechy" / "compiler" / "cuda.py"
    path.write_text(path.read_text() + "\n# Candidate revision in a test-only checkout.\n")


def fake_tests(checkout, tests, config, log, timeout_s):
    log.write_text("Synthetic runner result for gate unit tests.\n")
    return {"passed": True, "exit_code": 0, "log": str(log)}


def test_begin_binds_real_evidence_and_snapshots(experiment):
    checkout, session, _, proposal = experiment
    data = read_json(proposal / "proposal.json")
    assert data["base_sources"]
    assert (proposal / "source" / "entelechy" / "compiler" / "cuda.py").is_file()
    assert (proposal / "regression" / "tests" / "test_minimal.py").is_file()
    assert data["input_id"] == read_json(session / "experiment.json")["input_id"]
    assert read_json(proposal / "input.json") == read_json(session / "input.json")


def test_changed_evidence_is_refused(experiment, tmp_path):
    _, session, _, _ = experiment
    event = interaction.loop_status(session)["last_feedback"]
    path = Path(event["evidence_path"])
    event["status"] = "edited"
    write_json(path, event)
    with pytest.raises(ValueError, match="unchanged"):
        evolution.begin(session, path, "An edited hypothesis", tmp_path / "bad")


def test_no_changes_and_protected_oracle_changes_are_refused(experiment):
    checkout, _, _, proposal = experiment
    with pytest.raises(ValueError, match="no compiler"):
        evolution.validate(proposal)
    path = checkout / "entelechy" / "runtime" / "worker.py"
    path.write_text(path.read_text() + "\n# An oracle edit cannot qualify as compiler evolution.\n")
    with pytest.raises(ValueError, match="protected"):
        evolution.validate(proposal)


def test_frozen_regression_cannot_be_rewritten(experiment):
    _, _, _, proposal = experiment
    (proposal / "regression" / "tests" / "test_minimal.py").write_text("")
    with pytest.raises(ValueError, match="snapshot changed"):
        evolution.validate(proposal)


def test_validate_runs_both_test_sets_and_static_corpus(experiment, monkeypatch):
    checkout, _, _, proposal = experiment
    changed_compiler(checkout)
    calls = []

    def runner(*args):
        calls.append(args[1])
        return fake_tests(*args)

    monkeypatch.setattr(evolution, "_run_tests", runner)
    result = evolution.validate(proposal)
    assert result["status"] == "static_validated"
    assert calls == [proposal / "regression" / "tests", checkout / "tests"]
    assert len(result["corpus"]) == 3
    assert all(c["static_pass"] for c in result["corpus"])
    assert result["gpu_qualified"] is False


def test_failed_frozen_tests_block_trial_action(experiment, monkeypatch):
    checkout, _, _, proposal = experiment
    changed_compiler(checkout)
    monkeypatch.setattr(
        evolution,
        "_run_tests",
        lambda *args: {"passed": False, "exit_code": 1, "log": str(args[3])},
    )
    result = evolution.validate(proposal)
    assert result["status"] == "rejected"
    assert not any(a["action"] == "new_revision_trial" for a in result["next_actions"])


def test_matched_comparison_uses_training_and_refuses_budget_drift(
    experiment, tmp_path, monkeypatch
):
    checkout, before, candidate_id, proposal = experiment
    monkeypatch.setattr(
        interaction, "evaluate_candidate", lambda root, c, k, split, **kw: records(c, k, split)
    )
    interaction.evaluate(before, candidate_id)
    changed_compiler(checkout)
    monkeypatch.setattr(evolution, "_run_tests", fake_tests)
    for module in (evolution, interaction, search):
        monkeypatch.setattr(module, "evaluation_fingerprint", lambda: "test-only-revision-2")
    evolution.validate(proposal)
    after = tmp_path / "after"
    interaction.fork_loop(before, after)
    new_id = admit(after)
    monkeypatch.setattr(
        interaction,
        "evaluate_candidate",
        lambda root, c, k, split, **kw: records(c, k, split, latency=0.8),
    )
    interaction.evaluate(after, new_id)
    result = evolution.compare(proposal, after)
    assert result["observed_geomean_speedup"] == pytest.approx(1.25)
    assert result["generalization_qualified"] is False
    assert result["automatic_promotion"] is False
    assert not (after / "holdout_results.json").exists()
    data = read_json(after / "experiment.json")
    data["max_candidates"] = 3
    write_json(after / "experiment.json", data)
    with pytest.raises(ValueError, match="matched budgets"):
        evolution.compare(proposal, after)


def test_validation_for_different_source_is_refused(experiment, tmp_path, monkeypatch):
    checkout, _, _, proposal = experiment
    changed_compiler(checkout)
    monkeypatch.setattr(evolution, "_run_tests", fake_tests)
    evolution.validate(proposal)
    monkeypatch.setattr(evolution, "evaluation_fingerprint", lambda: "later-edit")
    with pytest.raises(ValueError, match="validate this exact"):
        evolution.compare(proposal, tmp_path / "unread-trial")


def test_rejected_ir_snapshot_is_required_to_validate_a_revision(tmp_path):
    session = tmp_path / "before"
    init_example(session)
    draft = session / "draft.ir.json"
    write_json(draft, {"unsupported": "program"})
    result = interaction.submit(session, draft)
    proposal = tmp_path / "proposal"
    evolution.begin(session, Path(result["evidence_path"]), "Investigate unsupported IR.", proposal)
    assert read_json(proposal / "rejected.ir.json") == {"unsupported": "program"}
    write_json(proposal / "rejected.ir.json", {})
    with pytest.raises(ValueError, match="frozen rejected IR changed"):
        evolution.validate(proposal)
