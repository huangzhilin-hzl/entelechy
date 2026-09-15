# SPDX-License-Identifier: Apache-2.0
"""Export admission and evidence checks through the native CLI session."""

from pathlib import Path

import pytest
from helpers import admit, init_example, records

from entelechy import __version__, interaction
from entelechy.artifacts import read_json, write_json
from entelechy.executor import evaluate_candidate
from entelechy.export import export_portfolio

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def session(tmp_path):
    root = tmp_path / "session"
    value = init_example(root)
    candidate_id = admit(root)
    return root, value, candidate_id


def complete(session, monkeypatch):
    root, _, candidate_id = session
    monkeypatch.setattr(
        interaction,
        "evaluate_candidate",
        lambda root, value, candidate, split, **kw: records(value, candidate, split),
    )
    interaction.evaluate(root, candidate_id)
    interaction.freeze(root)
    interaction.holdout(root)
    return root


def test_input_and_source_changes_cannot_execute(session):
    root, value, _ = session
    candidate = read_json(root / "experiment.json")["candidates"][0]
    original = read_json(root / "input.json")
    altered = {**original, "seed": original["seed"] + 1}
    write_json(root / "input.json", altered)
    with pytest.raises(ValueError, match="input snapshot changed"):
        interaction.loop_status(root)
    write_json(root / "input.json", original)
    (root / candidate["source"]).write_text("// altered after submission\n")
    with pytest.raises(ValueError, match="modified"):
        evaluate_candidate(root, value, candidate, "train")


def test_export_preserves_license_and_native_sources(session, tmp_path, monkeypatch):
    root = complete(session, monkeypatch)
    destination = tmp_path / "export"
    exported = export_portfolio(root, destination)
    assert exported["sources"]
    assert exported["deployment_qualified"] is False
    assert exported["package_version"] == __version__
    assert exported["license"] == "Apache-2.0"
    assert (destination / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert "[Apache-2.0](LICENSE)" in (destination / "README.md").read_text()
    assert (destination / "dataset/definitions/selected.json").is_file()

    def missing_license():
        raise RuntimeError("missing project license")

    monkeypatch.setattr("entelechy.export.project_license_text", missing_license)
    with pytest.raises(RuntimeError, match="missing project license"):
        export_portfolio(root, tmp_path / "unlicensed")
    assert not (tmp_path / "unlicensed").exists()


def test_report_flag_cannot_qualify_an_unfinished_session(session, tmp_path):
    root, _, _ = session
    write_json(root / "report.json", {"qualified": True})
    with pytest.raises(ValueError, match="completed"):
        export_portfolio(root, tmp_path / "export")


def test_export_rechecks_raw_evidence_and_candidate_identities(session, tmp_path, monkeypatch):
    root = complete(session, monkeypatch)
    rows = read_json(root / "holdout_results.json")
    rows[0]["candidate_samples_ms"] = [20.0] * 8
    write_json(root / "holdout_results.json", rows)
    with pytest.raises(ValueError, match="raw held-out"):
        export_portfolio(root, tmp_path / "slow-export")
    assert not (tmp_path / "slow-export").exists()


def test_export_rebuilds_native_metadata_from_the_frozen_input(session, tmp_path, monkeypatch):
    root = complete(session, monkeypatch)
    definition_path = root / "dataset/definitions/selected.json"
    original = read_json(definition_path)
    changed = {**original, "description": "Edited after evaluation"}
    write_json(definition_path, changed)
    write_json(root / "dataset/solutions/baseline.json", {"unrelated": "metadata"})
    destination = tmp_path / "export"
    export_portfolio(root, destination)
    assert read_json(destination / "dataset/definitions/selected.json") == original
    assert (
        read_json(destination / "dataset/solutions/baseline.json")
        == session[1].to_dict()["baseline_solution"]
    )
