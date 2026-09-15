# SPDX-License-Identifier: Apache-2.0
"""Native Trace ingestion, ABI, evidence and evolution regressions; no synthetic GPU claims."""

import contextlib
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import records

from entelechy import interaction
from entelechy.artifacts import read_json, validate_candidate_artifact, write_json
from entelechy.cli import main
from entelechy.compiler.trace import compile_trace, definition_id, seed_program
from entelechy.trace_input import (
    TraceInput,
    inspect_dataset,
    load_dataset,
    prepare_input,
    snapshot_input,
    validate_native,
    verify_blobs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("topic", ["loop", "compiler"])
def test_native_tutorial_is_executable(topic, capsys):
    assert main(["tutorial", topic]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["topic"] == topic


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "dataset"
    shutil.copytree(ROOT / "examples" / "trace", path)
    return path


def prepare(dataset, **kwargs):
    return prepare_input(dataset, "rmsnorm_example", "rmsnorm_example_torch_baseline", **kwargs)


def start(dataset, root):
    value = prepare(dataset)
    interaction.init_loop(dataset, root, value=value, budget=2)
    result = interaction.submit(root, root / "draft.ir.json")
    return value, result["candidate"]


@pytest.mark.parametrize("operator", ["rmsnorm", "softmax", "silu_mul"])
def test_native_examples_generate_schema_valid_cuda_solutions(dataset, tmp_path, operator):
    value = prepare_input(dataset, operator + "_example", operator + "_example_torch_baseline")
    root = tmp_path / "session"
    interaction.init_loop(dataset, root, value=value)
    assert (root / "input.json").exists()
    assert read_json(root / "input.json")["definition"] == value.to_dict()["definition"]
    result = interaction.submit(root, root / "draft.ir.json")
    candidate = result["candidate"]
    solution = read_json(root / candidate["solution"])
    validate_native("Solution", solution)
    assert solution["sources"][0]["content"] == (root / candidate["source"]).read_text()
    assert solution["spec"]["binding"] == "torch"
    assert solution["spec"]["destination_passing_style"] is True
    validate_candidate_artifact(root, value.input_id, candidate["compiler_id"], candidate)
    assert result["performance_claim"] is None


def test_definition_argument_order_and_scalar_binding_survive_snapshots(dataset, tmp_path):
    value = prepare(dataset)
    raw = value.to_dict()
    assert raw["input_order"] == ["weight", "x", "eps"]
    snapshot_input(value, dataset, tmp_path / "snapshot")
    stored = read_json(tmp_path / "snapshot" / "input.json")
    assert list(stored["definition"]["inputs"]) == ["weight", "x", "eps"]
    source = compile_trace(seed_program(raw["definition"]), raw["definition"])
    assert (
        "at::Tensor argument_0, at::Tensor argument_1, double argument_2, at::Tensor argument_3"
        in source
    )
    assert "run(argument_1, argument_0, argument_3, argument_2)" in source
    changed = copy.deepcopy(raw["definition"])
    changed["inputs"] = dict(reversed(list(changed["inputs"].items())))
    assert definition_id(changed) != definition_id(raw["definition"])
    stored["definition"] = changed
    with pytest.raises(ValueError, match="order changed"):
        TraceInput.from_dict(stored)


def test_reference_is_preserved_and_changes_input_and_solution_identity(dataset, tmp_path):
    before = prepare(dataset)
    path = dataset / "definitions" / "rmsnorm_example.json"
    definition = read_json(path)
    definition["reference"] = definition["reference"].replace("return ((", "return 7 + ((")
    write_json(path, definition)
    after = prepare(dataset)
    assert after.to_dict()["definition"]["reference"] == definition["reference"]
    assert before.input_id != after.input_id
    assert compile_trace(seed_program(definition), definition) != compile_trace(
        seed_program(before.to_dict()["definition"]), before.to_dict()["definition"]
    )
    # Static interface acceptance does not claim agreement with the changed mathematical function.
    root = tmp_path / "session"
    interaction.init_loop(dataset, root, value=after)
    assert interaction.submit(root, root / "draft.ir.json")["gpu_checked"] is False


def attach_blob(dataset):
    path = dataset / "workloads" / "rmsnorm_example.jsonl"
    traces = [json.loads(line) for line in path.read_text().splitlines()]
    traces[0]["workload"]["inputs"]["weight"] = {
        "type": "safetensors",
        "path": "blob/weights.safetensors",
        "tensor_key": "weight",
    }
    path.write_text("".join(json.dumps(t) + "\n" for t in traces))
    blob = dataset / "blob" / "weights.safetensors"
    blob.parent.mkdir()
    # This test checks byte preservation; native safetensors decoding is checked on the GPU path.
    blob.write_bytes(b"opaque input bytes for snapshot integrity regression")
    return blob


def test_safetensors_are_copied_and_hashed_instead_of_replaced_with_random(dataset, tmp_path):
    blob = attach_blob(dataset)
    value = prepare(dataset)
    root = tmp_path / "session"
    interaction.init_loop(dataset, root, value=value)
    frozen = root / "dataset" / "blob" / "weights.safetensors"
    assert frozen.read_bytes() == blob.read_bytes()
    blob.write_bytes(b"changed original dataset")
    verify_blobs(value, root / "dataset")
    frozen.write_bytes(b"changed frozen input")
    with pytest.raises(ValueError, match="input blob changed"):
        verify_blobs(value, root / "dataset")


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_lfs_inputs_are_refused(dataset, missing):
    blob = attach_blob(dataset)
    if missing:
        blob.unlink()
    else:
        blob.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:123\n")
    with pytest.raises(ValueError, match="missing input|Git LFS"):
        prepare(dataset)


def test_native_configuration_and_explicit_workload_selection(dataset, tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text(
        "num_trials: 4\nop_type_config:\n  rmsnorm:\n    rtol: 0.002\n"
        "definition_config:\n  rmsnorm_example:\n    rtol: 0.003\nrtol: 0.004\n"
    )
    value = prepare(
        dataset, config_path=config, workload_ids=["rmsnorm_example-0", "rmsnorm_example-3"]
    )
    assert value.to_dict()["resolved_config"]["rtol"] == 0.004
    assert len(value.cases("train")) == len(value.cases("holdout")) == 1
    assert value.to_dict()["benchmark"]["trials"] == 4
    config.write_text("num_trials: 1\n")
    with pytest.raises(ValueError, match="four"):
        prepare(dataset, config_path=config)


def test_policy_partition_and_duplicate_configuration_cannot_leak_holdouts(dataset, tmp_path):
    policy = tmp_path / "policy.json"
    write_json(policy, {"train": ["rmsnorm_example-0"], "holdout": ["rmsnorm_example-0"]})
    with pytest.raises(ValueError, match="partition"):
        prepare(dataset, policy_path=policy)
    path = dataset / "workloads" / "rmsnorm_example.jsonl"
    traces = [json.loads(line) for line in path.read_text().splitlines()]
    traces[1]["workload"] = {**traces[0]["workload"], "uuid": "rmsnorm_example-1"}
    path.write_text("".join(json.dumps(t) + "\n" for t in traces))
    with pytest.raises(ValueError, match="duplicate workload configurations"):
        prepare(dataset)


def test_unsupported_computation_is_inspectable_but_not_silently_lowered(dataset, tmp_path):
    path = dataset / "definitions" / "rmsnorm_example.json"
    definition = read_json(path)
    definition["op_type"] = "attention"
    write_json(path, definition)
    assert inspect_dataset(dataset, definition["name"])["definition"]["op_type"] == "attention"
    value = prepare(dataset)
    root = tmp_path / "unsupported"
    interaction.init_loop(dataset, root, value=value)
    result = interaction.submit(root, root / "draft.ir.json")
    assert result["status"] == "rejected"
    assert "UNSUPPORTED_OPERATOR" in result["findings"][0]["message"]
    assert any(a["action"] == "compiler_investigation" for a in result["next_actions"])


def test_actual_cli_uses_trace_and_reports_missing_gpu(dataset, tmp_path, capsys, monkeypatch):
    root = tmp_path / "session"
    assert (
        main(
            [
                "loop",
                "init",
                str(dataset),
                "--definition",
                "rmsnorm_example",
                "--baseline",
                "rmsnorm_example_torch_baseline",
                "--output",
                str(root),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["check", str(root), "--ir", str(root / "draft.ir.json")]) == 0
    assert json.loads(capsys.readouterr().out)["gpu_checked"] is False
    candidate = interaction.submit(root, root / "draft.ir.json")["candidate"]
    # Execute the actual isolated worker where CUDA is absent; do not emulate a speedup.
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        assert main(["loop", "evaluate", str(root), "--candidate", candidate["candidate_id"]]) == 2
        assert json.loads(capsys.readouterr().out)["status"] == "unavailable"
    solution = read_json(root / candidate["solution"])
    solution["sources"][0]["content"] += "\n// changed after submission\n"
    write_json(root / candidate["solution"], solution)
    with pytest.raises(ValueError, match="Solution changed"):
        interaction.loop_status(root)


def test_new_compiler_trial_reuses_native_snapshot_and_policy(dataset, tmp_path, monkeypatch):
    before = tmp_path / "before"
    value, _ = start(dataset, before)
    monkeypatch.setattr(interaction, "evaluation_fingerprint", lambda: "new-revision-test-only")
    after = tmp_path / "after"
    result = interaction.fork_loop(before, after)
    assert result["evaluator_id"] == "new-revision-test-only"
    assert read_json(before / "input.json") == read_json(after / "input.json")
    assert result["input_id"] == value.input_id


def test_native_timer_failure_routes_to_environment_repair(dataset, tmp_path, monkeypatch):
    from entelechy.runtime import timing, worker

    root = tmp_path / "session"
    value, candidate = start(dataset, root)
    job = {
        "input": value.to_dict(),
        "input_id": value.input_id,
        "candidate_id": candidate["candidate_id"],
        "source_sha256": candidate["source_sha256"],
        "source_path": str(root / candidate["source"]),
        "evaluator_id": candidate["evaluator_id"],
        "program": candidate["program"],
        "solution": read_json(root / candidate["solution"]),
        "dataset_root": str(root / "dataset"),
        "split": "train",
        "device": 0,
    }
    api = SimpleNamespace(
        source_identity={},
        evaluator=TraceInput,
        definition=None,
        registry=SimpleNamespace(build=lambda *args: None),
        Solution=SimpleNamespace(model_validate=lambda raw: raw),
    )
    monkeypatch.setattr(worker, "_backend", lambda raw: api)
    monkeypatch.setattr(worker, "doctor", lambda device: {"status": "ok"})
    monkeypatch.setattr(worker, "_environment", lambda *args: {})
    monkeypatch.setattr(
        worker,
        "_load_torch",
        lambda: SimpleNamespace(
            cuda=SimpleNamespace(set_device=lambda device: None), no_grad=contextlib.nullcontext
        ),
    )
    monkeypatch.setattr(timing, "_require_cupti", lambda: (object(), {}))
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: "test-only")

    def timing_failure(*args):
        raise timing.TimingUnavailable("strict CUPTI measurement failed")

    monkeypatch.setattr(worker, "_case", timing_failure)
    result = worker.run_job(job)
    assert all(row["status"] == "unavailable" for row in result["records"])
    assert all(not row["candidate_samples_ms"] for row in result["records"])
    findings = [d for row in result["records"] for d in row["diagnostics"]]
    actions = interaction._feedback(root, {"phase": "search"}, findings, "unused.json")
    assert all(f["repair_scope"] == "environment" for f in findings)
    assert [a["action"] for a in actions] == ["doctor"]


def test_native_loop_export_keeps_solution_and_definition(dataset, tmp_path, monkeypatch):
    root = tmp_path / "session"
    _, candidate = start(dataset, root)
    monkeypatch.setattr(
        interaction, "evaluate_candidate", lambda root, c, k, split, **kw: records(c, k, split)
    )
    interaction.evaluate(root, candidate["candidate_id"])
    interaction.freeze(root)
    interaction.holdout(root)
    destination = tmp_path / "export"
    assert main(["loop", "export", str(root), "--output", str(destination)]) == 0
    definition, workloads, solutions = load_dataset(destination / "dataset", "rmsnorm_example")
    assert list(definition["inputs"]) == ["weight", "x", "eps"]
    assert len(workloads) == 4
    assert any(s["name"].startswith("entelechy_") for s in solutions)


def test_native_evolution_from_rejection_compares_to_baseline(dataset, tmp_path, monkeypatch):
    from test_evolution import fake_tests

    import entelechy
    from entelechy import evolution

    checkout = tmp_path / "checkout"
    shutil.copytree(
        Path(entelechy.__file__).parent,
        checkout / "entelechy",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(ROOT / "examples", checkout / "examples")
    (checkout / "tests").mkdir()
    for name in ("LICENSE", "pyproject.toml"):
        shutil.copyfile(ROOT / name, checkout / name)
    monkeypatch.setattr(evolution, "_checkout", lambda: checkout)
    before = tmp_path / "before"
    interaction.init_loop(dataset, before, value=prepare(dataset), budget=2)
    draft = read_json(before / "draft.ir.json")
    draft["schedule"]["threads"] = 7
    write_json(before / "draft.ir.json", draft)
    rejected = interaction.submit(before, before / "draft.ir.json")
    proposal = tmp_path / "proposal"
    evolution.begin(before, Path(rejected["evidence_path"]), "Investigate rejected IR", proposal)
    path = checkout / "entelechy" / "compiler" / "cuda.py"
    path.write_text(path.read_text() + "\n# Test-only revision identity probe.\n")
    monkeypatch.setattr(evolution, "_run_tests", fake_tests)
    validation = evolution.validate(proposal)
    assert validation["status"] == "static_validated"
    assert validation["next_actions"][0]["argv"][1:3] == ["loop", "fork"]
    assert len(validation["corpus"]) == 3
    after = tmp_path / "after"
    interaction.fork_loop(before, after)
    candidate = interaction.submit(after, after / "draft.ir.json")["candidate"]
    monkeypatch.setattr(
        interaction, "evaluate_candidate", lambda root, c, k, split, **kw: records(c, k, split)
    )
    interaction.evaluate(after, candidate["candidate_id"])
    result = evolution.compare(proposal, after)
    assert result["comparison_kind"] == "new_capability_vs_baseline"
    assert result["observed_geomean_speedup"] == pytest.approx(2.0)
    assert result["automatic_promotion"] is False
    assert all(c["before"]["measurement_origin"] == "after_trial" for c in result["cases"])


@pytest.mark.cuda
@pytest.mark.parametrize("operator", ["rmsnorm", "softmax", "silu_mul"])
def test_native_gpu_solutions_produce_reference_checked_traces(dataset, tmp_path, operator):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    pytest.importorskip("flashinfer_bench")
    policy = tmp_path / "policy.json"
    write_json(policy, {"timing": "cuda_event", "cold_l2": False})
    value = prepare_input(
        dataset, operator + "_example", operator + "_example_torch_baseline", policy_path=policy
    )
    root = tmp_path / "session"
    interaction.init_loop(dataset, root, value=value)
    candidate = interaction.submit(root, root / "draft.ir.json")["candidate"]
    result = interaction.evaluate(root, candidate["candidate_id"])
    assert result["status"] == "evaluated", result
    measured = read_json(Path(result["measurement"]) / "result.json")
    assert len(measured["native_traces"]) == 2 * len(value.cases("train"))
    for trace in measured["native_traces"]:
        validate_native("Trace", trace)
        assert trace["evaluation"]["status"] == "PASSED"
        assert trace["evaluation"]["performance"]["reference_latency_ms"] > 0
    assert all(len(r["baseline_samples_ms"]) == 8 for r in measured["records"])


@pytest.mark.cuda
def test_native_gpu_reference_change_rejects_candidate_before_timing(dataset, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    pytest.importorskip("flashinfer_bench")
    definition_path = dataset / "definitions" / "rmsnorm_example.json"
    definition = read_json(definition_path)
    definition["reference"] = definition["reference"].replace("return ((", "return 7 + ((")
    write_json(definition_path, definition)
    # Keep the selected baseline valid so the failure belongs to the generated candidate.
    baseline_path = dataset / "solutions" / "rmsnorm_example.json"
    baseline = read_json(baseline_path)
    baseline["sources"][0]["content"] = baseline["sources"][0]["content"].replace(
        "return torch.nn.functional.rms_norm", "return 7 + torch.nn.functional.rms_norm"
    )
    write_json(baseline_path, baseline)
    policy = tmp_path / "policy.json"
    write_json(policy, {"timing": "cuda_event", "cold_l2": False})
    value = prepare(dataset, policy_path=policy)
    root = tmp_path / "session"
    interaction.init_loop(dataset, root, value=value)
    candidate = interaction.submit(root, root / "draft.ir.json")["candidate"]
    result = interaction.evaluate(root, candidate["candidate_id"])
    assert result["status"] == "failed"
    rows = read_json(root / "search_results.json")
    assert all(r["status"] == "invalid" and not r["candidate_samples_ms"] for r in rows)
