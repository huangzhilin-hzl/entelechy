# SPDX-License-Identifier: Apache-2.0
"""Native worker protocol, evaluation routing and strict GPU timing tests."""

import contextlib
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from entelechy.artifacts import evaluation_fingerprint, source_hash
from entelechy.compiler.trace import compile_trace, seed_program, solution_for
from entelechy.runtime import timing, worker
from entelechy.trace_input import prepare_input, snapshot_input

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job(tmp_path):
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"timing": "cuda_event", "cold_l2": False}))
    value = prepare_input(
        ROOT / "examples" / "trace",
        "softmax_example",
        "softmax_example_torch_baseline",
        policy_path=policy,
    )
    snapshot_input(value, ROOT / "examples" / "trace", tmp_path)
    raw = value.to_dict()
    program = seed_program(raw["definition"])
    source = compile_trace(program, raw["definition"])
    (tmp_path / "kernel.cu").write_text(source)
    digest = source_hash(source)
    return {
        "input": raw,
        "input_id": value.input_id,
        "program": program,
        "solution": solution_for(source, raw["definition"], target_sm=raw["target_sm"]),
        "split": "train",
        "source_path": str(tmp_path / "kernel.cu"),
        "dataset_root": str(tmp_path / "dataset"),
        "output_path": str(tmp_path / "result.json"),
        "device": 0,
        "source_sha256": digest,
        "candidate_id": digest[:24],
        "evaluator_id": evaluation_fingerprint(),
    }


def expected_cases(job):
    return [case["id"] for case in job["input"]["cases"] if case["split"] == job["split"]]


def fake_backend(monkeypatch, *, build=None):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(set_device=lambda device: None), no_grad=contextlib.nullcontext
    )
    api = SimpleNamespace(
        evaluator=SimpleNamespace,
        definition=None,
        registry=SimpleNamespace(build=build or (lambda *args: object())),
        Solution=SimpleNamespace(model_validate=lambda raw: raw),
        source_identity={"flashinfer_bench_source_sha256": "test-only"},
    )
    monkeypatch.setattr(worker, "doctor", lambda device: {"status": "ok", "sm": 90})
    monkeypatch.setattr(worker, "_load_torch", lambda: torch)
    monkeypatch.setattr(worker, "_backend", lambda raw: api)
    monkeypatch.setattr(worker.importlib.metadata, "version", lambda name: "test-only")
    return api


def test_import_does_not_import_torch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import entelechy.runtime.worker; assert 'torch' not in sys.modules",
        ],
        check=True,
    )


def test_doctor_missing_torch(monkeypatch):
    def fail():
        raise ImportError("test missing torch")

    monkeypatch.setattr(worker, "_load_torch", fail)
    report = worker.doctor()
    assert report["status"] == "unavailable"
    assert "test missing torch" in report["diagnostics"][0]["message"]


def test_unavailable_covers_every_split_case(monkeypatch, job):
    monkeypatch.setattr(
        worker, "doctor", lambda device: {"status": "unavailable", "diagnostics": []}
    )
    result = worker.run_job(job)
    assert result["status"] == "unavailable"
    assert [record["case_id"] for record in result["records"]] == expected_cases(job)
    assert all(not r["correct"] and not r["candidate_samples_ms"] for r in result["records"])
    assert all(r["environment"] and r["input_id"] == job["input_id"] for r in result["records"])
    assert all(r["baseline_id"].startswith("flashinfer-solution:") for r in result["records"])


@pytest.mark.parametrize("field", ["input_id", "source_sha256", "candidate_id", "evaluator_id"])
def test_mismatched_identity_is_rejected_before_cuda_probe(monkeypatch, job, field):
    job[field] = "0" * (24 if field == "candidate_id" else 64)
    monkeypatch.setattr(
        worker, "doctor", lambda device: pytest.fail("invalid identity probed CUDA")
    )
    result = worker.run_job(job)
    assert result["status"] == "invalid"
    assert len(result["records"]) == len(expected_cases(job))
    assert all(not r["candidate_samples_ms"] for r in result["records"])


def test_solution_and_source_must_match_frozen_program(job):
    job["solution"]["sources"][0]["content"] += "\n// changed"
    result = worker.run_job(job)
    assert result["status"] == "invalid"
    assert "Solution identity mismatch" in result["diagnostics"][0]["message"]


def test_native_builder_consumes_verified_solution_snapshot(monkeypatch, job):
    def build(definition, solution):
        Path(job["source_path"]).write_text("// modified after validation\n")
        if solution["name"] == job["solution"]["name"]:
            assert source_hash(solution["sources"][0]["content"]) == job["source_sha256"]
        return object()

    fake_backend(monkeypatch, build=build)
    monkeypatch.setattr(
        worker,
        "_case",
        lambda torch, api, raw, job, case, a, b, env, timer, flush: (
            {**worker._record(job, case, env), "status": "ok"},
            [],
        ),
    )
    assert worker.run_job(job)["status"] == "ok"


def test_unsupported_timing_is_rejected(job):
    job["input"]["benchmark"]["timing"] = "cpu_time"
    result = worker.run_job(job)
    assert result["status"] == "invalid"


def test_candidate_compile_error_covers_every_case(monkeypatch, job):
    def build(definition, solution):
        if solution["name"] == job["solution"]["name"]:
            raise RuntimeError("test compiler failure")
        return object()

    fake_backend(monkeypatch, build=build)
    result = worker.run_job(job)
    assert result["status"] == "compile_error"
    assert [r["case_id"] for r in result["records"]] == expected_cases(job)


def test_baseline_build_failure_is_environment_unavailable(monkeypatch, job):
    def build(*args):
        raise RuntimeError("selected baseline cannot build")

    fake_backend(monkeypatch, build=build)
    result = worker.run_job(job)
    assert result["status"] == "unavailable"
    assert result["diagnostics"][0]["code"] == "baseline_unavailable"


def test_baseline_fallback_builds_only_selected_native_solution(monkeypatch, job):
    job["candidate_id"] = "baseline"
    seen = []

    def build(definition, solution):
        seen.append(solution["name"])
        return object()

    fake_backend(monkeypatch, build=build)

    def run_case(torch, api, raw, job, case, baseline, candidate, env, timer, flush):
        assert baseline is candidate
        return {**worker._record(job, case, env), "status": "ok"}, []

    monkeypatch.setattr(worker, "_case", run_case)
    result = worker.run_job(job)
    assert result["status"] == "ok"
    assert seen == [job["input"]["baseline_solution"]["name"]]


@pytest.mark.parametrize(
    "error,status",
    [
        (RuntimeError("CUDA failure"), "runtime_error"),
        (timing.TimingUnavailable("strict CUPTI failure"), "unavailable"),
    ],
)
def test_failed_context_is_not_reused_for_later_workloads(monkeypatch, job, error, status):
    fake_backend(monkeypatch)
    calls = []

    def failed_case(*args):
        calls.append(1)
        raise error

    monkeypatch.setattr(worker, "_case", failed_case)
    result = worker.run_job(job)
    assert calls == [1]
    assert [r["case_id"] for r in result["records"]] == expected_cases(job)
    assert all(r["status"] == status and not r["candidate_samples_ms"] for r in result["records"])


@pytest.mark.parametrize(
    "failed_index,status,code",
    [(0, "unavailable", "baseline_unavailable"), (1, "invalid", "INCORRECT_NUMERICAL")],
)
def test_reference_failure_identifies_baseline_or_candidate(
    monkeypatch, job, failed_index, status, code
):
    raw = job["input"]
    torch = SimpleNamespace(
        manual_seed=lambda seed: None, cuda=SimpleNamespace(manual_seed_all=lambda seed: None)
    )
    api = SimpleNamespace(
        definition=None,
        cfg=SimpleNamespace(model_copy=lambda **kwargs: None),
        evaluator=SimpleNamespace(build_baseline=lambda *args: object()),
        Workload=SimpleNamespace(model_validate=lambda raw: raw),
    )
    failure = SimpleNamespace(status=SimpleNamespace(value="INCORRECT_NUMERICAL"))
    checks = iter([(None, None)] * failed_index + [(None, failure)])
    monkeypatch.setattr(worker, "_check", lambda *args: next(checks))
    monkeypatch.setattr(worker, "_native", lambda *args, **kwargs: {"failed": True})
    monkeypatch.setattr(
        timing, "paired_benchmark", lambda *args: pytest.fail("incorrect output timed")
    )
    case = next(case for case in raw["cases"] if case["split"] == "train")
    row, traces = worker._case(torch, api, raw, job, case, object(), object(), {}, None, None)
    assert row["status"] == status and row["diagnostics"][0]["code"] == code
    assert traces == [{"failed": True}] and not row["candidate_samples_ms"]


def test_upstream_failed_evaluation_stops_later_workloads(monkeypatch, job):
    fake_backend(monkeypatch)
    calls = []

    def failed_case(torch, api, raw, job, case, a, b, env, timer, flush):
        calls.append(case["id"])
        return {**worker._record(job, case, env), "status": "invalid"}, []

    monkeypatch.setattr(worker, "_case", failed_case)
    result = worker.run_job(job)
    assert calls == expected_cases(job)[:1]
    assert [r["case_id"] for r in result["records"]] == expected_cases(job)
    assert all(r["status"] == "invalid" for r in result["records"])


def test_backend_identity_covers_builder_and_editable_cuda_sources(tmp_path):
    root = tmp_path / "flashinfer"
    root.mkdir()
    (root / "builder.py").write_text("version = 1")
    cuda = tmp_path / "csrc"
    cuda.mkdir()
    (cuda / "kernel.cu").write_text("// original")
    first = worker._source_tree_hash(root)
    (root / "builder.py").write_text("version = 2")
    second = worker._source_tree_hash(root)
    (cuda / "kernel.cu").write_text("// changed")
    assert len({first, second, worker._source_tree_hash(root)}) == 3


def test_main_writes_machine_readable_unavailable(monkeypatch, tmp_path, job):
    monkeypatch.setattr(
        worker, "doctor", lambda device: {"status": "unavailable", "diagnostics": []}
    )
    path = tmp_path / "job.json"
    path.write_text(json.dumps(job))
    assert worker.main(["--job", str(path)]) == 2
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "unavailable"
    assert len(result["records"]) == len(expected_cases(job))


def test_abba_uses_real_independent_paired_measurements(monkeypatch):
    calls = []

    def baseline():
        calls.append("A")

    def candidate():
        calls.append("B")

    timings = iter([10, 4, 6, 14, 20, 8, 10, 24])

    def measure(torch, run, repeats, flush):
        run()
        return next(timings)

    monkeypatch.setattr(timing, "_timed", measure)
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    a, b = timing.paired_benchmark(
        torch, baseline, candidate, {"warmup": 1, "trials": 2, "repeats": 5}
    )
    assert calls == list("ABABBAABBA")
    assert a == [12, 22] and b == [5, 9]


def test_cold_l2_flush_is_before_every_timed_invocation():
    calls = []

    class Event:
        def __init__(self, **kwargs):
            self.name = "start" if not calls else "end"
            calls.append("create")

        def record(self):
            calls.append(self.name)

        def synchronize(self):
            calls.append("sync")

        def elapsed_time(self, other):
            return 2.0

    flush = SimpleNamespace(add_=lambda value: calls.append("flush"))
    torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event))
    value = timing._timed(torch, lambda: calls.append("run"), 3, flush)
    assert calls[2:] == ["flush", "start", "run", "end", "sync"] * 3
    assert value == 2.0


def test_strict_cupti_uses_verified_signature_and_measured_samples():
    seen = {}

    def timer(**kwargs):
        seen.update(kwargs)
        return [1.0, 3.0]

    run = lambda: None
    assert timing._timed_cupti(timer, run, 2, True) == 2.0
    assert seen == {
        "fn": run,
        "dry_run_iters": 0,
        "repeat_iters": 2,
        "use_cuda_graph": True,
        "cold_l2_cache": True,
    }


def test_strict_cupti_rejects_fallback_warning():
    def timer(**kwargs):
        warnings.warn("Falling back to CUDA events", UserWarning, stacklevel=2)
        return [1.0]

    with pytest.raises(timing.TimingUnavailable, match="Falling back"):
        timing._timed_cupti(timer, lambda: None, 1, False)


def test_strict_cupti_rejects_silent_known_fallback():
    namespace = {"bench_gpu_time_with_cudagraph": lambda **kwargs: [1.0]}
    exec("def timer(**kwargs):\n    return bench_gpu_time_with_cudagraph(**kwargs)\n", namespace)
    original = namespace["bench_gpu_time_with_cudagraph"]
    with pytest.raises(timing.TimingUnavailable, match="substitution"):
        timing._timed_cupti(namespace["timer"], lambda: None, 1, False)
    assert namespace["bench_gpu_time_with_cudagraph"] is original


@pytest.mark.parametrize("failure_kind", ["guard", "mutation"])
def test_native_correctness_wrapper_guards_outputs_and_inputs(monkeypatch, failure_kind):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    x = torch.ones((2, 17))
    baseline = SimpleNamespace(inputs=[[x]], outputs=[[x.clone()]])

    class Runnable:
        metadata = SimpleNamespace(destination_passing_style=True)

        def __call__(self, value, out):
            out.copy_(value)
            if failure_kind == "mutation":
                value[0, 0] = 2
            else:
                # The wrapper supplies an allocation slice; escape into its prefix guard.
                out.as_strided((out.numel() + 32,), (1,), storage_offset=0)[0] = 0

    class Evaluator:
        @staticmethod
        def check_correctness(definition, run, inputs, outputs, cfg, log, device):
            run(*inputs[0], torch.empty_like(x))
            return None, None

    api = SimpleNamespace(
        definition=SimpleNamespace(inputs={"x": {}}), evaluator=Evaluator, cfg=None
    )
    with pytest.raises(RuntimeError, match="guard corruption|mutated"):
        worker._check(torch, api, Runnable(), baseline, "cpu")
    assert torch.equal(x, torch.ones_like(x))


@pytest.mark.cuda
def test_real_cuda_native_baseline_fallback(job):
    torch = pytest.importorskip("torch")
    pytest.importorskip("flashinfer_bench")
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    job["candidate_id"] = "baseline"
    result = worker.run_job(job)
    assert result["status"] == "ok", result
    for record in result["records"]:
        assert record["correct"]
        assert len(record["baseline_samples_ms"]) == job["input"]["benchmark"]["trials"]
        assert len(record["candidate_samples_ms"]) == len(record["baseline_samples_ms"])
