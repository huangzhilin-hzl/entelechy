# SPDX-License-Identifier: Apache-2.0
"""Evaluate one native FlashInfer Trace candidate in an isolated GPU process.

Usage: python -m entelechy.runtime.worker --job /absolute/job.json
The native reference defines correctness; paired measurements use an explicit
baseline Solution. Missing hardware never produces synthesized measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..artifacts import canonical_hash, evaluation_fingerprint, read_json, write_json
from ..compiler.trace import compile_trace, solution_for
from ..trace_input import TraceInput, verify_blobs
from . import timing


class JobIdentityError(ValueError):
    """A workload, source, or evaluator no longer matches the submitted job."""


def _load_torch() -> Any:
    return importlib.import_module("torch")


def doctor(device: int = 0) -> dict:
    """Return a JSON-safe capability report, including import failures."""
    report: dict = {"status": "unavailable", "device": device, "diagnostics": []}
    try:
        torch = _load_torch()
        report["torch"] = str(torch.__version__)
        report["torch_git_version"] = getattr(torch.version, "git_version", None)
        report["cuda"] = getattr(torch.version, "cuda", None)
        if not torch.cuda.is_available():
            report["diagnostics"].append(
                {"code": "cuda_unavailable", "path": "device", "message": "CUDA is not available"}
            )
            return report
        if device < 0 or device >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device} does not exist")
        props = torch.cuda.get_device_properties(device)
        report.update(
            status="ok",
            GPU_name=str(props.name),
            sm=int(props.major) * 10 + int(props.minor),
            uuid=str(getattr(props, "uuid", "unavailable")),
            total_memory=int(props.total_memory),
            multiprocessor_count=int(props.multi_processor_count),
        )
    except Exception as error:
        report["diagnostics"].append(
            {
                "code": "cuda_probe_failed",
                "path": "device",
                "message": f"{type(error).__name__}: {error}",
            }
        )
    return report


def _environment(report: dict, raw: dict) -> dict:
    return {
        key: report.get(key)
        for key in (
            "GPU_name",
            "sm",
            "uuid",
            "torch",
            "torch_git_version",
            "cuda",
            "multiprocessor_count",
        )
    } | {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": report.get("device"),
        "baseline_solution_sha256": canonical_hash(raw["baseline_solution"]),
        "timing": raw.get("benchmark", {}).get("timing", "cuda_event"),
        "cold_l2": raw.get("benchmark", {}).get("cold_l2", False),
        # Settings that can change native FlashInfer dispatch must participate
        # in the evidence identity rather than being silently inherited.
        "backend_environment": {
            key: os.environ.get(key)
            for key in (
                "FLASHINFER_USE_CUDA_NORM",
                "FLASHINFER_CUDA_ARCH_LIST",
                "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS",
                "CUDA_VISIBLE_DEVICES",
                "TORCH_CUDA_ARCH_LIST",
            )
        },
    }


def _record(job: dict, case: dict, environment: dict) -> dict:
    return {
        "input_id": job.get("input_id"),
        "candidate_id": job.get("candidate_id"),
        "source_sha256": job.get("source_sha256"),
        "evaluator_id": job.get("evaluator_id"),
        "case_id": case.get("id"),
        "split": job.get("split"),
        "environment": dict(environment),
        "baseline_id": "flashinfer-solution:"
        + canonical_hash(job.get("input", {}).get("baseline_solution")),
        "status": "runtime_error",
        "correct": False,
        "baseline_samples_ms": [],
        "candidate_samples_ms": [],
        "diagnostics": [],
    }


def failure_result(
    job: dict, status: str, code: str, message: str, environment: dict | None = None
) -> dict:
    """Represent failures for every requested case; missing rows are never success."""
    records = []
    for case in job.get("input", {}).get("cases", []):
        if case.get("split") != job.get("split"):
            continue
        record = _record(job, case, environment or {})
        record.update(status=status)
        record["diagnostics"].append({"code": code, "path": "job", "message": message})
        records.append(record)
    return {
        "schema_version": 1,
        "status": status,
        "records": records,
        "diagnostics": [{"code": code, "path": "job", "message": message}],
    }


def _verified_source(job: dict) -> bytes:
    expected = job.get("source_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise JobIdentityError("source_sha256 must be a full lowercase SHA-256")
    if job.get("candidate_id") not in ("baseline", expected[:24]):
        raise JobIdentityError("candidate_id must be the first 24 source SHA-256 characters")
    try:
        content = Path(job["source_path"]).read_bytes()
    except OSError as error:
        raise JobIdentityError(f"cannot read submitted source: {error}") from error
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise JobIdentityError(f"source SHA-256 mismatch: expected {expected}, got {actual}")
    return content


def _source_tree_hash(root: Path) -> str:
    """Identify installed backend sources, including editable FlashInfer CUDA trees."""
    roots = {"python": root}
    for name in ("csrc", "include"):
        for candidate in (root / "data" / name, root.parent / name):
            if candidate.is_dir():
                roots[name] = candidate.resolve()
                break
    suffixes = {".py", ".cu", ".cuh", ".h", ".hpp", ".jinja", ".inc", ".json"}
    digest = hashlib.sha256()
    for label, source_root in sorted(roots.items()):
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts:
                digest.update(
                    f"{label}/{path.relative_to(source_root).as_posix()}".encode() + b"\0"
                )
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _backend(raw: dict):
    import flashinfer
    import flashinfer_bench
    from flashinfer_bench.bench import BenchmarkConfig
    from flashinfer_bench.bench.evaluators.default import DefaultEvaluator
    from flashinfer_bench.bench.evaluators.registry import resolve_evaluator
    from flashinfer_bench.bench.evaluators.utils import allocate_outputs
    from flashinfer_bench.compile import BuilderRegistry
    from flashinfer_bench.data import Definition, Evaluation, Solution, Trace, Workload

    definition = Definition.model_validate(raw["definition"])
    cfg = BenchmarkConfig.model_validate(raw["benchmark_config"]).resolve_eval_config(definition)
    if cfg.model_dump() != raw["resolved_config"]:
        raise RuntimeError("installed flashinfer-bench resolves BenchmarkConfig differently")
    if resolve_evaluator(definition) is not DefaultEvaluator:
        raise RuntimeError("this compiler currently supports only the default row evaluator")
    return SimpleNamespace(
        definition=definition,
        cfg=cfg,
        evaluator=DefaultEvaluator,
        registry=BuilderRegistry.get_instance(),
        Solution=Solution,
        Workload=Workload,
        Evaluation=Evaluation,
        Trace=Trace,
        allocate_outputs=allocate_outputs,
        source_identity={
            "flashinfer_bench_source_sha256": _source_tree_hash(
                Path(flashinfer_bench.__file__).resolve().parent
            ),
            "flashinfer_source_sha256": _source_tree_hash(
                Path(flashinfer.__file__).resolve().parent
            ),
        },
    )


def _clone(torch, values):
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in values]


def _check(torch, api, runnable, baseline, device: str):
    inputs = [_clone(torch, inp) for inp in baseline.inputs]
    originals = [_clone(torch, inp) for inp in inputs]

    class Guarded:
        metadata = runnable.metadata

        def __call__(self, *args):
            if not self.metadata.destination_passing_style:
                return runnable(*args)
            count = len(api.definition.inputs)
            pads, views = [], []
            for output in args[count:]:
                pad = torch.full(
                    (output.numel() + 64,), 937.0, dtype=output.dtype, device=output.device
                )
                view = pad[32:-32].view(output.shape)
                view.fill_(float("nan"))
                pads.append(pad)
                views.append(view)
            runnable(*args[:count], *views)
            torch.cuda.synchronize(device)
            for pad, view, output in zip(pads, views, args[count:], strict=True):
                if not bool((pad[:32] == 937).all()) or not bool((pad[-32:] == 937).all()):
                    raise RuntimeError("output guard corruption")
                output.copy_(view)

    correctness, failure = api.evaluator.check_correctness(
        api.definition, Guarded(), inputs, baseline.outputs, api.cfg, "", device
    )
    for before, after in zip(originals, inputs, strict=True):
        for original, value in zip(before, after, strict=True):
            if isinstance(original, torch.Tensor) and not torch.equal(
                original.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8)
            ):
                raise RuntimeError("solution mutated a Definition input")
    return correctness, failure


def _native(
    api,
    raw,
    workload,
    solution_name,
    environment,
    *,
    correctness=None,
    latency=None,
    reference=None,
    failure=None,
):
    payload = {
        "status": "PASSED",
        "environment": {
            "hardware": environment["GPU_name"],
            "libs": {
                "flashinfer_bench": environment["flashinfer_bench"],
                "torch": environment["torch"],
                "cuda": str(environment["cuda"]),
                "entelechy_evaluator": environment["entelechy_evaluator"],
            },
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "log": "",
        "correctness": None,
        "performance": None,
    }
    if failure is not None:
        payload.update(json.loads(failure.model_dump_json()))
    else:
        payload["correctness"] = json.loads(correctness.model_dump_json())
        payload["performance"] = {
            "latency_ms": latency,
            "reference_latency_ms": reference,
            "speedup_factor": reference / latency,
        }
    evaluation = api.Evaluation.model_validate(payload)
    trace = api.Trace(
        definition=raw["definition"]["name"],
        workload=api.Workload.model_validate(workload),
        solution=solution_name,
        evaluation=evaluation,
    )
    return json.loads(trace.model_dump_json())


def _case(torch, api, raw, job, case, baseline_run, candidate_run, environment, timer, flush):
    record = _record(job, case, environment)
    record["baseline_id"] = "flashinfer-solution:" + canonical_hash(raw["baseline_solution"])
    workload = next(t["workload"] for t in raw["workloads"] if t["workload"]["uuid"] == case["id"])
    seed = int(hashlib.sha256(f"{raw['seed']}:{case['id']}".encode()).hexdigest()[:8], 16)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = f"cuda:{job['device']}"
    baseline = api.evaluator.build_baseline(
        api.definition,
        api.Workload.model_validate(workload),
        api.cfg.model_copy(update={"profile_baseline": False}),
        device,
        Path(job["dataset_root"]),
    )
    native = []
    observations = []
    names = [raw["baseline_solution"]["name"], job["solution"]["name"]]
    if job["candidate_id"] == "baseline":
        names[1] = names[0]
    for index, (label, runnable) in enumerate(
        zip(names, (baseline_run, candidate_run), strict=True)
    ):
        correctness, failure = _check(torch, api, runnable, baseline, device)
        if failure is not None:
            native.append(_native(api, raw, workload, label, environment, failure=failure))
            record.update(
                status="unavailable" if index == 0 else "invalid",
                diagnostics=[
                    {
                        "code": "baseline_unavailable" if index == 0 else failure.status.value,
                        "path": f"workload.{case['id']}.solution.{label}",
                        "message": (
                            "FlashInfer reference agreement failed; see native_traces and log"
                        ),
                    }
                ],
            )
            return record, native
        observations.append(correctness)

    reference_run = api.registry.build_reference(api.definition)
    a_samples, b_samples, ref_samples = [], [], []
    for inputs in baseline.inputs:

        def closure(runnable, inputs=inputs):
            values = _clone(torch, inputs)
            outputs = (
                api.allocate_outputs(api.definition, values, device)
                if runnable.metadata.destination_passing_style
                else []
            )
            return lambda: runnable(*values, *outputs)

        a, b = timing.paired_benchmark(
            torch,
            closure(baseline_run),
            closure(candidate_run),
            {**raw["benchmark"], "trials": 1},
            flush,
            timer,
        )
        a_samples.extend(a)
        b_samples.extend(b)
        reference = closure(reference_run)
        if timer is not None:
            ref_samples.append(
                timing._timed_cupti(
                    timer, reference, raw["benchmark"]["repeats"], raw["benchmark"]["cold_l2"]
                )
            )
        else:
            ref_samples.append(timing._timed(torch, reference, raw["benchmark"]["repeats"], flush))
    record.update(
        status="ok",
        correct=True,
        baseline_samples_ms=a_samples,
        candidate_samples_ms=b_samples,
        reference_samples_ms=ref_samples,
        kernel_qualified=timer is not None,
        sample_unit="ABBA trial on the same native Workload inputs; trials retain generation order",
        correctness_checks=len(baseline.inputs),
        timing_scope="selected Solution GPU activity; no production dispatcher",
        workload_uuid=case["id"],
    )
    for label, correctness, samples in zip(
        names, observations, (a_samples, b_samples), strict=True
    ):
        native.append(
            _native(
                api,
                raw,
                workload,
                label,
                environment,
                correctness=correctness,
                latency=statistics.fmean(samples),
                reference=statistics.fmean(ref_samples),
            )
        )
    return record, native


def _validate_job(job: dict) -> TraceInput:
    if job.get("split") not in {"train", "holdout"}:
        raise ValueError("split must be train or holdout")
    if type(job.get("device")) is not int or job["device"] < 0:
        raise ValueError("device must be a nonnegative integer")
    for key in ("source_path", "dataset_root"):
        if not Path(job[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    value = TraceInput.from_dict(job["input"])
    if not value.cases(job["split"]):
        raise ValueError("requested split has no workloads")
    raw = value.to_dict()
    if value.input_id != job.get("input_id") or job.get("evaluator_id") != evaluation_fingerprint():
        raise JobIdentityError("Trace input or evaluator identity mismatch")
    source = _verified_source(job).decode()
    if source != compile_trace(job["program"], raw["definition"]):
        raise JobIdentityError("Trace program does not match source")
    if job["solution"] != solution_for(source, raw["definition"], target_sm=raw["target_sm"]):
        raise JobIdentityError("native Solution identity mismatch")
    verify_blobs(value, Path(job["dataset_root"]))
    return value


def run_job(job: dict) -> dict:
    """Run one frozen native input split and retain a record for every workload."""
    try:
        value = _validate_job(job)
    except (KeyError, ValueError, TypeError, OSError) as error:
        return failure_result(job, "invalid", "invalid_job", str(error))
    raw = value.to_dict()
    report = doctor(job["device"])
    if report["status"] != "ok":
        return failure_result(
            job,
            "unavailable",
            "cuda_unavailable",
            str(report["diagnostics"]),
            _environment(report, raw),
        )
    if raw["target_sm"] is not None and raw["target_sm"] != report["sm"]:
        return failure_result(
            job, "unavailable", "target_sm_mismatch", "device does not match target_sm"
        )
    torch = _load_torch()
    torch.cuda.set_device(job["device"])
    timer, timer_metadata = None, {}
    try:
        api = _backend(raw)
        if raw["benchmark"]["timing"] == "cupti":
            timer, timer_metadata = timing._require_cupti()
    except Exception as error:
        return failure_result(job, "unavailable", "trace_backend_unavailable", str(error))
    environment = (
        _environment(report, raw)
        | timer_metadata
        | api.source_identity
        | {
            "flashinfer_bench": importlib.metadata.version("flashinfer-bench"),
        }
    )
    # Compiler revisions are compared separately; implementation identity is already on each row.
    native_environment = environment | {"entelechy_evaluator": evaluation_fingerprint()}
    try:
        baseline_run = api.registry.build(
            api.definition, api.Solution.model_validate(raw["baseline_solution"])
        )
    except Exception as error:
        return failure_result(job, "unavailable", "baseline_unavailable", str(error), environment)
    try:
        candidate_run = (
            baseline_run
            if job["candidate_id"] == "baseline"
            else api.registry.build(api.definition, api.Solution.model_validate(job["solution"]))
        )
    except Exception as error:
        return failure_result(job, "compile_error", "COMPILE_ERROR", str(error), environment)
    flush = None
    try:
        if timer is None and raw["benchmark"]["cold_l2"]:
            flush = torch.empty(
                timing._l2_cache_bytes(torch, job["device"]) * 2,
                dtype=torch.uint8,
                device=f"cuda:{job['device']}",
            )
    except Exception as error:
        return failure_result(job, "unavailable", "l2_flush_unavailable", str(error), environment)
    rows, native = [], []
    cases = value.cases(job["split"])
    for index, case in enumerate(cases):
        try:
            with torch.no_grad():
                row, traces = _case(
                    torch,
                    api,
                    raw,
                    job,
                    case,
                    baseline_run,
                    candidate_run,
                    native_environment,
                    timer,
                    flush,
                )
            row["environment"].pop("entelechy_evaluator", None)
            rows.append(row)
            native.extend(traces)
            if row["status"] != "ok":
                # An upstream failure can include a CUDA runtime error. Stop
                # after the first failed workload and preserve explicit coverage.
                for remaining in cases[index + 1 :]:
                    skipped = _record(job, remaining, environment)
                    skipped.update(status=row["status"], diagnostics=row["diagnostics"])
                    rows.append(skipped)
                break
        except Exception as error:
            status = (
                "unavailable" if isinstance(error, timing.TimingUnavailable) else "runtime_error"
            )
            # CUDA execution failures can poison the context. Preserve coverage
            # without attempting another workload in the same worker process.
            for remaining in cases[index:]:
                row = _record(job, remaining, environment)
                row["status"] = status
                row["diagnostics"] = [
                    {
                        "code": "trace_timing_unavailable"
                        if status == "unavailable"
                        else "TRACE_EXECUTION",
                        "path": f"workload.{case['id']}",
                        "message": str(error),
                    }
                ]
                rows.append(row)
            break
    verify_blobs(value, Path(job["dataset_root"]))
    return {
        "schema_version": 1,
        "status": "ok" if all(r["status"] == "ok" for r in rows) else "failed",
        "records": rows,
        "native_traces": native,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        job = read_json(args.job)
        result = run_job(job)
        write_json(Path(job["output_path"]), result)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "runtime_error",
                    "diagnostics": [
                        {
                            "code": "worker_protocol_error",
                            "path": "job",
                            "message": f"{type(error).__name__}: {error}",
                        }
                    ],
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_path": job["output_path"],
                "records": len(result["records"]),
            }
        )
    )
    return 0 if result["status"] == "ok" else (2 if result["status"] == "unavailable" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
