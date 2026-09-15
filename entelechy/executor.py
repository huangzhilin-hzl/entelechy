# SPDX-License-Identifier: Apache-2.0
"""Process-isolated GPU jobs with source identities, deadlines and an exclusive lease."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import (
    canonical_hash,
    compiler_fingerprint,
    evaluation_fingerprint,
    read_json,
    validate_candidate_artifact,
    write_json,
)
from .trace_input import TraceInput, verify_blobs


def _failed_records(
    value: TraceInput,
    candidate: dict[str, Any],
    split: str,
    status: str,
    message: str,
) -> list[dict[str, Any]]:
    return [
        {
            "input_id": value.input_id,
            "candidate_id": candidate["candidate_id"],
            "source_sha256": candidate["source_sha256"],
            "evaluator_id": candidate["evaluator_id"],
            "case_id": case["id"],
            "split": split,
            "environment": {},
            "baseline_id": "flashinfer-solution:"
            + canonical_hash(value.to_dict()["baseline_solution"]),
            "status": status,
            "correct": False,
            "baseline_samples_ms": [],
            "candidate_samples_ms": [],
            "diagnostics": [{"code": status, "path": "worker", "message": message}],
        }
        for case in value.cases(split)
    ]


@contextlib.contextmanager
def gpu_lease(device: int):
    """Host-local advisory lease. Separate users/containers still need a cluster scheduler."""
    import fcntl

    root = Path.home() / ".cache" / "entelechy"
    root.mkdir(parents=True, exist_ok=True)
    # Serialize local workers even when CUDA_VISIBLE_DEVICES remaps ordinal numbers.
    with (root / "gpu-worker.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another Entelechy GPU worker owns this host lease") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def evaluate_candidate(
    run_dir: Path,
    value: TraceInput,
    candidate: dict[str, Any],
    split: str,
    *,
    device: int = 0,
    timeout_s: float = 300,
    attempt: int | None = None,
) -> list[dict[str, Any]]:
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if split not in {"train", "holdout"}:
        raise ValueError("invalid evaluation split")
    if candidate["input_id"] != value.input_id:
        raise ValueError("candidate and value identity mismatch")
    if candidate["compiler_id"] != compiler_fingerprint():
        raise ValueError("compiler changed; create a new experiment instead of mixing versions")
    if candidate.get("evaluator_id") != evaluation_fingerprint():
        raise ValueError("evaluation implementation changed after planning")
    source_path = validate_candidate_artifact(
        run_dir,
        value.input_id,
        compiler_fingerprint(),
        candidate,
        allow_baseline=candidate["candidate_id"] == "baseline",
    )
    job_dir = run_dir / "evaluations" / candidate["candidate_id"] / split
    if attempt is not None:
        if type(attempt) is not int or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        job_dir = job_dir / str(attempt)
    job_dir.mkdir(parents=True, exist_ok=False)
    output = job_dir / "result.json"
    job = {
        "input": value.to_dict(),
        "input_id": value.input_id,
        "candidate_id": candidate["candidate_id"],
        "source_sha256": candidate["source_sha256"],
        "evaluator_id": candidate["evaluator_id"],
        "split": split,
        "source_path": str(source_path),
        "output_path": str(output.resolve()),
        "device": device,
    }
    verify_blobs(value, run_dir / "dataset")
    job.update(
        program=candidate["program"],
        solution=read_json(run_dir / candidate["solution"]),
        dataset_root=str((run_dir / "dataset").resolve()),
    )
    job_path = job_dir / "job.json"
    write_json(job_path, job)
    with gpu_lease(device), (job_dir / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "entelechy.runtime.worker", "--job", str(job_path.resolve())],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            records = _failed_records(value, candidate, split, "timeout", f"exceeded {timeout_s}s")
            write_json(output, {"status": "timeout", "records": records})
            return records
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    if returncode < 0:
        records = _failed_records(
            value, candidate, split, "runtime_error", f"worker signal {-returncode}"
        )
        write_json(output, {"status": "runtime_error", "records": records})
        return records
    if not output.exists():
        records = _failed_records(
            value, candidate, split, "runtime_error", f"worker exit {returncode}"
        )
        write_json(output, {"status": "runtime_error", "records": records})
        return records
    result = read_json(output)
    records = result.get("records", [])
    expected = {case["id"] for case in value.cases(split)}
    if len(records) != len(expected) or {r.get("case_id") for r in records} != expected:
        raise ValueError("worker result has missing, extra or duplicate cases")
    for record in records:
        if (
            record.get("input_id") != value.input_id
            or record.get("candidate_id") != candidate["candidate_id"]
            or record.get("split") != split
            or record.get("source_sha256") != candidate["source_sha256"]
            or record.get("evaluator_id") != candidate["evaluator_id"]
        ):
            raise ValueError("worker result identity mismatch")
    return records
