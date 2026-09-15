# SPDX-License-Identifier: Apache-2.0
"""Deterministic identities and atomic, human-readable experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def read_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # FlashInfer Definition input/output object order is part of its calling convention.
    text = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def compiler_fingerprint() -> str:
    root = Path(__file__).parent / "compiler"
    return canonical_hash({p.name: source_hash(p.read_text()) for p in sorted(root.glob("*.py"))})


def evaluation_fingerprint() -> str:
    """Identify the evaluator, oracle, admission and acceptance implementations.

    A fixed workload JSON cannot freeze an oracle whose implementation changes.
    Include the entire package so a controller or runtime edit also invalidates
    an existing experiment. These hashes detect drift relative to a trusted
    manifest; they do not authenticate arbitrary files supplied by a third party.
    """
    root = Path(__file__).resolve().parent
    return canonical_hash(
        {
            path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and (
                path.suffix == ".py" or (path.suffix == ".json" and path.parent == root / "schemas")
            )
            if "__pycache__" not in path.relative_to(root).parts
        }
    )


def validate_candidate_artifact(
    run_dir: Path,
    input_id: str,
    compiler_id: str,
    candidate: dict[str, Any],
    *,
    allow_baseline: bool = False,
) -> Path:
    """Bind a candidate identity to its actual source and canonical schedule.

    The caller supplies the frozen input/compiler identities and separately
    checks that the executing evaluator still has the frozen fingerprint. The
    explicit baseline route retains a real seed artifact for job bookkeeping;
    its reserved name is accepted only when requested by the caller.
    """
    # Local imports avoid a compiler/artifact import cycle during initialization.
    from .compiler import Schedule, verify

    if not isinstance(candidate, dict):
        raise ValueError("candidate must be an object")
    if candidate.get("input_id") != input_id:
        raise ValueError("candidate and input identity mismatch")
    if candidate.get("compiler_id") != compiler_id:
        raise ValueError("candidate and compiler identity mismatch")
    digest = candidate.get("source_sha256")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("candidate source_sha256 must be a SHA-256 hexadecimal digest")
    candidate_id = candidate.get("candidate_id")
    if candidate_id == "baseline":
        if not allow_baseline:
            raise ValueError("the reserved baseline route is not an ordinary candidate")
    elif candidate_id != digest[:24]:
        raise ValueError("candidate_id is not bound to source_sha256")
    source = candidate.get("source")
    if not isinstance(source, str) or not source or Path(source).is_absolute():
        raise ValueError("candidate source must be a relative artifact path")
    root = Path(run_dir).resolve()
    path = (root / source).resolve()
    if not path.is_relative_to(root):
        raise ValueError("candidate source must be inside the experiment")
    if not path.is_file():
        raise ValueError("candidate source does not exist or is not a file")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("generated source was modified after planning")
    if "program" not in candidate or "solution" not in candidate:
        raise ValueError("candidate requires a native program and Solution")
    program = candidate["program"]
    schedule = Schedule.from_dict(program.get("schedule"))
    if canonical_hash(schedule.to_dict()) != canonical_hash(program["schedule"]):
        raise ValueError("candidate schedule must use the canonical serialized representation")
    if candidate.get("program_id") != canonical_hash(program):
        raise ValueError("candidate program identity mismatch")
    verification = verify(schedule)
    if not verification.valid:
        raise ValueError("candidate contains an invalid schedule")
    from .compiler.trace import compile_trace, solution_for
    from .trace_input import load_input

    raw = load_input(root / "input.json").to_dict()
    generated = compile_trace(candidate["program"], raw["definition"])
    if source_hash(generated) != digest:
        raise ValueError("Trace IR no longer generates the submitted CUDA")
    solution_path = (root / candidate["solution"]).resolve()
    if not solution_path.is_relative_to(root):
        raise ValueError("Solution must stay inside the experiment")
    solution = read_json(solution_path)
    expected = solution_for(generated, raw["definition"], target_sm=raw["target_sm"])
    if canonical_hash(solution) != candidate["solution_sha256"] or solution != expected:
        raise ValueError("Solution changed after submission")
    return path
