# SPDX-License-Identifier: Apache-2.0
"""Seed schedules, static resource analysis and native candidate artifacts."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from .artifacts import (
    canonical_hash,
    compiler_fingerprint,
    evaluation_fingerprint,
    source_hash,
    write_json,
)
from .compiler import Schedule, verify
from .trace_input import TraceInput


def seed_schedules(operator: str) -> list[Schedule]:
    """Interleave two physical mappings, then change launch and unroll decisions."""
    return [
        Schedule.for_mapping(mapping, threads=threads, items_per_thread=items, operator=operator)
        for items, threads, mapping in product(
            (1, 2, 4), (128, 256, 64), ("warp_per_row", "cta_per_row")
        )
    ]


def analyze_schedule(
    schedule: Schedule, trace_input: TraceInput, *, program: dict
) -> dict[str, Any]:
    """Inspectable resource facts, not fabricated nanosecond predictions."""
    raw = trace_input.to_dict()
    cases = []
    for case in trace_input.cases("train"):
        from .compiler.trace import dimensions

        workload = next(
            t["workload"] for t in raw["workloads"] if t["workload"]["uuid"] == case["id"]
        )
        rows, cols = dimensions(raw["definition"], workload, program)
        case = {**case, "rows": rows, "cols": cols}
        rows_per_cta = schedule.threads // 32 if schedule.mapping == "warp_per_row" else 1
        cases.append(
            {
                "case_id": case["id"],
                "grid_ctas": (case["rows"] + rows_per_cta - 1) // rows_per_cta,
                "rows_per_cta": rows_per_cta,
                "reduction_scope": "warp" if schedule.mapping == "warp_per_row" else "cta",
                "elements_per_row": case["cols"],
            }
        )
    verification = verify(
        schedule,
        operator=raw["definition"]["op_type"],
        dtype=raw["definition"]["outputs"][program["bindings"]["output"]]["dtype"],
    )
    return {
        "model": "resource_facts_v1",
        "calibrated": False,
        "predicted_latency_ms": None,
        "shared_memory_bytes": verification.shared_memory_bytes,
        "cases": cases,
        "note": "No calibrated performance predictor yet. GPU measurements decide ranking.",
    }


def add_candidate(run_dir: Path, trace_input: TraceInput, program: dict) -> dict:
    from .compiler.trace import compile_trace, dimensions, solution_for, validate_program
    from .trace_input import validate_native

    raw = trace_input.to_dict()
    schedule = validate_program(program, raw["definition"])
    program = {**program, "schedule": schedule.to_dict()}
    for trace in raw["workloads"]:
        dimensions(raw["definition"], trace["workload"], program)
    source = compile_trace(program, raw["definition"])
    digest = source_hash(source)
    solution = solution_for(source, raw["definition"], target_sm=raw["target_sm"])
    validate_native("Solution", solution)
    directory = run_dir / "candidates" / digest[:24]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.cu").write_text(source)
    write_json(directory / "solution.json", solution)
    candidate = {
        "candidate_id": digest[:24],
        "input_id": trace_input.input_id,
        "compiler_id": compiler_fingerprint(),
        "evaluator_id": evaluation_fingerprint(),
        "program": program,
        "program_id": canonical_hash(program),
        "source_sha256": digest,
        "source": str((directory / "kernel.cu").relative_to(run_dir)),
        "solution": str((directory / "solution.json").relative_to(run_dir)),
        "solution_sha256": canonical_hash(solution),
        "origin": "agent_cli",
        "analysis": analyze_schedule(schedule, trace_input, program=program),
    }
    write_json(directory / "candidate.json", candidate)
    return candidate
