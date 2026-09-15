# SPDX-License-Identifier: Apache-2.0
"""Static legality checks for the bounded row schedule IR.

These checks establish legality of our templates, not arbitrary CUDA programs.
Correctness and performance still require execution against independent oracles.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

from .ir import Barrier, Schedule, SharedResource

OPERATORS = frozenset({"rmsnorm", "softmax", "silu_mul"})
DTYPES = frozenset({"float16", "bfloat16", "float32"})


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    path: str
    severity: str = "error"


@dataclass(frozen=True)
class VerificationReport:
    diagnostics: tuple[Diagnostic, ...]
    shared_memory_bytes: int = 0

    @property
    def valid(self) -> bool:
        return not self.diagnostics

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "diagnostics": [asdict(item) for item in self.diagnostics],
            "shared_memory_bytes": self.shared_memory_bytes,
        }


class ScheduleValidationError(ValueError):
    def __init__(self, report: VerificationReport):
        self.report = report
        super().__init__("; ".join(f"{d.code} ({d.path}): {d.message}" for d in report.diagnostics))


def verify(
    schedule: Schedule,
    operator: str | None = None,
    cols: int | None = None,
    dtype: str | None = None,
    target: str | None = None,
) -> VerificationReport:
    """Check schedule, optional workload shape, dtype and CUDA SM target.

    When operator is omitted, CTA schedules with reduction declarations are
    checked as reductions, and empty CTA declarations as elementwise schedules.
    Compile always supplies the operator, so incompatible protocols cannot lower.
    """
    errors: list[Diagnostic] = []

    def fail(code: str, message: str, path: str) -> None:
        errors.append(Diagnostic(code, message, path))

    if not isinstance(schedule, Schedule):
        fail("SCHEMA_TYPE", "expected a Schedule instance", "schedule")
        return VerificationReport(tuple(errors))
    if operator is not None and (not isinstance(operator, str) or operator not in OPERATORS):
        fail("UNSUPPORTED_OPERATOR", f"unsupported operator {operator!r}", "operator")
    if dtype is not None and (not isinstance(dtype, str) or dtype not in DTYPES):
        fail("UNSUPPORTED_DTYPE", f"unsupported dtype {dtype!r}", "dtype")
    if cols is not None and (type(cols) is not int or cols <= 0 or cols > 2**31 - 1):
        fail("INVALID_SHAPE", "cols must be an integer in [1, 2**31 - 1]", "cols")
    if target is not None:
        match = re.fullmatch(r"sm_(\d+)(?:a|f)?", target) if isinstance(target, str) else None
        if not match or int(match[1]) < 70:
            fail("UNSUPPORTED_TARGET", "target must be an SM identifier >= sm_70", "target")
        elif dtype == "bfloat16" and int(match[1]) < 80:
            fail("UNSUPPORTED_TARGET", "bfloat16 kernels require sm_80 or newer", "target")
    if schedule.mapping not in ("warp_per_row", "cta_per_row"):
        fail("UNSUPPORTED_MAPPING", "expected warp_per_row or cta_per_row", "mapping")
    threads_valid = (
        type(schedule.threads) is int
        and 32 <= schedule.threads <= 1024
        and schedule.threads % 32 == 0
    )
    if not threads_valid:
        fail("INVALID_THREADS", "threads must be a multiple of 32 in [32, 1024]", "threads")
    if type(schedule.items_per_thread) is not int or schedule.items_per_thread not in (1, 2, 4, 8):
        fail("INVALID_UNROLL", "items_per_thread must be one of 1, 2, 4, 8", "items_per_thread")
    if not isinstance(schedule.resources, tuple) or any(
        not isinstance(r, SharedResource) for r in schedule.resources
    ):
        fail("SCHEMA_TYPE", "resources must be a tuple of SharedResource objects", "resources")
        return VerificationReport(tuple(errors))
    if not isinstance(schedule.barriers, tuple) or any(
        not isinstance(b, Barrier) for b in schedule.barriers
    ):
        fail("SCHEMA_TYPE", "barriers must be a tuple of Barrier objects", "barriers")
        return VerificationReport(tuple(errors))

    resource_names = [r.name for r in schedule.resources]
    if any(not isinstance(name, str) for name in resource_names):
        fail("INVALID_RESOURCE_NAME", "resource names must be strings", "resources")
    elif len(set(resource_names)) != len(resource_names):
        fail("DUPLICATE_RESOURCE", "resource names must be unique", "resources")
    memory_bytes = 0
    for index, resource in enumerate(schedule.resources):
        path = f"resources[{index}]"
        if resource.name != "warp_partials":
            fail(
                "UNSUPPORTED_RESOURCE", "only warp_partials is supported by this IR", f"{path}.name"
            )
        if resource.dtype != "float32":
            fail("RESOURCE_DTYPE", "reduction partials must accumulate in float32", f"{path}.dtype")
        if type(resource.elements) is not int or resource.elements <= 0:
            fail(
                "RESOURCE_SIZE",
                "shared resource elements must be a positive integer",
                f"{path}.elements",
            )
        else:
            memory_bytes += resource.elements * 4
            if threads_valid and resource.elements != schedule.threads // 32:
                fail(
                    "RESOURCE_SIZE",
                    "warp_partials must have exactly one element per warp",
                    f"{path}.elements",
                )

    barrier_names: list[str] = []
    for index, barrier in enumerate(schedule.barriers):
        path = f"barriers[{index}]"
        if not isinstance(barrier.name, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", barrier.name
        ):
            fail("INVALID_BARRIER_NAME", "barrier name must be a C identifier", f"{path}.name")
        else:
            barrier_names.append(barrier.name)
        if barrier.resource not in resource_names:
            fail(
                "UNKNOWN_RESOURCE",
                "barrier refers to an undeclared shared resource",
                f"{path}.resource",
            )
        if barrier.scope != "cta":
            fail(
                "BARRIER_SCOPE", "cross-warp partials require CTA synchronization", f"{path}.scope"
            )
        if barrier.participants != "all":
            fail(
                "BARRIER_DIVERGENCE",
                "every thread must reach each CTA barrier",
                f"{path}.participants",
            )
        if barrier.role not in ("publish", "release"):
            fail("BARRIER_ROLE", "expected publish or release role", f"{path}.role")
    if len(set(barrier_names)) != len(barrier_names):
        fail("DUPLICATE_BARRIER", "barrier names must be unique", "barriers")

    reduction = (
        operator in ("rmsnorm", "softmax")
        if operator is not None
        else bool(schedule.resources or schedule.barriers)
    )
    needs_shared = schedule.mapping == "cta_per_row" and reduction
    if needs_shared:
        if len(schedule.resources) != 1 or resource_names != ["warp_partials"]:
            fail(
                "MISSING_REDUCTION_RESOURCE",
                "CTA reduction requires the warp_partials resource",
                "resources",
            )
        if [b.role for b in schedule.barriers] != ["publish", "release"]:
            fail(
                "REDUCTION_DATAFLOW",
                "CTA reduction must publish partials before reads and release them before reuse",
                "barriers",
            )
    elif schedule.resources or schedule.barriers:
        fail(
            "UNUSED_PROTOCOL",
            "this mapping/operator uses no shared memory or CTA barriers",
            "resources",
        )
    if memory_bytes > 48 * 1024:
        fail(
            "SHARED_MEMORY_LIMIT",
            "static shared memory exceeds the portable 48 KiB limit",
            "resources",
        )
    return VerificationReport(tuple(errors), memory_bytes)
