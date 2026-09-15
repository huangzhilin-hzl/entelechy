# SPDX-License-Identifier: Apache-2.0
"""A bounded, hardware-explicit schedule IR for contiguous row operators.

The algorithm is fixed by the operator. This IR controls physical row assignment,
loop unrolling, and the shared-memory protocol used for CTA reductions.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any


class ScheduleSchemaError(ValueError):
    """A serialized schedule does not conform to the IR schema."""


def _fields(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise ScheduleSchemaError(f"{where} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise ScheduleSchemaError(f"{where}: unknown fields {sorted(unknown)!r}")


@dataclass(frozen=True)
class SharedResource:
    name: str
    elements: int
    dtype: str = "float32"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SharedResource:
        _fields(value, {"name", "elements", "dtype"}, "resource")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ScheduleSchemaError(str(exc)) from exc


@dataclass(frozen=True)
class Barrier:
    name: str
    resource: str
    role: str
    scope: str = "cta"
    participants: str = "all"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Barrier:
        _fields(value, {"name", "resource", "role", "scope", "participants"}, "barrier")
        try:
            return cls(**value)
        except TypeError as exc:
            raise ScheduleSchemaError(str(exc)) from exc


@dataclass(frozen=True)
class Schedule:
    mapping: str = "warp_per_row"
    threads: int = 128
    items_per_thread: int = 1
    resources: tuple[SharedResource, ...] = ()
    barriers: tuple[Barrier, ...] = ()

    @classmethod
    def for_mapping(
        cls,
        mapping: str,
        *,
        threads: int = 128,
        items_per_thread: int = 1,
        operator: str = "rmsnorm",
    ) -> Schedule:
        """Build explicit synchronization declarations for a known algorithm."""
        if operator not in {"rmsnorm", "softmax", "silu_mul"}:
            raise ScheduleSchemaError(f"unsupported operator: {operator!r}")
        if mapping not in {"warp_per_row", "cta_per_row"}:
            raise ScheduleSchemaError(f"unsupported mapping: {mapping!r}")
        if type(threads) is not int:
            raise ScheduleSchemaError("threads must be an integer")
        if mapping == "cta_per_row" and operator != "silu_mul":
            return cls(
                mapping,
                threads,
                items_per_thread,
                (SharedResource("warp_partials", threads // 32),),
                (
                    Barrier("partials_ready", "warp_partials", "publish"),
                    Barrier("partials_consumed", "warp_partials", "release"),
                ),
            )
        return cls(mapping, threads, items_per_thread)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Schedule:
        _fields(
            value,
            {"mapping", "threads", "items_per_thread", "resources", "barriers"},
            "schedule",
        )
        data = dict(value)
        for key, kind in (("resources", SharedResource), ("barriers", Barrier)):
            items = data.get(key, [])
            if not isinstance(items, (list, tuple)):
                raise ScheduleSchemaError(f"{key} must be an array")
            data[key] = tuple(kind.from_dict(item) for item in items)
        try:
            return cls(**data)
        except TypeError as exc:
            raise ScheduleSchemaError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping": self.mapping,
            "threads": self.threads,
            "items_per_thread": self.items_per_thread,
            "resources": [asdict(item) for item in self.resources],
            "barriers": [asdict(item) for item in self.barriers],
        }

    @property
    def schedule_id(self) -> str:
        """Stable content identity; operator/dtype belong in the artifact identity."""
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
