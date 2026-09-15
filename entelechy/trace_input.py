# SPDX-License-Identifier: Apache-2.0
"""Native FlashInfer Trace input, frozen selection policy and data snapshots.

The bundled upstream JSON schemas keep CPU discovery independent of GPU imports.
The runtime additionally validates these same objects with flashinfer-bench.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .artifacts import canonical_hash, read_json, write_json


@lru_cache
def schema_bundle() -> dict:
    return read_json(Path(__file__).parent / "schemas" / "flashinfer_trace.json")


def validate_native(kind: str, value: dict) -> None:
    from jsonschema import Draft202012Validator

    error = next(Draft202012Validator(schema_bundle()["schemas"][kind]).iter_errors(value), None)
    if error:
        location = ".".join(map(str, error.absolute_path))
        raise ValueError(f"FlashInfer {kind}.{location}: {error.message}")
    if kind == "Definition":
        try:
            tree = ast.parse(value["reference"])
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"]
            if len(functions) != 1:
                raise ValueError("reference must define one top-level run function")
            for expression in value.get("constraints", []):
                ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"invalid reference or constraint syntax: {exc}") from exc
        if set(value["inputs"]) & set(value["outputs"]):
            raise ValueError("Definition input and output names overlap")
        for tensor in [*value["inputs"].values(), *value["outputs"].values()]:
            if tensor["shape"] is not None and not set(tensor["shape"]) <= set(value["axes"]):
                raise ValueError("tensor shape references an unknown axis")
    if kind == "Solution":
        if value["spec"]["entry_point"].count("::") != 1:
            raise ValueError("Solution entry_point must be file::function")
        paths = [s["path"] for s in value["sources"]]
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate Solution source path")
        for name in paths:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Solution source paths must stay inside the build directory")
        if value["spec"]["entry_point"].split("::")[0] not in paths:
            raise ValueError("Solution entry source is missing")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_blob(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root.resolve()) or "://" in name:
        raise ValueError("safetensors paths must be local and relative to the TraceSet root")
    if not path.is_file():
        raise ValueError(f"missing input blob: {name}; materialize the dataset first")
    with path.open("rb") as stream:
        if stream.read(42).startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"unmaterialized Git LFS input: {name}")
    return path


def load_dataset(
    root: Path, definition_name: str | None = None
) -> tuple[dict, list[dict], list[dict]]:
    root = root.resolve()
    if not (root / "definitions").is_dir():
        raise ValueError("expected a FlashInfer TraceSet directory containing definitions/")
    definitions = [read_json(p) for p in sorted((root / "definitions").rglob("*.json"))]
    names = [d.get("name") for d in definitions]
    if len(names) != len(set(names)):
        raise ValueError("duplicate Definition names")
    selected = [d for d in definitions if d.get("name") == definition_name]
    if len(selected) != 1:
        raise ValueError(f"select --definition from: {names}")
    definition = selected[0]
    validate_native("Definition", definition)
    workloads = []
    ids = set()
    for path in sorted((root / "workloads").rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            trace = json.loads(line)
            if trace.get("definition") != definition_name:
                continue
            validate_native("Trace", trace)
            if trace.get("solution") is not None or trace.get("evaluation") is not None:
                raise ValueError("workloads/ must contain workload-only traces")
            workload = trace["workload"]
            if workload["uuid"] in ids:
                raise ValueError("duplicate workload UUID")
            ids.add(workload["uuid"])
            variable = {k for k, v in definition["axes"].items() if v["type"] == "var"}
            if set(workload["axes"]) != variable:
                raise ValueError("Workload must bind exactly the variable Definition axes")
            if set(workload["inputs"]) != set(definition["inputs"]):
                raise ValueError("Workload inputs must match Definition inputs")
            workloads.append(trace)
    if not workloads:
        raise ValueError("no workloads for the selected Definition")
    solutions = []
    for path in sorted((root / "solutions").rglob("*.json")):
        value = read_json(path)
        if value.get("definition") == definition_name:
            validate_native("Solution", value)
            solutions.append(value)
    if len({s["name"] for s in solutions}) != len(solutions):
        raise ValueError("duplicate Solution names")
    return definition, workloads, solutions


def inspect_dataset(root: Path, definition_name: str | None = None) -> dict:
    if definition_name is None:
        if not (root / "definitions").is_dir():
            raise ValueError("expected a FlashInfer TraceSet directory")
        return {
            "protocol": "entelechy.agent.v1",
            "status": "listed",
            "definitions": [
                {k: d.get(k) for k in ("name", "op_type", "description")}
                for p in sorted((root / "definitions").rglob("*.json"))
                for d in [read_json(p)]
            ],
        }
    definition, workloads, solutions = load_dataset(root, definition_name)
    return {
        "protocol": "entelechy.agent.v1",
        "status": "inspected",
        "schema_revision": schema_bundle()["upstream_revision"],
        "definition": definition,
        "input_order": list(definition["inputs"]),
        "output_order": list(definition["outputs"]),
        "workloads": workloads,
        "solutions": [{k: s.get(k) for k in ("name", "author", "spec")} for s in solutions],
        "gpu_checked": False,
    }


def load_config(path: Path | None) -> dict:
    if path is None:
        return {"num_trials": 8}
    import yaml

    value = yaml.safe_load(path.read_text())
    validate_native("BenchmarkConfig", value)
    return value


def resolve_config(value: dict, definition: dict) -> dict:
    validate_native("BenchmarkConfig", value)
    supported = {
        "warmup_runs",
        "iterations",
        "num_trials",
        "rtol",
        "atol",
        "required_matched_ratio",
        "op_type_config",
        "definition_config",
        "profile_baseline",
    }
    if set(value) - supported:
        raise ValueError(f"unsupported BenchmarkConfig controls: {sorted(set(value) - supported)}")
    result = {
        "warmup_runs": 10,
        "iterations": 50,
        "num_trials": 3,
        "rtol": 1e-2,
        "atol": 1e-2,
        "required_matched_ratio": None,
        "profile_baseline": value.get("profile_baseline", True),
        "extra": {},
    }
    for layer in (
        value.get("op_type_config", {}).get(definition["op_type"], {}),
        value.get("definition_config", {}).get(definition["name"], {}),
        value,
    ):
        for key in result:
            if layer.get(key) is not None:
                result[key] = layer[key]
    if result["extra"]:
        raise ValueError("custom evaluators are not supported by the initial row compiler")
    if not result["profile_baseline"]:
        raise ValueError("profile_baseline must remain enabled for native Trace performance fields")
    if result["num_trials"] < 4:
        raise ValueError("at least four num_trials are required for paired timing evidence")
    return result


@dataclass(frozen=True)
class TraceInput:
    """Immutable native data and session policy, with a content-derived identity."""

    _json: str

    @classmethod
    def from_dict(cls, value: dict) -> TraceInput:
        fields = {
            "kind",
            "schema_revision",
            "definition",
            "input_order",
            "output_order",
            "workloads",
            "baseline_solution",
            "blobs",
            "benchmark_config",
            "resolved_config",
            "seed",
            "target_sm",
            "cases",
            "acceptance",
            "benchmark",
        }
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("expected the native Trace input snapshot fields")
        if value.get("kind") != "flashinfer_trace":
            raise ValueError("expected a FlashInfer Trace input snapshot")
        validate_native("Definition", value["definition"])
        validate_native("Solution", value["baseline_solution"])
        if (
            list(value["definition"]["inputs"]) != value["input_order"]
            or list(value["definition"]["outputs"]) != value["output_order"]
        ):
            raise ValueError("Definition argument order changed")
        for trace in value["workloads"]:
            validate_native("Trace", trace)
        if (
            resolve_config(value["benchmark_config"], value["definition"])
            != value["resolved_config"]
        ):
            raise ValueError("resolved BenchmarkConfig changed")
        definition_name = value["definition"]["name"]
        if value["baseline_solution"]["definition"] != definition_name or any(
            trace["definition"] != definition_name
            or trace.get("solution") is not None
            or trace.get("evaluation") is not None
            for trace in value["workloads"]
        ):
            raise ValueError("input references must match the selected Definition")
        ids = [trace["workload"]["uuid"] for trace in value["workloads"]]
        case_ids = [case["id"] for case in value["cases"]]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(case_ids):
            raise ValueError("cases must select each native Workload exactly once")
        if any(set(case) != {"id", "split", "bucket"} for case in value["cases"]):
            raise ValueError("cases contain only workload UUID, split and bucket")
        benchmark = value["benchmark"]
        resolved = value["resolved_config"]
        expected = {
            "warmup": resolved["warmup_runs"],
            "repeats": resolved["iterations"],
            "trials": resolved["num_trials"],
            "timing": benchmark["timing"],
            "cold_l2": benchmark["cold_l2"],
        }
        if (
            benchmark != expected
            or benchmark["timing"] not in {"cupti", "cuda_event"}
            or type(benchmark["cold_l2"]) is not bool
        ):
            raise ValueError("timing settings disagree with the resolved BenchmarkConfig")
        return cls(json.dumps(value, allow_nan=False))

    def to_dict(self) -> dict:
        return json.loads(self._json)

    @property
    def input_id(self) -> str:
        return canonical_hash(self.to_dict())

    def cases(self, split: str) -> list[dict]:
        if split not in {"train", "holdout"}:
            raise ValueError("split must be train or holdout")
        return [c for c in self.to_dict()["cases"] if c["split"] == split]


def prepare_input(
    dataset: Path,
    name: str,
    baseline_name: str,
    *,
    policy_path: Path | None = None,
    config_path: Path | None = None,
    workload_ids: list[str] | None = None,
) -> TraceInput:
    definition, workloads, solutions = load_dataset(dataset, name)
    if workload_ids is not None:
        if len(workload_ids) != len(set(workload_ids)) or not set(workload_ids) <= {
            t["workload"]["uuid"] for t in workloads
        }:
            raise ValueError("--workloads contains duplicate or unknown UUIDs")
        workloads = [t for t in workloads if t["workload"]["uuid"] in workload_ids]
    if len(workloads) < 2:
        raise ValueError("select at least two workloads for training and holdout")
    baseline = next((s for s in solutions if s["name"] == baseline_name), None)
    if baseline is None:
        raise ValueError(f"select --baseline from: {[s['name'] for s in solutions]}")
    policy = read_json(policy_path) if policy_path else {}
    allowed = {
        "train",
        "holdout",
        "buckets",
        "acceptance",
        "seed",
        "timing",
        "cold_l2",
        "target_sm",
    }
    if not isinstance(policy, dict) or set(policy) - allowed:
        raise ValueError("unknown session policy fields")
    ids = sorted(t["workload"]["uuid"] for t in workloads)
    if "train" not in policy and "holdout" not in policy:
        cut = max(1, min(len(ids) - 1, len(ids) * 3 // 4))
        policy.update(train=ids[:cut], holdout=ids[cut:])
    train, holdout = policy.get("train", []), policy.get("holdout", [])
    if (
        not train
        or not holdout
        or len(set(train + holdout)) != len(train + holdout)
        or set(train + holdout) != set(ids)
    ):
        raise ValueError("policy train/holdout must partition the selected workload UUIDs")
    buckets = policy.get("buckets", dict.fromkeys(ids, "all"))
    if set(buckets) != set(ids) or any(not isinstance(b, str) or not b for b in buckets.values()):
        raise ValueError("policy buckets must assign every selected workload UUID")
    for bucket in set(buckets.values()):
        if not any(buckets[i] == bucket for i in train) or not any(
            buckets[i] == bucket for i in holdout
        ):
            raise ValueError("every bucket needs training and holdout workloads")
    acceptance = policy.get(
        "acceptance", {"min_speedup": 1.03, "max_regression": 0.02, "confidence": 0.95}
    )
    if set(acceptance) != {"min_speedup", "max_regression", "confidence"}:
        raise ValueError("invalid acceptance policy")
    for k, lo, hi in (
        ("min_speedup", 1, 1000),
        ("max_regression", 0, 0.5),
        ("confidence", 0.8, 0.999),
    ):
        value = acceptance[k]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not lo <= value <= hi
        ):
            raise ValueError(f"acceptance.{k} must be a finite number in [{lo}, {hi}]")
    seed = policy.get("seed", 0)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    timing = policy.get("timing", "cupti")
    if timing not in {"cupti", "cuda_event"} or type(policy.get("cold_l2", True)) is not bool:
        raise ValueError("invalid timing policy")
    target = policy.get("target_sm")
    if target is not None and (type(target) is not int or target < 70):
        raise ValueError("invalid target_sm")
    config = load_config(config_path)
    resolved = resolve_config(config, definition)
    cases, blobs, signatures = [], {}, set()
    for trace in workloads:
        workload = trace["workload"]
        signature = canonical_hash({k: workload[k] for k in ("axes", "inputs")})
        if signature in signatures:
            raise ValueError(
                "duplicate workload configurations cannot provide independent holdouts"
            )
        signatures.add(signature)
        uid = workload["uuid"]
        cases.append(
            {
                "id": uid,
                "split": "train" if uid in train else "holdout",
                "bucket": buckets[uid],
            }
        )
        for descriptor in workload["inputs"].values():
            if descriptor["type"] == "safetensors":
                path = descriptor["path"]
                blobs[path] = file_hash(local_blob(dataset, path))
    return TraceInput.from_dict(
        {
            "kind": "flashinfer_trace",
            "schema_revision": schema_bundle()["upstream_revision"],
            "definition": definition,
            "input_order": list(definition["inputs"]),
            "output_order": list(definition["outputs"]),
            "workloads": workloads,
            "baseline_solution": baseline,
            "blobs": blobs,
            "benchmark_config": config,
            "resolved_config": resolved,
            "seed": seed,
            "target_sm": target,
            "cases": cases,
            "acceptance": acceptance,
            "benchmark": {
                "warmup": resolved["warmup_runs"],
                "repeats": resolved["iterations"],
                "trials": resolved["num_trials"],
                "timing": timing,
                "cold_l2": policy.get("cold_l2", True),
            },
        }
    )


def snapshot_input(value: TraceInput, source: Path, destination: Path) -> None:
    raw = value.to_dict()
    root = destination / "dataset"
    for name, expected in raw["blobs"].items():
        origin = local_blob(source, name)
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
        if file_hash(target) != expected:
            raise ValueError("input blob changed while snapshotting")
    write_json(root / "definitions" / "selected.json", raw["definition"])
    write_json(root / "solutions" / "baseline.json", raw["baseline_solution"])
    (root / "workloads").mkdir(parents=True, exist_ok=True)
    (root / "workloads" / "selected.jsonl").write_text(
        "".join(json.dumps(t, allow_nan=False) + "\n" for t in raw["workloads"])
    )
    write_json(destination / "input.json", raw)


def verify_blobs(value: TraceInput, root: Path) -> None:
    for name, expected in value.to_dict()["blobs"].items():
        if file_hash(local_blob(root, name)) != expected:
            raise ValueError(f"input blob changed: {name}")


def load_input(path: Path) -> TraceInput:
    return TraceInput.from_dict(read_json(path))
