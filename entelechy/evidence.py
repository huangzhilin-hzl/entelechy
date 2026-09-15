# SPDX-License-Identifier: Apache-2.0
"""Validate measurement evidence and report what the samples actually support.

This module never executes a candidate or rewrites an acceptance policy. All
statistics are deterministic functions of recorded observations. Paired samples
must represent matched baseline/candidate timing rounds, not independently sorted
latencies. Confidence intervals quantify timing noise, not search-selection bias.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


class EvidenceError(ValueError):
    """Evidence is malformed or cannot support the requested comparison."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("Evidence must contain finite JSON-compatible values") from exc


def _samples(values: Sequence[float], name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise EvidenceError(f"{name} must contain at least two paired samples")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvidenceError(f"{name} must contain numeric samples")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise EvidenceError(f"{name} samples must be finite and positive")
        result.append(value)
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_options(confidence: float, bootstrap_iterations: int) -> None:
    if not 0 < confidence < 1:
        raise EvidenceError("confidence must be between zero and one")
    if isinstance(bootstrap_iterations, bool) or not isinstance(bootstrap_iterations, int):
        raise EvidenceError("bootstrap_iterations must be a positive integer")
    if bootstrap_iterations < 1:
        raise EvidenceError("bootstrap_iterations must be a positive integer")


def paired_speedup(
    baseline_samples: Sequence[float],
    candidate_samples: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate a paired geometric speedup and percentile bootstrap interval.

    Speedup is ``exp(mean(log(baseline) - log(candidate)))``. A timing round
    stays paired during resampling. The median latencies are descriptive and are
    deliberately not substituted for this estimator.
    """
    baseline = _samples(baseline_samples, "baseline_samples")
    candidate = _samples(candidate_samples, "candidate_samples")
    if len(baseline) != len(candidate):
        raise EvidenceError("baseline and candidate samples must be paired and equal in length")
    _bootstrap_options(confidence, bootstrap_iterations)
    logs = [math.log(b) - math.log(c) for b, c in zip(baseline, candidate, strict=True)]
    rng = random.Random(seed)
    bootstrap = [
        statistics.fmean(rng.choices(logs, k=len(logs))) for _ in range(bootstrap_iterations)
    ]
    alpha = (1 - confidence) / 2
    try:
        estimate = math.exp(statistics.fmean(logs))
        interval = [
            math.exp(_quantile(bootstrap, alpha)),
            math.exp(_quantile(bootstrap, 1 - alpha)),
        ]
    except OverflowError as exc:
        raise EvidenceError("sample ratios exceed the supported numeric range") from exc
    if not all(math.isfinite(value) and value > 0 for value in [estimate, *interval]):
        raise EvidenceError("sample ratios exceed the supported numeric range")
    return {
        "speedup": estimate,
        "confidence_interval": interval,
        "confidence": confidence,
        "sample_pairs": len(logs),
        "baseline_median_ms": statistics.median(baseline),
        "candidate_median_ms": statistics.median(candidate),
        "estimator": "geometric_mean_paired_speedup",
        "interval_method": "paired_log_ratio_percentile_bootstrap",
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
    }


def input_identity(trace_input: Mapping[str, Any]) -> str:
    """Hash a frozen input snapshot, optionally checking an attached identity."""
    payload = dict(trace_input)
    claimed = payload.pop("input_id", None)
    identity = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    if claimed is not None and claimed != identity:
        raise EvidenceError("input_id does not match the input contents")
    return identity


def validate_evaluation(
    evaluation: Mapping[str, Any], *, input_id: str | None = None
) -> dict[str, Any]:
    """Validate a row without turning missing or failed measurements into wins."""
    if not isinstance(evaluation, Mapping):
        raise EvidenceError("evaluation must be an object")
    result = dict(evaluation)
    for name in ("input_id", "candidate_id", "case_id"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise EvidenceError(f"evaluation requires a nonempty {name}")
    if input_id is not None and result["input_id"] != input_id:
        raise EvidenceError("evaluation input_id does not match the fixed input")
    if result.get("split") not in ("train", "holdout"):
        raise EvidenceError("evaluation split must be train or holdout")
    statuses = {"ok", "invalid", "compile_error", "runtime_error", "timeout", "unavailable"}
    if result.get("status") not in statuses:
        raise EvidenceError("evaluation has an unsupported status")
    if not isinstance(result.get("correct"), bool):
        raise EvidenceError("evaluation correct must be a boolean")
    baseline = result.get("baseline_samples_ms")
    candidate = result.get("candidate_samples_ms")
    has_baseline = baseline is not None and baseline != []
    has_candidate = candidate is not None and candidate != []
    if has_baseline != has_candidate:
        raise EvidenceError("baseline and candidate measurements must both be present")
    if has_baseline:
        b = _samples(baseline, "baseline_samples_ms")
        c = _samples(candidate, "candidate_samples_ms")
        if len(b) != len(c):
            raise EvidenceError("baseline and candidate sample counts differ")
        if result["status"] != "ok" or result["correct"] is not True:
            raise EvidenceError("failed or incorrect evaluations cannot contain accepted timings")
        if not isinstance(result.get("baseline_id"), str) or not result["baseline_id"]:
            raise EvidenceError("measured evaluation requires a concrete baseline_id")
        if not isinstance(result.get("environment"), Mapping) or not result["environment"]:
            raise EvidenceError("measured evaluation requires an explicit nonempty environment")
        result["baseline_samples_ms"] = b
        result["candidate_samples_ms"] = c
    environment = result.get("environment")
    if environment is not None and not isinstance(environment, Mapping):
        raise EvidenceError("environment must be an object or unknown")
    if environment is not None:
        result["environment"] = json.loads(_canonical(environment))
    result["measured"] = bool(has_baseline and result["status"] == "ok" and result["correct"])
    return result


def _validated_rows(
    evaluations: Iterable[Mapping[str, Any]], *, input_id: str | None = None
) -> list[dict[str, Any]]:
    rows = [validate_evaluation(row, input_id=input_id) for row in evaluations]
    if rows:
        if len({row["input_id"] for row in rows}) != 1:
            raise EvidenceError("cannot combine evaluations with different input_id")
        measured = [row for row in rows if row["measured"]]
        for key in ("baseline_id", "environment"):
            if len({_canonical(row[key]) for row in measured}) > 1:
                raise EvidenceError(f"cannot combine evaluations with different {key}")
        seen = set()
        cases: dict[str, str] = {}
        for row in rows:
            key = (row["candidate_id"], row["case_id"])
            if key in seen:
                raise EvidenceError("duplicate candidate/case evaluation")
            seen.add(key)
            if row["case_id"] in cases and cases[row["case_id"]] != row["split"]:
                raise EvidenceError("a case cannot change its train/holdout split")
            cases[row["case_id"]] = row["split"]
    return rows


def _input_cases(trace_input: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases = trace_input.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvidenceError("input must declare cases")
    result = {}
    for case in cases:
        if not isinstance(case, Mapping) or not isinstance(case.get("id"), str):
            raise EvidenceError("input cases require string ids")
        if not case["id"] or case["id"] in result:
            raise EvidenceError("input case ids must be nonempty and unique")
        if case.get("split") not in ("train", "holdout"):
            raise EvidenceError("input case split must be train or holdout")
        if not isinstance(case.get("bucket"), str) or not case["bucket"]:
            raise EvidenceError("input cases require a predefined bucket")
        result[case["id"]] = dict(case)
    return result


def _check_cases(rows: Iterable[Mapping[str, Any]], cases: Mapping[str, Mapping[str, Any]]) -> None:
    for row in rows:
        if row["case_id"] not in cases:
            raise EvidenceError(f"unknown input case: {row['case_id']}")
        case = cases[row["case_id"]]
        if row["split"] != case["split"]:
            raise EvidenceError("evaluation split differs from the fixed input")
        for key in ("rows", "cols", "bucket"):
            if key in row and row[key] != case.get(key):
                raise EvidenceError(f"evaluation {key} differs from the fixed case")


def _acceptance(values: Mapping[str, Any] | None) -> dict[str, float]:
    result = {"min_speedup": 1.03, "max_regression": 0.02, "confidence": 0.95}
    if values is not None:
        result.update({key: values[key] for key in result if key in values})
    for key, value in result.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise EvidenceError(f"acceptance {key} must be a finite number")
    if result["min_speedup"] < 1 or result["max_regression"] < 0:
        raise EvidenceError("acceptance cannot redefine a slowdown as a performance win")
    if not 0 < result["confidence"] < 1:
        raise EvidenceError("acceptance confidence must be between zero and one")
    return result


def summarize_evaluations(
    evaluations: Iterable[Mapping[str, Any]],
    *,
    input_id: str | None = None,
    trace_input: Mapping[str, Any] | None = None,
    expected_case_ids: Iterable[str] | None = None,
    acceptance: Mapping[str, Any] | None = None,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize one preselected candidate or portfolio with one row per case.

    Supplying the full fixed input (or expected case ids) is necessary for a
    coverage claim. Missing cases never disappear from that denominator. The
    aggregate interval resamples paired rounds within each fixed shape; shapes
    are equally weighted and are not sampled as a hypothetical population.
    """
    if trace_input is not None:
        identity = input_identity(trace_input)
        if input_id is not None and input_id != identity:
            raise EvidenceError("explicit input_id differs from input contents")
        input_id = identity
        if acceptance is not None and dict(acceptance) != trace_input.get("acceptance", {}):
            raise EvidenceError("cannot override a fixed input's acceptance criteria")
        acceptance = trace_input.get("acceptance")
    policy = _acceptance(acceptance)
    _bootstrap_options(policy["confidence"], bootstrap_iterations)
    rows = _validated_rows(evaluations, input_id=input_id)
    observed_ids = [row["case_id"] for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        raise EvidenceError("summary requires one preselected evaluation per case")
    splits = {row["split"] for row in rows}
    if len(splits) > 1:
        raise EvidenceError("training and holdout evidence must be summarized separately")
    if trace_input is not None:
        cases = _input_cases(trace_input)
        _check_cases(rows, cases)
        if expected_case_ids is None:
            expected_case_ids = [
                key for key, case in cases.items() if not splits or case["split"] in splits
            ]
    domain_declared = expected_case_ids is not None
    expected_list = list(expected_case_ids) if domain_declared else observed_ids
    if len(expected_list) != len(set(expected_list)):
        raise EvidenceError("expected_case_ids must be unique")
    expected = set(expected_list)
    if not set(observed_ids).issubset(expected):
        raise EvidenceError("evaluation contains a case outside the declared reporting domain")
    by_case = {row["case_id"]: row for row in rows}
    results = []
    log_groups = []
    speedups = []
    measured_ids = []
    for case_id in sorted(expected):
        row = by_case.get(case_id)
        case_result: dict[str, Any] = {"case_id": case_id, "status": "missing", "speedup": None}
        if row is not None:
            case_result.update(
                candidate_id=row["candidate_id"], status=row["status"], correct=row["correct"]
            )
            if row["measured"]:
                stats = paired_speedup(
                    row["baseline_samples_ms"],
                    row["candidate_samples_ms"],
                    confidence=policy["confidence"],
                    bootstrap_iterations=bootstrap_iterations,
                    seed=seed,
                )
                case_result.update(stats)
                speedups.append(stats["speedup"])
                measured_ids.append(case_id)
                log_groups.append(
                    [
                        math.log(b) - math.log(c)
                        for b, c in zip(
                            row["baseline_samples_ms"], row["candidate_samples_ms"], strict=True
                        )
                    ]
                )
        results.append(case_result)
    full_coverage = bool(expected) and len(measured_ids) == len(expected)
    geomean = (
        math.exp(statistics.fmean(math.log(value) for value in speedups)) if speedups else None
    )
    interval = None
    if speedups:
        rng = random.Random(seed)
        distribution = [
            statistics.fmean(
                statistics.fmean(rng.choices(group, k=len(group))) for group in log_groups
            )
            for _ in range(bootstrap_iterations)
        ]
        alpha = (1 - policy["confidence"]) / 2
        interval = [
            math.exp(_quantile(distribution, alpha)),
            math.exp(_quantile(distribution, 1 - alpha)),
        ]
    floor = 1 / (1 + policy["max_regression"])
    material_regressions = sum(value < floor for value in speedups)
    thresholds_passed = bool(
        domain_declared
        and full_coverage
        and interval is not None
        and interval[0] >= policy["min_speedup"]
        and material_regressions == 0
    )
    measured_rows = [row for row in rows if row["measured"]]
    anchor = measured_rows[0] if measured_rows else None
    kernel_measurement_qualified = bool(measured_rows) and all(
        row.get("kernel_qualified") is True and row["environment"].get("timing") == "cupti"
        for row in measured_rows
    )
    if trace_input is not None and trace_input.get("benchmark", {}).get("timing") != "cupti":
        kernel_measurement_qualified = False
    return {
        "input_id": input_id or (rows[0]["input_id"] if rows else None),
        "baseline_id": anchor["baseline_id"] if anchor else None,
        "environment": anchor["environment"] if anchor else None,
        "split": next(iter(splits)) if splits else None,
        "geomean_speedup": geomean,
        "min_speedup": min(speedups) if speedups else None,
        "win_count": sum(value > 1 for value in speedups),
        "regression_count": sum(value < 1 for value in speedups),
        "material_regression_count": material_regressions,
        "confidence_interval": interval,
        "confidence": policy["confidence"],
        "coverage": {
            "expected": len(expected),
            "observed": len(rows),
            "measured": len(measured_ids),
            "fraction": len(measured_ids) / len(expected) if expected else 0.0,
            "complete": full_coverage,
            "domain_declared": domain_declared,
            "missing_case_ids": sorted(expected - set(observed_ids)),
            "unmeasured_case_ids": sorted(expected - set(measured_ids)),
        },
        "performance_thresholds_passed": thresholds_passed,
        "kernel_measurement_qualified": kernel_measurement_qualified,
        "qualified": bool(thresholds_passed and kernel_measurement_qualified),
        "acceptance": policy,
        "case_results": results,
        "statistical_scope": (
            "equal-weight fixed cases; paired timing noise only; no selection correction"
        ),
        "performance_scope": "complete_domain" if full_coverage else "measured_subset_only",
    }
