# SPDX-License-Identifier: Apache-2.0
"""Freeze routes using training measurements, then assess independent holdouts.

Buckets are declared by the session policy before tuning. Holdout observations
can reject a portfolio but cannot repair its routing. The manifest hash detects
accidental mutation; it is an integrity check, not a cryptographic signature or
proof that an untrusted party never inspected holdout data.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from .evidence import (
    EvidenceError,
    _canonical,
    _check_cases,
    _input_cases,
    _validated_rows,
    input_identity,
    summarize_evaluations,
)


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def build_portfolio(
    trace_input: Mapping[str, Any],
    train_evaluations: Iterable[Mapping[str, Any]],
    *,
    fallback_id: str = "baseline",
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Select a fully measured candidate per declared bucket, or use fallback.

    Every training case in a bucket must pass the unchanged acceptance policy.
    Candidate ranking uses equal-shape geometric speedup; missing measurements,
    failures, and excessive regressions disqualify candidates rather than being
    averaged away. No holdout record is accepted by this function.
    """
    if not isinstance(fallback_id, str) or not fallback_id:
        raise EvidenceError("fallback_id must be a nonempty string")
    identity = input_identity(trace_input)
    cases = _input_cases(trace_input)
    rows = _validated_rows(train_evaluations, input_id=identity)
    _check_cases(rows, cases)
    if any(row["split"] != "train" for row in rows):
        raise EvidenceError("routing selection must not inspect holdout evaluations")
    by_bucket: dict[str, list[str]] = defaultdict(list)
    for case_id, case in cases.items():
        if case["split"] == "train":
            by_bucket[case["bucket"]].append(case_id)
    routing = {}
    evidence = {}
    for bucket in sorted({case["bucket"] for case in cases.values()}):
        expected = sorted(by_bucket[bucket])
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["case_id"] in expected and row["candidate_id"] != fallback_id:
                grouped[row["candidate_id"]].append(row)
        candidates = []
        for candidate_id, candidate_rows in sorted(grouped.items()):
            summary = summarize_evaluations(
                candidate_rows,
                trace_input=trace_input,
                expected_case_ids=expected,
                bootstrap_iterations=bootstrap_iterations,
                seed=seed,
            )
            candidates.append({"candidate_id": candidate_id, **summary})
        eligible = [candidate for candidate in candidates if candidate["qualified"]]
        eligible.sort(
            key=lambda candidate: (-candidate["geomean_speedup"], candidate["candidate_id"])
        )
        chosen = eligible[0]["candidate_id"] if eligible else fallback_id
        routing[bucket] = chosen
        evidence[bucket] = {
            "chosen": chosen,
            "reason": "qualified_training_candidate"
            if eligible
            else "no_qualified_training_candidate",
            "training_case_ids": expected,
            "candidates": candidates,
        }
    measured_rows = [row for row in rows if row["measured"]]
    anchor = measured_rows[0] if measured_rows else None
    portfolio: dict[str, Any] = {
        "schema_version": 1,
        "input_id": identity,
        "baseline_id": anchor["baseline_id"] if anchor else None,
        "environment": anchor["environment"] if anchor else None,
        "routing": routing,
        "fallback_id": fallback_id,
        "training_case_ids": sorted(
            case_id for case_id, case in cases.items() if case["split"] == "train"
        ),
        "selection_evidence": evidence,
        "selection_method": "fixed_bucket_training_geomean_with_coverage_and_regression_gates",
        "evidence_hash": hashlib.sha256(_canonical(rows).encode()).hexdigest(),
    }
    portfolio["manifest_hash"] = _manifest_hash(portfolio)
    return portfolio


def evaluate_portfolio(
    trace_input: Mapping[str, Any],
    frozen_portfolio: Mapping[str, Any],
    holdout_evaluations: Iterable[Mapping[str, Any]],
    *,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Assess a frozen routing map; never route around a holdout regression.

    The input may contain every candidate on every holdout case. All original
    candidate summaries are retained, but only the frozen route selects the
    row used in the portfolio score. Fallbacks require their own measured rows.
    Ordinary kernel measurements exclude dispatch overhead and therefore cannot
    qualify a deployment as dispatcher-inclusive.
    """
    identity = input_identity(trace_input)
    if frozen_portfolio.get("input_id") != identity:
        raise EvidenceError("portfolio belongs to a different input")
    if frozen_portfolio.get("manifest_hash") != _manifest_hash(frozen_portfolio):
        raise EvidenceError("portfolio manifest was changed after routing was frozen")
    cases = _input_cases(trace_input)
    routing = frozen_portfolio.get("routing")
    if not isinstance(routing, Mapping) or set(routing) != {
        case["bucket"] for case in cases.values()
    }:
        raise EvidenceError("portfolio routing must cover exactly the session's declared buckets")
    if any(not isinstance(value, str) or not value for value in routing.values()):
        raise EvidenceError("portfolio routes require nonempty candidate ids")
    rows = _validated_rows(holdout_evaluations, input_id=identity)
    _check_cases(rows, cases)
    if any(row["split"] != "holdout" for row in rows):
        raise EvidenceError("portfolio evaluation requires independent holdout records")
    measured_rows = [row for row in rows if row["measured"]]
    if measured_rows:
        for key in ("environment", "baseline_id"):
            previous = frozen_portfolio.get(key)
            if previous is not None and _canonical(measured_rows[0][key]) != _canonical(previous):
                raise EvidenceError(f"holdout {key} differs from the training environment")
    expected = sorted(case_id for case_id, case in cases.items() if case["split"] == "holdout")
    lookup = {(row["candidate_id"], row["case_id"]): row for row in rows}
    selected = []
    case_routing = {}
    for case_id in expected:
        candidate_id = routing[cases[case_id]["bucket"]]
        case_routing[case_id] = candidate_id
        row = lookup.get((candidate_id, case_id))
        if row is not None:
            selected.append(row)
    report = summarize_evaluations(
        selected,
        trace_input=trace_input,
        expected_case_ids=expected,
        bootstrap_iterations=bootstrap_iterations,
        seed=seed,
    )
    training_context_verified = bool(
        frozen_portfolio.get("environment") and frozen_portfolio.get("baseline_id")
    )
    report["qualified"] = bool(report["qualified"] and training_context_verified)
    raw_reports = {}
    for candidate_id in sorted({row["candidate_id"] for row in rows}):
        raw_reports[candidate_id] = summarize_evaluations(
            [row for row in rows if row["candidate_id"] == candidate_id],
            trace_input=trace_input,
            expected_case_ids=expected,
            bootstrap_iterations=bootstrap_iterations,
            seed=seed,
        )
    fallback_id = frozen_portfolio["fallback_id"]
    fallback_cases = [case_id for case_id in expected if case_routing[case_id] == fallback_id]
    candidate_case_results = [
        result
        for result in report["case_results"]
        if case_routing[result["case_id"]] != fallback_id and result["speedup"] is not None
    ]
    dispatcher_inclusive = (
        bool(expected)
        and len(selected) == len(expected)
        and all(
            row.get("dispatcher_inclusive") is True
            and row.get("portfolio_id") == frozen_portfolio["manifest_hash"]
            for row in selected
        )
    )
    report.update(
        {
            "portfolio_id": frozen_portfolio["manifest_hash"],
            "training_context_verified": training_context_verified,
            "routing": dict(routing),
            "case_routing": case_routing,
            "fallback_id": fallback_id,
            "fallback_case_count": len(fallback_cases),
            "fallback_fraction": len(fallback_cases) / len(expected) if expected else 0.0,
            "fallback_case_ids": fallback_cases,
            "raw_candidate_regression_count": sum(
                result["speedup"] < 1 for result in candidate_case_results
            ),
            "raw_candidate_min_speedup": min(
                (result["speedup"] for result in candidate_case_results), default=None
            ),
            "raw_candidate_reports": raw_reports,
            "dispatcher_inclusive": dispatcher_inclusive,
            "measurement_scope": "dispatcher_inclusive"
            if dispatcher_inclusive
            else "selected_kernel_timings",
            "deployment_qualified": bool(report["qualified"] and dispatcher_inclusive),
            "holdout_policy": (
                "frozen routing; holdout failures reject the portfolio without route reselection"
            ),
        }
    )
    return report
