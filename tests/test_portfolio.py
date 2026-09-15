# SPDX-License-Identifier: Apache-2.0
"""Synthetic CPU tests ensure holdout data cannot select or conceal routes."""

from __future__ import annotations

import copy
import unittest

from entelechy.evidence import EvidenceError, input_identity
from entelechy.portfolio import build_portfolio, evaluate_portfolio


def trace_input():
    return {
        "schema_version": 1,
        "name": "synthetic-portfolio-test",
        "operator": "rmsnorm",
        "benchmark": {"timing": "cupti"},
        "cases": [
            {"id": "small-train", "rows": 2, "cols": 64, "split": "train", "bucket": "small"},
            {"id": "large-train", "rows": 2, "cols": 512, "split": "train", "bucket": "large"},
            {"id": "small-holdout", "rows": 3, "cols": 65, "split": "holdout", "bucket": "small"},
            {"id": "large-holdout", "rows": 3, "cols": 513, "split": "holdout", "bucket": "large"},
        ],
        "acceptance": {"min_speedup": 1.03, "max_regression": 0.02, "confidence": 0.95},
    }


def observation(candidate_id, case_id, speedup=2.0):
    return {
        "input_id": input_identity(trace_input()),
        "candidate_id": candidate_id,
        "case_id": case_id,
        "split": "holdout" if case_id.endswith("holdout") else "train",
        "baseline_id": "torch@fixture-revision",
        "environment": {"device": "synthetic-test-only", "timing": "cupti"},
        "kernel_qualified": True,
        "status": "ok",
        "correct": True,
        "baseline_samples_ms": [10, 20, 30],
        "candidate_samples_ms": [10 / speedup, 20 / speedup, 30 / speedup],
    }


class PortfolioTests(unittest.TestCase):
    def test_each_bucket_uses_only_training_and_retains_raw_holdout_regressions(self):
        train = [
            observation("one", "small-train", 2),
            observation("one", "large-train", 0.9),
            observation("two", "small-train", 0.9),
            observation("two", "large-train", 2),
        ]
        portfolio = build_portfolio(trace_input(), train, bootstrap_iterations=100)
        self.assertEqual(portfolio["routing"], {"small": "one", "large": "two"})
        holdout = [
            observation("one", "small-holdout", 0.8),
            observation("one", "large-holdout", 3),
            observation("two", "small-holdout", 3),
            observation("two", "large-holdout", 2),
        ]
        report = evaluate_portfolio(trace_input(), portfolio, holdout, bootstrap_iterations=100)
        self.assertEqual(report["case_routing"]["small-holdout"], "one")
        self.assertFalse(report["qualified"])
        self.assertEqual(report["raw_candidate_regression_count"], 1)
        self.assertAlmostEqual(report["raw_candidate_min_speedup"], 0.8)
        self.assertEqual(set(report["raw_candidate_reports"]), {"one", "two"})
        self.assertEqual(report["fallback_fraction"], 0)

    def test_holdout_records_cannot_enter_route_selection(self):
        with self.assertRaises(EvidenceError):
            build_portfolio(trace_input(), [observation("one", "small-holdout")])

    def test_missing_bucket_measurements_route_to_explicit_fallback(self):
        portfolio = build_portfolio(
            trace_input(), [observation("one", "small-train")], bootstrap_iterations=100
        )
        self.assertEqual(portfolio["routing"]["large"], "baseline")
        missing = evaluate_portfolio(
            trace_input(),
            portfolio,
            [observation("one", "small-holdout")],
            bootstrap_iterations=100,
        )
        self.assertFalse(missing["qualified"])
        self.assertEqual(missing["coverage"]["unmeasured_case_ids"], ["large-holdout"])
        measured = evaluate_portfolio(
            trace_input(),
            portfolio,
            [
                observation("one", "small-holdout"),
                observation("baseline", "large-holdout", 1),
            ],
            bootstrap_iterations=100,
        )
        self.assertTrue(measured["qualified"])
        self.assertEqual(measured["fallback_fraction"], 0.5)
        self.assertEqual(measured["fallback_case_count"], 1)
        self.assertFalse(measured["dispatcher_inclusive"])
        self.assertFalse(measured["deployment_qualified"])

    def test_manifest_change_is_detected_before_holdout(self):
        portfolio = build_portfolio(
            trace_input(), [observation("one", "small-train")], bootstrap_iterations=100
        )
        portfolio["routing"]["small"] = "two"
        with self.assertRaises(EvidenceError):
            evaluate_portfolio(trace_input(), portfolio, [])

    def test_changed_input_or_environment_is_rejected(self):
        portfolio = build_portfolio(
            trace_input(), [observation("one", "small-train")], bootstrap_iterations=100
        )
        changed = trace_input()
        changed["acceptance"]["min_speedup"] = 1.0
        with self.assertRaises(EvidenceError):
            evaluate_portfolio(changed, portfolio, [])
        for field, value in (
            ("baseline_id", "weaker-baseline"),
            ("environment", {"device": "other"}),
        ):
            row = observation("one", "small-holdout")
            row[field] = value
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                evaluate_portfolio(trace_input(), portfolio, [row])

    def test_failed_holdout_does_not_get_replaced_by_a_fast_alternative(self):
        portfolio = build_portfolio(
            trace_input(),
            [observation("one", "small-train"), observation("one", "large-train")],
            bootstrap_iterations=100,
        )
        broken = observation("one", "small-holdout")
        broken.update(
            status="invalid", correct=False, baseline_samples_ms=[], candidate_samples_ms=[]
        )
        report = evaluate_portfolio(
            trace_input(),
            portfolio,
            [
                broken,
                observation("one", "large-holdout"),
                observation("two", "small-holdout", 10),
            ],
            bootstrap_iterations=100,
        )
        self.assertFalse(report["qualified"])
        self.assertEqual(report["coverage"]["measured"], 1)
        self.assertEqual(report["case_routing"]["small-holdout"], "one")

    def test_incomplete_training_candidate_cannot_win_bucket(self):
        changed = trace_input()
        changed["cases"].append(
            {"id": "small-train-tail", "rows": 9, "cols": 71, "split": "train", "bucket": "small"}
        )
        row = observation("one", "small-train", 100)
        row["input_id"] = input_identity(changed)
        portfolio = build_portfolio(changed, [row], bootstrap_iterations=100)
        self.assertEqual(portfolio["routing"]["small"], "baseline")

    def test_dispatcher_claim_requires_matching_manifest_measurements(self):
        portfolio = build_portfolio(
            trace_input(),
            [observation("one", "small-train"), observation("one", "large-train")],
            bootstrap_iterations=100,
        )
        rows = [observation("one", "small-holdout"), observation("one", "large-holdout")]
        for row in rows:
            row.update(dispatcher_inclusive=True, portfolio_id="wrong-manifest")
        report = evaluate_portfolio(trace_input(), portfolio, rows, bootstrap_iterations=100)
        self.assertFalse(report["deployment_qualified"])
        for row in rows:
            row["portfolio_id"] = portfolio["manifest_hash"]
        measured = evaluate_portfolio(trace_input(), portfolio, rows, bootstrap_iterations=100)
        self.assertTrue(measured["deployment_qualified"])

    def test_inputs_are_not_mutated_and_duplicate_rows_are_rejected(self):
        rows = [observation("one", "small-train")]
        original = copy.deepcopy(rows)
        build_portfolio(trace_input(), rows, bootstrap_iterations=100)
        self.assertEqual(rows, original)
        with self.assertRaises(EvidenceError):
            build_portfolio(trace_input(), rows + rows, bootstrap_iterations=100)

    def test_compile_failure_before_valid_candidate_does_not_set_environment(self):
        failed = observation("broken", "small-train")
        failed.update(
            status="compile_error",
            correct=False,
            environment=None,
            baseline_id=None,
            baseline_samples_ms=[],
            candidate_samples_ms=[],
        )
        portfolio = build_portfolio(
            trace_input(),
            [
                failed,
                observation("one", "small-train"),
                observation("one", "large-train"),
            ],
            bootstrap_iterations=100,
        )
        self.assertEqual(
            portfolio["environment"], {"device": "synthetic-test-only", "timing": "cupti"}
        )
        self.assertEqual(portfolio["baseline_id"], "torch@fixture-revision")
        report = evaluate_portfolio(
            trace_input(),
            portfolio,
            [
                observation("one", "small-holdout"),
                observation("one", "large-holdout"),
            ],
            bootstrap_iterations=100,
        )
        self.assertTrue(report["qualified"])

    def test_all_failed_training_has_no_verified_context(self):
        failed = observation("broken", "small-train")
        failed.update(
            status="timeout",
            correct=False,
            environment={},
            baseline_id=None,
            baseline_samples_ms=[],
            candidate_samples_ms=[],
        )
        portfolio = build_portfolio(trace_input(), [failed], bootstrap_iterations=100)
        self.assertIsNone(portfolio["environment"])
        self.assertIsNone(portfolio["baseline_id"])
        report = evaluate_portfolio(
            trace_input(),
            portfolio,
            [
                observation("baseline", "small-holdout", 1.1),
                observation("baseline", "large-holdout", 1.1),
            ],
            bootstrap_iterations=100,
        )
        self.assertFalse(report["training_context_verified"])
        self.assertFalse(report["qualified"])


if __name__ == "__main__":
    unittest.main()
