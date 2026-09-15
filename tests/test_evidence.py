# SPDX-License-Identifier: Apache-2.0
"""CPU-only adversarial checks of reporting semantics, using synthetic timings."""

from __future__ import annotations

import copy
import math
import unittest

from entelechy.evidence import (
    EvidenceError,
    input_identity,
    paired_speedup,
    summarize_evaluations,
    validate_evaluation,
)


def evaluation(case_id="a", speedup=2.0, candidate_id="candidate"):
    return {
        "input_id": "fixed-trace_input",
        "candidate_id": candidate_id,
        "case_id": case_id,
        "split": "train",
        "baseline_id": "torch@fixture-revision",
        "environment": {"device": "synthetic-test-only", "timing": "cupti"},
        "kernel_qualified": True,
        "status": "ok",
        "correct": True,
        "baseline_samples_ms": [10.0, 20.0, 30.0],
        "candidate_samples_ms": [10.0 / speedup, 20.0 / speedup, 30.0 / speedup],
        "diagnostics": [],
    }


class PairedStatisticsTests(unittest.TestCase):
    def test_paired_rounds_preserve_shared_timing_drift(self):
        result = paired_speedup([10, 100, 1000], [5, 50, 500], bootstrap_iterations=100)
        self.assertAlmostEqual(result["speedup"], 2)
        self.assertAlmostEqual(result["confidence_interval"][0], 2)
        self.assertAlmostEqual(result["confidence_interval"][1], 2)
        self.assertEqual(result["sample_pairs"], 3)

    def test_interval_is_deterministic_and_reflects_disagreement(self):
        first = paired_speedup([2, 1], [1, 2], bootstrap_iterations=300, seed=42)
        second = paired_speedup([2, 1], [1, 2], bootstrap_iterations=300, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["speedup"], 1)
        self.assertLess(first["confidence_interval"][0], 1)
        self.assertGreater(first["confidence_interval"][1], 1)

    def test_invalid_samples_cannot_enter_statistics(self):
        for bad in ([], [1], [0, 1], [-1, 1], [math.nan, 1], [math.inf, 1], [True, 1], ["1", 1]):
            with self.subTest(samples=bad), self.assertRaises(EvidenceError):
                paired_speedup(bad, [1, 1])
        with self.assertRaises(EvidenceError):
            paired_speedup([1, 2], [1, 2, 3])


class EvidenceTests(unittest.TestCase):
    def test_undeclared_domain_does_not_qualify(self):
        report = summarize_evaluations([evaluation()], bootstrap_iterations=100)
        self.assertGreater(report["geomean_speedup"], 1)
        self.assertFalse(report["qualified"])
        self.assertFalse(report["coverage"]["domain_declared"])

    def test_missing_measurements_never_become_speedup(self):
        row = evaluation()
        row.update(
            status="unavailable", correct=False, baseline_samples_ms=[], candidate_samples_ms=[]
        )
        report = summarize_evaluations(
            [row], expected_case_ids=["a", "b"], bootstrap_iterations=100
        )
        self.assertIsNone(report["geomean_speedup"])
        self.assertIsNone(report["min_speedup"])
        self.assertIsNone(report["confidence_interval"])
        self.assertEqual(report["coverage"]["measured"], 0)
        self.assertEqual(report["coverage"]["missing_case_ids"], ["b"])
        self.assertFalse(report["qualified"])

    def test_fast_subset_cannot_hide_missing_cases(self):
        report = summarize_evaluations(
            [evaluation()], expected_case_ids=["a", "b"], bootstrap_iterations=100
        )
        self.assertAlmostEqual(report["geomean_speedup"], 2)
        self.assertEqual(report["coverage"]["fraction"], 0.5)
        self.assertEqual(report["performance_scope"], "measured_subset_only")
        self.assertFalse(report["qualified"])

    def test_shapes_have_equal_weight_despite_different_sample_counts(self):
        first = evaluation("a", 2)
        second = evaluation("b", 0.5)
        second["baseline_samples_ms"] *= 10
        second["candidate_samples_ms"] *= 10
        report = summarize_evaluations(
            [first, second], expected_case_ids=["a", "b"], bootstrap_iterations=100
        )
        self.assertAlmostEqual(report["geomean_speedup"], 1)
        self.assertEqual(report["win_count"], 1)
        self.assertEqual(report["regression_count"], 1)
        self.assertFalse(report["qualified"])

    def test_latency_regression_threshold_is_not_speedup_subtraction(self):
        report = summarize_evaluations(
            [evaluation("a", 2), evaluation("b", 0.98)],
            expected_case_ids=["a", "b"],
            bootstrap_iterations=100,
        )
        self.assertGreater(report["geomean_speedup"], 1.03)
        self.assertEqual(report["material_regression_count"], 1)
        self.assertFalse(report["qualified"])

    def test_complete_positive_evidence_can_qualify(self):
        report = summarize_evaluations(
            [evaluation("a", 2), evaluation("b", 1.2)],
            expected_case_ids=["a", "b"],
            bootstrap_iterations=100,
        )
        self.assertTrue(report["qualified"])
        self.assertTrue(report["coverage"]["complete"])
        self.assertEqual(report["regression_count"], 0)

    def test_cuda_event_debug_timings_cannot_qualify_kernel_performance(self):
        row = evaluation()
        row["environment"]["timing"] = "cuda_event"
        row["kernel_qualified"] = False
        report = summarize_evaluations([row], expected_case_ids=["a"], bootstrap_iterations=100)
        self.assertTrue(report["performance_thresholds_passed"])
        self.assertFalse(report["kernel_measurement_qualified"])
        self.assertFalse(report["qualified"])
        row["kernel_qualified"] = True
        self.assertFalse(
            summarize_evaluations([row], expected_case_ids=["a"], bootstrap_iterations=100)[
                "qualified"
            ]
        )

    def test_mixed_comparisons_are_rejected(self):
        for key, value in (
            ("input_id", "other-trace_input"),
            ("baseline_id", "weaker-baseline"),
            ("environment", {"device": "different-gpu"}),
            ("split", "holdout"),
        ):
            first, second = evaluation("a"), evaluation("b")
            second[key] = value
            with self.subTest(field=key), self.assertRaises(EvidenceError):
                summarize_evaluations([first, second], bootstrap_iterations=100)

    def test_duplicate_or_alternative_candidate_is_not_pooled(self):
        with self.assertRaises(EvidenceError):
            summarize_evaluations([evaluation(), evaluation()], bootstrap_iterations=100)
        with self.assertRaises(EvidenceError):
            summarize_evaluations(
                [evaluation(candidate_id="a"), evaluation(candidate_id="b")],
                bootstrap_iterations=100,
            )

    def test_incorrect_or_partially_paired_timings_are_rejected(self):
        for update in (
            {"correct": False},
            {"status": "invalid"},
            {"candidate_samples_ms": []},
            {"baseline_samples_ms": [1, 2]},
        ):
            row = evaluation()
            row.update(update)
            with self.subTest(update=update), self.assertRaises(EvidenceError):
                validate_evaluation(row)

    def test_unknown_failure_metadata_does_not_poison_valid_measurements(self):
        for status in ("compile_error", "timeout"):
            failed = evaluation("a")
            failed.update(
                status=status,
                correct=False,
                environment=None,
                baseline_id=None,
                baseline_samples_ms=[],
                candidate_samples_ms=[],
            )
            measured = evaluation("b")
            report = summarize_evaluations(
                [failed, measured], expected_case_ids=["a", "b"], bootstrap_iterations=100
            )
            self.assertEqual(report["environment"], measured["environment"])
            self.assertEqual(report["baseline_id"], measured["baseline_id"])
            self.assertEqual(report["coverage"]["measured"], 1)
            self.assertFalse(report["qualified"])

    def test_known_failure_environment_does_not_count_as_a_timing_context(self):
        failed = evaluation("a")
        failed.update(
            status="runtime_error",
            correct=False,
            environment={"device": "failure-host"},
            baseline_id="not-yet-measured-baseline",
            baseline_samples_ms=[],
            candidate_samples_ms=[],
        )
        report = summarize_evaluations([failed, evaluation("b")], bootstrap_iterations=100)
        self.assertEqual(report["baseline_id"], "torch@fixture-revision")

    def test_measured_rows_still_require_concrete_comparison_metadata(self):
        for update in ({"baseline_id": None}, {"environment": None}, {"environment": {}}):
            row = evaluation()
            row.update(update)
            with self.subTest(update=update), self.assertRaises(EvidenceError):
                validate_evaluation(row)

    def test_input_identity_and_case_cannot_be_overridden(self):
        trace_input = {
            "cases": [{"id": "a", "rows": 2, "cols": 64, "bucket": "small", "split": "train"}],
            "acceptance": {"min_speedup": 1.03, "max_regression": 0.02, "confidence": 0.95},
            "benchmark": {"timing": "cupti"},
        }
        row = evaluation()
        row["input_id"] = input_identity(trace_input)
        with self.assertRaises(EvidenceError):
            summarize_evaluations([row], trace_input=trace_input, acceptance={"min_speedup": 1.0})
        altered = copy.deepcopy(row)
        altered["cols"] = 128
        with self.assertRaises(EvidenceError):
            summarize_evaluations([altered], trace_input=trace_input)
        with self.assertRaises(EvidenceError):
            input_identity({**trace_input, "input_id": "forged"})


if __name__ == "__main__":
    unittest.main()
