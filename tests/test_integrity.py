# SPDX-License-Identifier: Apache-2.0
"""Artifact checks use real planned sources, without compiling or running CUDA."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from helpers import init_example

from entelechy import interaction
from entelechy.artifacts import (
    evaluation_fingerprint,
    read_json,
    source_hash,
    validate_candidate_artifact,
)


class CandidateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="entelechy-integrity-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.run_dir = self.root / "run"
        self.trace_input = init_example(self.run_dir, "rmsnorm")
        self.candidate = interaction.submit(self.run_dir, self.run_dir / "draft.ir.json")[
            "candidate"
        ]
        self.manifest = read_json(self.run_dir / "experiment.json")

    def validate(self, candidate=None, **kwargs):
        return validate_candidate_artifact(
            self.run_dir,
            self.trace_input.input_id,
            self.manifest["compiler_id"],
            self.candidate if candidate is None else candidate,
            **kwargs,
        )

    def test_real_planned_artifact_passes(self):
        self.assertEqual(self.validate(), (self.run_dir / self.candidate["source"]).resolve())

    def test_altered_source_fails_even_if_it_still_contains_valid_cuda(self):
        source = self.run_dir / self.candidate["source"]
        source.write_text(source.read_text() + "\n// later edit\n")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.validate()

    def test_updating_source_and_digest_cannot_reuse_old_candidate_evidence(self):
        source = self.run_dir / self.candidate["source"]
        altered = source.read_text() + "\n// replacement source\n"
        source.write_text(altered)
        candidate = copy.deepcopy(self.candidate)
        candidate["source_sha256"] = source_hash(altered)
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            self.validate(candidate)

    def test_fixed_input_and_compiler_are_required(self):
        for field in ("input_id", "compiler_id"):
            candidate = copy.deepcopy(self.candidate)
            candidate[field] = "different"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                self.validate(candidate)

    def test_schedule_identity_and_canonical_form_are_required(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["program"]["schedule"]["threads"] = 64
        with self.assertRaisesRegex(ValueError, "program identity"):
            self.validate(candidate)
        candidate = copy.deepcopy(self.candidate)
        candidate["program"]["schedule"].pop("barriers")
        with self.assertRaisesRegex(ValueError, "canonical"):
            self.validate(candidate)

    def test_outside_and_symlink_sources_are_rejected(self):
        outside = self.root / "outside.cu"
        outside.write_bytes((self.run_dir / self.candidate["source"]).read_bytes())
        for relative in ("../outside.cu", "link.cu", str(outside)):
            if relative == "link.cu":
                (self.run_dir / relative).symlink_to(outside)
            candidate = copy.deepcopy(self.candidate)
            candidate["source"] = relative
            with self.subTest(path=relative), self.assertRaises(ValueError):
                self.validate(candidate)

    def test_candidate_id_cannot_be_used_as_a_path(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["candidate_id"] = "../../outside"
        with self.assertRaisesRegex(ValueError, "candidate_id"):
            self.validate(candidate)

    def test_baseline_exception_is_explicit_and_keeps_source_checks(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["candidate_id"] = "baseline"
        with self.assertRaisesRegex(ValueError, "reserved baseline"):
            self.validate(candidate)
        self.assertTrue(self.validate(candidate, allow_baseline=True).is_file())
        source = self.run_dir / candidate["source"]
        source.write_text("altered")
        with self.assertRaisesRegex(ValueError, "modified"):
            self.validate(candidate, allow_baseline=True)


class EvaluationFingerprintTests(unittest.TestCase):
    def test_oracle_and_acceptance_changes_invalidate_fingerprint(self):
        with tempfile.TemporaryDirectory(prefix="entelechy-fingerprint-") as directory:
            root = Path(directory)
            (root / "runtime").mkdir()
            (root / "artifacts.py").write_text("# artifacts\n")
            (root / "runtime" / "worker.py").write_text("# oracle v1\n")
            (root / "evidence.py").write_text("# acceptance v1\n")
            with patch("entelechy.artifacts.__file__", str(root / "artifacts.py")):
                initial = evaluation_fingerprint()
                self.assertEqual(evaluation_fingerprint(), initial)
                (root / "runtime" / "worker.py").write_text("# oracle v2\n")
                changed_oracle = evaluation_fingerprint()
                self.assertNotEqual(changed_oracle, initial)
                (root / "evidence.py").write_text("# acceptance v2\n")
                self.assertNotEqual(evaluation_fingerprint(), changed_oracle)

    def test_pycache_and_non_source_files_do_not_change_fingerprint(self):
        with tempfile.TemporaryDirectory(prefix="entelechy-fingerprint-") as directory:
            root = Path(directory)
            (root / "artifacts.py").write_text("# artifacts\n")
            with patch("entelechy.artifacts.__file__", str(root / "artifacts.py")):
                initial = evaluation_fingerprint()
                (root / "__pycache__").mkdir()
                (root / "__pycache__" / "ignored.py").write_text("# ignored cache content\n")
                (root / "result.json").write_text("{}")
                self.assertEqual(evaluation_fingerprint(), initial)


if __name__ == "__main__":
    unittest.main()


def test_native_candidate_cannot_downgrade_to_source_only_validation(tmp_path):
    root = tmp_path / "session"
    value = init_example(root)
    candidate = interaction.submit(root, root / "draft.ir.json")["candidate"]
    candidate.pop("program")
    with pytest.raises(ValueError, match="requires a native program"):
        validate_candidate_artifact(root, value.input_id, candidate["compiler_id"], candidate)
