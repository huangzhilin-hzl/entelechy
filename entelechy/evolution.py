# SPDX-License-Identifier: Apache-2.0
"""Snapshot, test and compare compiler revisions without changing the workload oracle."""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from .artifacts import (
    canonical_hash,
    compiler_fingerprint,
    evaluation_fingerprint,
    read_json,
    write_json,
)
from .compiler.trace import seed_program
from .evidence import paired_speedup
from .interaction import PROTOCOL, _load, action, training_rows
from .portfolio import build_portfolio
from .search import add_candidate, seed_schedules
from .trace_input import TraceInput, load_input, prepare_input, snapshot_input, verify_blobs


def _checkout() -> Path:
    root = Path(__file__).resolve().parents[1]
    if not (root / "pyproject.toml").is_file() or not (root / "tests").is_dir():
        raise ValueError(
            "compiler evolution needs a source checkout installed with pip install -e '.[dev]'"
        )
    return root


def _files(root: Path) -> dict[str, str]:
    paths = [
        p for p in (root / "entelechy").rglob("*") if p.is_file() and p.suffix in {".py", ".json"}
    ]
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
        if "__pycache__" not in p.parts
    }


def _tree_hash(root: Path) -> str:
    return canonical_hash(
        {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        }
    )


def begin(session: Path, evidence: Path, hypothesis: str, destination: Path) -> dict:
    session, evidence, destination = session.resolve(), evidence.resolve(), destination.resolve()
    trace_input, manifest = _load(session)
    if manifest["phase"] != "search":
        raise ValueError(
            "compiler evolution must use training evidence before the holdout boundary"
        )
    if not hypothesis.strip():
        raise ValueError("state the suspected missing capability or defect")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("evolution destination must be empty")
    event = read_json(evidence)
    digest = canonical_hash({k: v for k, v in event.items() if k != "event_id"})
    registered = any(
        (session / e["path"]).resolve() == evidence and e["event_id"] == digest
        for e in manifest["events"]
    )
    if not registered or event.get("event_id") != digest:
        raise ValueError("evidence is not an unchanged event from this session")
    if (
        event.get("command") not in {"loop submit", "loop evaluate"}
        or event.get("phase") != "search"
    ):
        raise ValueError("use a candidate submission or training evaluation event")
    root = _checkout()
    if any(destination.is_relative_to(root / name) for name in ("entelechy", "tests", "examples")):
        raise ValueError("store evolution artifacts outside package, tests and example sources")
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / "source"
    for name in _files(root):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    regression = destination / "regression"
    regression.mkdir()
    for name in ("tests", "examples"):
        shutil.copytree(
            root / name, regression / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    for name in ("LICENSE", "pyproject.toml"):
        shutil.copyfile(root / name, regression / name)
    write_json(destination / "evidence.json", event)
    if event.get("submitted_source_sha256"):
        shutil.copyfile(event["submitted_source"], destination / "rejected.ir.json")
    snapshot_input(trace_input, session / "dataset", destination)
    proposal = {
        "protocol": PROTOCOL,
        "status": "prepared",
        "session": str(session),
        "hypothesis": hypothesis,
        "input_id": trace_input.input_id,
        "base_compiler_id": manifest["compiler_id"],
        "base_evaluator_id": manifest["evaluator_id"],
        "base_sources": _files(root),
        "source_snapshot_hash": _tree_hash(source),
        "regression_hash": _tree_hash(regression),
        "evidence_id": digest,
        "rejected_ir_sha256": event.get("submitted_source_sha256"),
        "editable_package_paths": [
            "entelechy/compiler/*.py",
            "entelechy/search.py",
            "entelechy/guide.py",
        ],
        "required_work": [
            "Minimize the failure or identify a measured scheduling limitation.",
            "Update IR semantics, verifier, CUDA lowering and CLI spec together as needed.",
            "Add a regression test. Keep the oracle, workload and acceptance implementation fixed.",
        ],
        "next_actions": [
            action(
                "validate_revision",
                ["evolve", "validate", str(destination)],
                "After editing, run the frozen tests, current tests and fixed corpus.",
                editable=[str(root / "entelechy" / "compiler"), str(root / "tests")],
            )
        ],
    }
    proposal["proposal_id"] = canonical_hash(proposal)
    write_json(destination / "proposal.json", proposal)
    return proposal


def _proposal(root: Path) -> dict:
    proposal = read_json(root / "proposal.json")
    if proposal.get("proposal_id") != canonical_hash(
        {k: v for k, v in proposal.items() if k != "proposal_id"}
    ):
        raise ValueError("proposal was modified after creation")
    for name, key in (("source", "source_snapshot_hash"), ("regression", "regression_hash")):
        if _tree_hash(root / name) != proposal[key]:
            raise ValueError(f"frozen {name} snapshot changed")
    frozen_input = load_input(root / "input.json")
    if frozen_input.input_id != proposal["input_id"]:
        raise ValueError("evolution input changed")
    verify_blobs(frozen_input, root / "dataset")
    event = read_json(root / "evidence.json")
    if proposal.get("rejected_ir_sha256") and (
        not (root / "rejected.ir.json").is_file()
        or hashlib.sha256((root / "rejected.ir.json").read_bytes()).hexdigest()
        != proposal["rejected_ir_sha256"]
    ):
        raise ValueError("frozen rejected IR changed")
    if (
        canonical_hash({k: v for k, v in event.items() if k != "event_id"})
        != proposal["evidence_id"]
    ):
        raise ValueError("evolution evidence changed")
    return proposal


def _changes(proposal: dict) -> tuple[dict, list[str]]:
    current = _files(_checkout())
    before = proposal["base_sources"]
    changes = sorted(
        name for name in current.keys() | before.keys() if current.get(name) != before.get(name)
    )
    allowed = {"entelechy/search.py", "entelechy/guide.py"}
    forbidden = [
        name
        for name in changes
        if not name.startswith("entelechy/compiler/") and name not in allowed
    ]
    if forbidden:
        raise ValueError(f"protected evaluation implementation changed: {forbidden}")
    if not changes:
        raise ValueError("no compiler, search or CLI specification change to validate")
    return current, changes


def _run_tests(checkout: Path, tests: Path, config: Path, log: Path, timeout_s: float) -> dict:
    # Import the candidate implementation before pytest adds frozen-test paths.
    code = (
        "import sys; sys.path.insert(0, " + repr(str(checkout)) + "); "
        "import entelechy, pytest; "
        "raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', '-c', "
        + repr(str(config))
        + ", "
        + repr(str(tests))
        + "]))"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=checkout,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            exit_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return {"passed": False, "status": "timeout", "log": str(log)}
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    return {"passed": exit_code == 0, "exit_code": exit_code, "log": str(log)}


def validate(root: Path, *, timeout_s: float = 300) -> dict:
    root = root.resolve()
    if timeout_s <= 0:
        raise ValueError("timeout must be positive")
    proposal = _proposal(root)
    source_ids, changes = _changes(proposal)
    checkout = _checkout()
    directory = root / "checks" / uuid4().hex
    directory.mkdir(parents=True)
    frozen = root / "regression"
    tests = [
        _run_tests(
            checkout,
            frozen / "tests",
            frozen / "pyproject.toml",
            directory / "frozen-tests.log",
            timeout_s,
        ),
        _run_tests(
            checkout,
            checkout / "tests",
            checkout / "pyproject.toml",
            directory / "current-tests.log",
            timeout_s,
        ),
    ]
    corpus = []
    dataset = frozen / "examples" / "trace"
    inputs = [(load_input(root / "input.json"), root / "dataset")]
    for path in sorted((dataset / "definitions").glob("*.json")):
        name = read_json(path)["name"]
        inputs.append((prepare_input(dataset, name, name + "_torch_baseline"), dataset))
    seen = set()
    for trace_input, source in inputs:
        if trace_input.input_id in seen:
            continue
        seen.add(trace_input.input_id)
        try:
            case_root = directory / "corpus" / trace_input.input_id
            snapshot_input(trace_input, source, case_root)
            definition = trace_input.to_dict()["definition"]
            program = seed_program(definition)
            candidates = [
                add_candidate(case_root, trace_input, {**program, "schedule": schedule.to_dict()})
                for schedule in seed_schedules(definition["op_type"])[:12]
            ]
            corpus.append(
                {
                    "input_id": trace_input.input_id,
                    "definition": definition["name"],
                    "static_pass": bool(candidates),
                    "candidates": len(candidates),
                }
            )
        except (ValueError, RuntimeError) as error:
            corpus.append(
                {"input_id": trace_input.input_id, "static_pass": False, "error": str(error)}
            )
    _proposal(root)
    if _files(checkout) != source_ids:
        raise ValueError("candidate implementation changed during validation")
    passed = (
        bool(tests and corpus)
        and all(t["passed"] for t in tests)
        and all(c["static_pass"] for c in corpus)
    )
    result = {
        "protocol": PROTOCOL,
        "status": "static_validated" if passed else "rejected",
        "proposal_id": proposal["proposal_id"],
        "compiler_id": compiler_fingerprint(),
        "evaluator_id": evaluation_fingerprint(),
        "changed_sources": changes,
        "tests": tests,
        "corpus": corpus,
        "gpu_qualified": False,
        "next_actions": [
            action(
                "new_revision_trial",
                ["loop", "fork", proposal["session"], "--output", str(root / "trial")],
                "Reuse the frozen Trace input, baseline, policy and budget for the new compiler.",
            )
        ]
        if passed
        else [action("inspect_spec", ["spec", "evolution"], "Repair the failed gates; see logs.")],
    }
    result["validation_id"] = canonical_hash(result)
    write_json(directory / "validation.json", result)
    write_json(root / "validation.json", result)
    return result


def _selection_latencies(trace_input: TraceInput, manifest: dict, rows: list[dict]) -> dict:
    expected = {
        (c["candidate_id"], case["id"])
        for c in manifest["candidates"]
        for case in trace_input.cases("train")
    }
    if not expected or {(r["candidate_id"], r["case_id"]) for r in rows} != expected:
        raise ValueError("every submitted candidate needs complete training evidence")
    portfolio = build_portfolio(trace_input.to_dict(), rows)
    result = {}
    for case in trace_input.cases("train"):
        measured = [
            r
            for r in rows
            if r["case_id"] == case["id"]
            and r["status"] == "ok"
            and r.get("kernel_qualified") is True
            and r.get("baseline_samples_ms")
        ]
        if not measured:
            raise ValueError(
                "comparison requires strict measured GPU evidence for every training case"
            )
        selected = portfolio["routing"][case["bucket"]]
        if selected == "baseline":
            latency = statistics.fmean(statistics.fmean(r["baseline_samples_ms"]) for r in measured)
            normalized_speedup = 1.0
        else:
            row = next(r for r in measured if r["candidate_id"] == selected)
            latency = statistics.fmean(row["candidate_samples_ms"])
            normalized_speedup = paired_speedup(
                row["baseline_samples_ms"], row["candidate_samples_ms"]
            )["speedup"]
        result[case["id"]] = {
            "selected": selected,
            "latency_ms": latency,
            "baseline_normalized_speedup": normalized_speedup,
        }
    return result


def compare(root: Path, after: Path) -> dict:
    """A matched training comparison; held-out data cannot select compiler revisions."""
    root, after = root.resolve(), after.resolve()
    proposal = _proposal(root)
    _changes(proposal)
    validation = read_json(root / "validation.json")
    if validation.get("validation_id") != canonical_hash(
        {k: v for k, v in validation.items() if k != "validation_id"}
    ):
        raise ValueError("validation report changed")
    if (
        validation["status"] != "static_validated"
        or validation["evaluator_id"] != evaluation_fingerprint()
        or validation["proposal_id"] != proposal["proposal_id"]
    ):
        raise ValueError("validate this exact revision before comparing")
    before = Path(proposal["session"])
    trace_input, old = _load(before, current=False)
    new_input, new = _load(after)
    if old["phase"] not in {"search", "frozen"} or new["phase"] not in {"search", "frozen"}:
        raise ValueError("compiler selection must precede held-out evaluation")
    if trace_input.input_id != new_input.input_id or trace_input.input_id != proposal["input_id"]:
        raise ValueError("compiler comparison requires the unchanged Trace input")
    if old["evaluator_id"] != proposal["base_evaluator_id"]:
        raise ValueError("before session is not the snapshotted implementation")
    event = read_json(root / "evidence.json")
    new_capability = (
        not old["candidates"]
        and event.get("command") == "loop submit"
        and event.get("status") == "rejected"
    )
    if old["max_candidates"] != new["max_candidates"] or (
        not new_capability
        and (
            len(old["candidates"]) != len(new["candidates"])
            or sum(old["attempts"].values()) != sum(new["attempts"].values())
        )
    ):
        raise ValueError(
            "comparison requires matched budgets, submitted candidates and GPU attempts"
        )
    old_rows, new_rows = (
        training_rows(before, trace_input, old),
        training_rows(after, trace_input, new),
    )
    new_values = _selection_latencies(trace_input, new, new_rows)
    if new_capability:
        if old_rows or old["attempts"]:
            raise ValueError(
                "new-capability comparison requires an unmeasured rejected before trial"
            )
        old_values = {}
        for name, selected in new_values.items():
            rows = [
                r
                for r in new_rows
                if r["case_id"] == name
                and r["status"] == "ok"
                and r.get("kernel_qualified") is True
                and r.get("baseline_samples_ms")
            ]
            if selected["selected"] != "baseline":
                rows = [r for r in rows if r["candidate_id"] == selected["selected"]]
            old_values[name] = {
                "selected": "baseline",
                "measurement_origin": "after_trial",
                "latency_ms": statistics.fmean(
                    statistics.fmean(r["baseline_samples_ms"]) for r in rows
                ),
            }
    else:
        old_values = _selection_latencies(trace_input, old, old_rows)
    contexts = {
        canonical_hash({"baseline_id": r["baseline_id"], "environment": r["environment"]})
        for r in old_rows + new_rows
        if r["status"] == "ok" and r.get("baseline_samples_ms")
    }
    if len(contexts) != 1:
        raise ValueError("baseline or measurement environment differs between compiler trials")
    cases = [
        {
            "case_id": name,
            "before": old_values[name],
            "after": new_values[name],
            "observed_speedup": old_values[name]["latency_ms"] / new_values[name]["latency_ms"],
        }
        for name in sorted(old_values)
    ]
    result = {
        "protocol": PROTOCOL,
        "status": "compared",
        "proposal_id": proposal["proposal_id"],
        "before_evaluator_id": old["evaluator_id"],
        "after_evaluator_id": new["evaluator_id"],
        "cases": cases,
        "observed_geomean_speedup": math.exp(
            statistics.fmean(math.log(c["observed_speedup"]) for c in cases)
        ),
        "generalization_qualified": False,
        "automatic_promotion": False,
        "comparison_kind": "new_capability_vs_baseline" if new_capability else "matched_revisions",
        "interpretation": "Descriptive training comparison of selected routes. "
        "Across-revision runs are not paired; "
        "this is not a confidence bound or a held-out performance claim.",
        "next_actions": [
            action(
                "inspect_trial",
                ["loop", "status", str(after)],
                "Use the comparison to revise the compiler or finish search and freeze routing.",
            )
        ],
    }
    if new_capability:
        result["interpretation"] = (
            "The before trial admitted no candidate and has no kernel timing. Compare the new "
            "training-selected implementation with its measured baseline Solution. This does not "
            "establish speedup over an old generated kernel or prove the rejected IR was "
            "inexpressible."
        )
    write_json(root / f"comparison-{uuid4().hex}.json", result)
    return result
