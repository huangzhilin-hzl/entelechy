# SPDX-License-Identifier: Apache-2.0
"""An external agent drives individual steps through an evidence-backed CLI."""

from __future__ import annotations

import contextlib
import hashlib
import io
from functools import wraps
from pathlib import Path
from typing import Any

from .artifacts import (
    canonical_hash,
    compiler_fingerprint,
    evaluation_fingerprint,
    read_json,
    validate_candidate_artifact,
    write_json,
)
from .evidence import summarize_evaluations, validate_evaluation
from .executor import evaluate_candidate
from .portfolio import build_portfolio, evaluate_portfolio
from .search import add_candidate, analyze_schedule
from .trace_input import TraceInput, load_input, snapshot_input

PROTOCOL = "entelechy.agent.v1"


def _serialized(function):
    """A session has one writer; concurrent CLI calls cannot lose each other's events."""

    @wraps(function)
    def wrapped(root: Path, *args, **kwargs):
        import fcntl

        with (root / ".loop.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("another CLI operation owns this session") from error
            try:
                return function(root, *args, **kwargs)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    return wrapped


def action(name: str, argv: list[str], reason: str, **extra: Any) -> dict:
    return {"action": name, "argv": ["entelechy", *argv], "reason": reason, **extra}


def check_program(input_path: Path, ir_path: Path, *, analyze: bool = False) -> dict:
    """Construction errors are feedback; static acceptance never implies GPU agreement."""
    trace_input = load_input(input_path)
    try:
        raw = trace_input.to_dict()
        from .compiler.trace import dimensions, validate_program

        program = read_json(ir_path)
        schedule = validate_program(program, raw["definition"])
        for trace in raw["workloads"]:
            dimensions(raw["definition"], trace["workload"], program)
        findings = []
    except (ValueError, TypeError, KeyError) as error:
        schedule = None
        findings = [{"code": "IR_CONSTRUCTION", "path": "program", "message": str(error)}]
    result = {
        "protocol": PROTOCOL,
        "status": "rejected" if findings else "static_valid",
        "input_id": trace_input.input_id,
        "compiler_id": compiler_fingerprint(),
        "evaluator_id": evaluation_fingerprint(),
        "findings": findings,
        "gpu_checked": False,
        "performance_claim": None,
        "editable": [str(ir_path.resolve())],
        "next_actions": [action("read_ir", ["spec", "ir"], "Inspect the executable vocabulary.")],
    }
    if analyze and not findings:
        result["analysis"] = analyze_schedule(schedule, trace_input, program=program)
    return result


def _load(root: Path, *, current: bool = True) -> tuple[TraceInput, dict]:
    trace_input = load_input(root / "input.json")
    manifest = read_json(root / "experiment.json")
    if manifest.get("mode") != "agent_cli":
        raise ValueError("expected an agent CLI session created by loop init")
    if manifest["input_id"] != trace_input.input_id:
        raise ValueError("input snapshot changed after session creation")
    if current and (
        manifest["compiler_id"] != compiler_fingerprint()
        or manifest["evaluator_id"] != evaluation_fingerprint()
    ):
        raise ValueError(
            "implementation changed: retain this session as evidence; use evolve validate "
            "on the proposal created before editing, then loop fork into a fresh directory"
        )
    ids = [c["candidate_id"] for c in manifest["candidates"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate IDs")
    for item in manifest["events"]:
        event_path = (root / item["path"]).resolve()
        if not event_path.is_relative_to(root.resolve()):
            raise ValueError("event path escapes the session")
        event = read_json(event_path)
        digest = canonical_hash({k: v for k, v in event.items() if k != "event_id"})
        if (
            digest != item["event_id"]
            or digest != event.get("event_id")
            or event.get("input_id") != manifest["input_id"]
            or event.get("evaluator_id") != manifest["evaluator_id"]
        ):
            raise ValueError("session event is no longer unchanged evidence")
        if event.get("command") == "loop submit" and event.get("status") == "rejected":
            rejected = (root / event["submitted_source"]).resolve()
            if (
                not rejected.is_relative_to(root.resolve())
                or not rejected.is_file()
                or hashlib.sha256(rejected.read_bytes()).hexdigest()
                != event.get("submitted_source_sha256")
            ):
                raise ValueError("rejected IR changed after submission")
    if current:
        for candidate in manifest["candidates"]:
            validate_candidate_artifact(
                root, trace_input.input_id, manifest["compiler_id"], candidate
            )
            if candidate["evaluator_id"] != manifest["evaluator_id"]:
                raise ValueError("candidate evaluator mismatch")
    else:
        # A new verifier may not understand a historical IR. Check its recorded source bytes.
        for candidate in manifest["candidates"]:
            source = (root / candidate["source"]).resolve()
            if not source.is_relative_to(root.resolve()) or not source.is_file():
                raise ValueError("historical source is outside the session or missing")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if (
                candidate["source_sha256"] != digest
                or candidate["candidate_id"] != digest[:24]
                or candidate["evaluator_id"] != manifest["evaluator_id"]
            ):
                raise ValueError("historical source identity mismatch")
    return trace_input, manifest


def training_rows(root: Path, trace_input: TraceInput, manifest: dict) -> list[dict]:
    """Recheck observations against the original candidate and evaluator identities."""
    rows = read_json(root / "search_results.json")
    candidates = {c["candidate_id"]: c for c in manifest["candidates"]}
    expected = {c["id"] for c in trace_input.cases("train")}
    seen = set()
    for row in rows:
        validate_evaluation(row, input_id=trace_input.input_id)
        candidate = candidates.get(row["candidate_id"])
        key = row["candidate_id"], row["case_id"]
        if (
            candidate is None
            or row["split"] != "train"
            or row["case_id"] not in expected
            or key in seen
            or row.get("source_sha256") != candidate["source_sha256"]
            or row.get("evaluator_id") != manifest["evaluator_id"]
        ):
            raise ValueError("training evidence identity or coverage mismatch")
        seen.add(key)
    for candidate_id in {r["candidate_id"] for r in rows}:
        if {r["case_id"] for r in rows if r["candidate_id"] == candidate_id} != expected:
            raise ValueError("incomplete training evidence")
    return rows


def _actions(root: Path, manifest: dict, rows: list[dict]) -> list[dict]:
    session = str(root)
    phase = manifest["phase"]
    device_args = ["--device", str(manifest.get("device", 0))]
    if phase == "frozen":
        return [
            action(
                "holdout",
                ["loop", "holdout", session, *device_args],
                "Evaluate the frozen routes once.",
            )
        ]
    if phase == "complete":
        if read_json(root / "report.json").get("qualified"):
            return [
                action(
                    "export",
                    ["loop", "export", session, "--output", session + "-export"],
                    "Export verified source and evidence for integration review.",
                )
            ]
        return [
            action(
                "read_result",
                ["report", session],
                "Holdout rejected this portfolio; do not repair routes on these holdouts.",
            )
        ]
    if phase != "search":
        return [
            action(
                "inspect",
                ["loop", "status", session],
                "This session is interrupted or running; inspect the retained artifacts.",
            )
        ]
    result = []
    for candidate in manifest["candidates"]:
        selected = [r for r in rows if r["candidate_id"] == candidate["candidate_id"]]
        if not selected or all(r["status"] == "unavailable" for r in selected):
            result.append(
                action(
                    "evaluate",
                    [
                        "loop",
                        "evaluate",
                        session,
                        "--candidate",
                        candidate["candidate_id"],
                        *device_args,
                    ],
                    "Check correctness before measuring; retry only unavailable runs.",
                )
            )
    if len(manifest["candidates"]) < manifest["max_candidates"]:
        result.append(
            action(
                "submit",
                ["loop", "submit", session, "--ir", str(root / "draft.ir.json")],
                "Edit the draft IR, then verify and generate a new CUDA candidate.",
                editable=[str(root / "draft.ir.json")],
            )
        )
    completed = {r["candidate_id"] for r in rows}
    if (
        manifest["candidates"]
        and all(c["candidate_id"] in completed for c in manifest["candidates"])
        and any(r["status"] == "ok" for r in rows)
    ):
        result.append(
            action(
                "freeze",
                ["loop", "freeze", session],
                "Finish training selection before accessing held-out results.",
            )
        )
    return result


def _feedback(root: Path, manifest: dict, findings: list[dict], evidence: str) -> list[dict]:
    """Route evidence to a repair scope without claiming a proven root cause."""
    for finding in findings:
        code = str(finding.get("code", ""))
        environment = "unavailable" in code or code == "target_sm_mismatch"
        finding["repair_scope"] = "environment" if environment else "candidate_or_compiler"
        finding["root_cause_proven"] = False
    if not findings or manifest["phase"] != "search":
        return []
    result = [
        action(
            "inspect_ir",
            ["spec", "ir"],
            "Compare the failing schedule with supported semantics before changing code.",
        )
    ]
    if all(f["repair_scope"] == "environment" for f in findings):
        return [
            action(
                "doctor",
                ["doctor", "--device", str(manifest.get("device", 0))],
                "Resolve device, baseline or timer availability.",
            )
        ]
    result.append(
        action(
            "compiler_investigation",
            [
                "evolve",
                "begin",
                str(root),
                "--evidence",
                str(root / evidence),
                "--hypothesis",
                "Investigate this finding; minimize it before extending the compiler.",
                "--output",
                str(root / f"evolution-{len(manifest['events']) + 1}"),
            ],
            "Snapshot code and regression tests BEFORE editing IR, verification or CUDA lowering.",
        )
    )
    return result


def _emit(root: Path, manifest: dict, command: str, status: str, **data: Any) -> dict:
    trace_input = load_input(root / "input.json")
    rows = training_rows(root, trace_input, manifest)
    path = f"events/{len(manifest['events']) + 1:06d}.json"
    findings = data.pop("findings", [])
    next_actions = _actions(root, manifest, rows)
    next_actions += _feedback(root, manifest, findings, path)
    result = {
        "protocol": PROTOCOL,
        "command": command,
        "status": status,
        "phase": manifest["phase"],
        "session": str(root),
        "input_id": manifest["input_id"],
        "compiler_id": manifest["compiler_id"],
        "evaluator_id": manifest["evaluator_id"],
        "findings": findings,
        "next_actions": next_actions,
        "evidence_path": str(root / path),
        "budget": {
            "candidates": manifest["max_candidates"],
            "submitted": len(manifest["candidates"]),
        },
        **data,
    }
    result["definition"] = trace_input.to_dict()["definition"]["name"]
    result["event_id"] = canonical_hash(result)
    write_json(root / path, result)
    manifest["events"].append({"path": path, "event_id": result["event_id"]})
    write_json(root / "experiment.json", manifest)
    return result


def init_loop(dataset: Path, root: Path, *, value: TraceInput, budget: int = 12) -> dict:
    root = root.resolve()
    trace_input = value
    if type(budget) is not int or not 1 <= budget <= 10000:
        raise ValueError("budget must be in [1, 10000]")
    if root.exists() and any(root.iterdir()):
        raise ValueError("session directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    from .compiler.trace import draft_program

    snapshot_input(trace_input, dataset, root)
    draft = draft_program(trace_input.to_dict()["definition"])
    write_json(root / "search_results.json", [])
    write_json(root / "draft.ir.json", draft)
    manifest = {
        "schema_version": 1,
        "mode": "agent_cli",
        "status": "interactive",
        "phase": "search",
        "input_id": trace_input.input_id,
        "compiler_id": compiler_fingerprint(),
        "evaluator_id": evaluation_fingerprint(),
        "max_candidates": budget,
        "candidates": [],
        "events": [],
        "attempts": {},
        "performance_claim": None,
    }
    return _emit(
        root,
        manifest,
        "loop init",
        "ready",
        reference={
            "kind": "flashinfer_definition",
            "gpu_validated": False,
            "definition": trace_input.to_dict()["definition"]["name"],
        },
    )


def fork_loop(previous: Path, root: Path) -> dict:
    trace_input, manifest = _load(previous.resolve(), current=False)
    if manifest["phase"] not in {"search", "frozen"}:
        raise ValueError("a new compiler trial must precede held-out evaluation")
    return init_loop(
        previous / "dataset", root, budget=manifest["max_candidates"], value=trace_input
    )


def loop_status(root: Path) -> dict:
    root = root.resolve()
    trace_input, manifest = _load(root)
    rows = training_rows(root, trace_input, manifest)
    last = read_json(root / manifest["events"][-1]["path"]) if manifest["events"] else None
    return {
        "protocol": PROTOCOL,
        "status": manifest["status"],
        "phase": manifest["phase"],
        "input_id": trace_input.input_id,
        "compiler_id": manifest["compiler_id"],
        "evaluator_id": manifest["evaluator_id"],
        "candidates": manifest["candidates"],
        "events": manifest["events"],
        "last_feedback": last,
        "next_actions": _actions(root, manifest, rows),
        "budget": {
            "candidates": manifest["max_candidates"],
            "submitted": len(manifest["candidates"]),
        },
    }


@_serialized
def submit(root: Path, ir_path: Path) -> dict:
    root = root.resolve()
    trace_input, manifest = _load(root)
    if manifest["phase"] != "search":
        raise ValueError("new candidates are forbidden after routing is frozen")
    inspection = check_program((root / "input.json"), ir_path, analyze=True)
    if inspection["status"] == "rejected":
        # Preserve the actual failed input; an agent may edit the draft immediately afterwards.
        rejected = root / "rejected" / f"{len(manifest['events']) + 1}.json"
        rejected.parent.mkdir(parents=True, exist_ok=True)
        rejected.write_bytes(ir_path.read_bytes())
        return _emit(
            root,
            manifest,
            "loop submit",
            "rejected",
            findings=inspection["findings"],
            submitted_source=str(rejected),
            submitted_source_sha256=hashlib.sha256(rejected.read_bytes()).hexdigest(),
        )
    if len(manifest["candidates"]) >= manifest["max_candidates"]:
        raise ValueError("candidate budget exhausted")
    candidate = add_candidate(root, trace_input, read_json(ir_path))
    duplicate = any(c["candidate_id"] == candidate["candidate_id"] for c in manifest["candidates"])
    if not duplicate:
        manifest["candidates"].append(candidate)
    return _emit(
        root,
        manifest,
        "loop submit",
        "duplicate" if duplicate else "generated",
        candidate=candidate,
        gpu_checked=False,
        performance_claim=None,
    )


def _device(manifest: dict, requested: int | None, *, fixed: bool) -> int:
    previous = manifest.get("device", 0)
    selected = previous if requested is None else requested
    if type(selected) is not int or selected < 0:
        raise ValueError("device must be a nonnegative integer")
    if fixed and selected != previous:
        raise ValueError("device cannot change after measured training evidence")
    return selected


@_serialized
def evaluate(
    root: Path, candidate_id: str, *, device: int | None = None, timeout_s: float = 300
) -> dict:
    root = root.resolve()
    trace_input, manifest = _load(root)
    if manifest["phase"] != "search":
        raise ValueError("training evaluation is forbidden after routing is frozen")
    candidate = next((c for c in manifest["candidates"] if c["candidate_id"] == candidate_id), None)
    if candidate is None:
        raise ValueError("candidate is not admitted; use loop submit first")
    previous = training_rows(root, trace_input, manifest)
    device = _device(manifest, device, fixed=any(r["status"] == "ok" for r in previous))
    selected = [r for r in previous if r["candidate_id"] == candidate_id]
    if selected and not all(r["status"] == "unavailable" for r in selected):
        raise ValueError("candidate was already evaluated; only unavailable attempts can retry")
    attempt = manifest["attempts"].get(candidate_id, 0) + 1
    manifest["device"] = device
    manifest["attempts"][candidate_id] = attempt
    write_json(root / "experiment.json", manifest)
    with contextlib.redirect_stdout(io.StringIO()):
        rows = evaluate_candidate(
            root,
            trace_input,
            candidate,
            "train",
            device=device,
            timeout_s=timeout_s,
            attempt=attempt,
        )
    write_json(
        root / "search_results.json",
        [r for r in previous if r["candidate_id"] != candidate_id] + rows,
    )
    summary = summarize_evaluations(
        rows,
        trace_input=trace_input.to_dict(),
        expected_case_ids=[c["id"] for c in trace_input.cases("train")],
    )
    statuses = {r["status"] for r in rows}
    status = (
        "evaluated"
        if statuses == {"ok"}
        else "unavailable"
        if statuses == {"unavailable"}
        else "failed"
    )
    findings = [dict(d) for r in rows for d in r.get("diagnostics", [])]
    if status == "evaluated" and not summary["qualified"]:
        findings.append(
            {
                "code": "PERFORMANCE_TARGET_NOT_MET",
                "path": "summary.cases",
                "message": "Inspect per-case timings and confidence bounds; propose a scheduling "
                "change or investigate an IR capability gap. "
                "Bottleneck attribution is unavailable.",
            }
        )
    return _emit(
        root,
        manifest,
        "loop evaluate",
        status,
        findings=findings,
        candidate_id=candidate_id,
        device=device,
        summary=summary,
        measurement=str(root / "evaluations" / candidate_id / "train" / str(attempt)),
        interpretation="A mismatch needs localization; it does not establish a compiler bug.",
    )


@_serialized
def freeze(root: Path) -> dict:
    root = root.resolve()
    trace_input, manifest = _load(root)
    if manifest["phase"] != "search":
        raise ValueError("routing may be frozen only once")
    rows = training_rows(root, trace_input, manifest)
    completed = {r["candidate_id"] for r in rows}
    if not manifest["candidates"] or any(
        c["candidate_id"] not in completed for c in manifest["candidates"]
    ):
        raise ValueError("evaluate every submitted candidate before freezing")
    if not any(r["status"] == "ok" and r.get("baseline_samples_ms") for r in rows):
        raise ValueError("cannot freeze without measured training evidence")
    portfolio = build_portfolio(trace_input.to_dict(), rows, fallback_id="baseline")
    write_json(root / "portfolio.json", portfolio)
    write_json(root / "portfolio.freeze.json", {"sha256": canonical_hash(portfolio)})
    manifest["phase"] = "frozen"
    return _emit(root, manifest, "loop freeze", "frozen", portfolio=portfolio)


@_serialized
def holdout(root: Path, *, device: int | None = None, timeout_s: float = 300) -> dict:
    root = root.resolve()
    trace_input, manifest = _load(root)
    if manifest["phase"] != "frozen":
        raise ValueError("holdout requires frozen routing and is single-use")
    device = _device(manifest, device, fixed=True)
    portfolio = read_json(root / "portfolio.json")
    if canonical_hash(portfolio) != read_json(root / "portfolio.freeze.json")["sha256"]:
        raise ValueError("routing changed after freeze")
    rebuilt = build_portfolio(trace_input.to_dict(), training_rows(root, trace_input, manifest))
    if canonical_hash(rebuilt) != canonical_hash(portfolio):
        raise ValueError("routing no longer matches its training evidence")
    manifest["phase"] = "holdout_running"
    write_json(root / "experiment.json", manifest)
    selected = {c["candidate_id"]: c for c in manifest["candidates"]}
    rows = []
    try:
        for candidate_id in sorted(set(portfolio["routing"].values())):
            candidate = (
                {**manifest["candidates"][0], "candidate_id": "baseline"}
                if candidate_id == "baseline"
                else selected[candidate_id]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                rows.extend(
                    evaluate_candidate(
                        root, trace_input, candidate, "holdout", device=device, timeout_s=timeout_s
                    )
                )
        if canonical_hash(read_json(root / "portfolio.json")) != canonical_hash(portfolio):
            raise ValueError("routing changed during holdout")
        _load(root)
        write_json(root / "holdout_results.json", rows)
        report = evaluate_portfolio(trace_input.to_dict(), portfolio, rows)
        report.update(
            status="completed",
            compiler_id=manifest["compiler_id"],
            evaluator_id=manifest["evaluator_id"],
            deployment_qualified=False,
        )
        write_json(root / "report.json", report)
        manifest.update(
            phase="complete",
            status="completed",
            performance_claim=report.get("geomean_speedup") if report["qualified"] else None,
        )
        return _emit(
            root,
            manifest,
            "loop holdout",
            "qualified" if report["qualified"] else "rejected",
            report=report,
        )
    except BaseException:
        manifest.update(phase="interrupted", status="failed")
        write_json(root / "experiment.json", manifest)
        raise
