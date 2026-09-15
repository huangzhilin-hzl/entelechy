# SPDX-License-Identifier: Apache-2.0
"""A frozen compiler per search, then frozen routing before held-out evaluation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from . import __version__
from .artifacts import (
    canonical_hash,
    read_json,
    validate_candidate_artifact,
    write_json,
)
from .interaction import _load, training_rows
from .licensing import project_license_text
from .portfolio import build_portfolio, evaluate_portfolio
from .trace_input import snapshot_input, verify_blobs


def export_portfolio(run_dir: Path, destination: Path) -> dict[str, Any]:
    """Package measured source for review; deployment qualification is a separate gate."""
    run_dir = run_dir.resolve()
    trace_input, manifest = _load(run_dir)
    report = read_json(run_dir / "report.json")
    if manifest["status"] != "completed" or not report.get("qualified"):
        raise ValueError("export requires a completed, held-out-qualified portfolio")
    portfolio = read_json(run_dir / "portfolio.json")
    if canonical_hash(portfolio) != read_json(run_dir / "portfolio.freeze.json")["sha256"]:
        raise ValueError("portfolio no longer matches its pre-holdout freeze")
    rebuilt = build_portfolio(
        trace_input.to_dict(),
        training_rows(run_dir, trace_input, manifest),
        fallback_id="baseline",
    )
    if canonical_hash(rebuilt) != canonical_hash(portfolio):
        raise ValueError("portfolio no longer matches its training-only selection evidence")
    # Recompute acceptance from raw observations, rather than trusting an edited report flag.
    verified = evaluate_portfolio(
        trace_input.to_dict(), portfolio, read_json(run_dir / "holdout_results.json")
    )
    if not verified.get("qualified"):
        raise ValueError("raw held-out evidence does not qualify for export")
    selected = set(portfolio["routing"].values()) - {"baseline"}
    available = {c["candidate_id"] for c in manifest["candidates"]}
    if not selected or not selected <= available:
        raise ValueError("every selected non-fallback candidate needs a source artifact")
    # Validate every source before creating a partial export directory.
    source_paths = {
        candidate["candidate_id"]: validate_candidate_artifact(
            run_dir,
            trace_input.input_id,
            manifest["compiler_id"],
            candidate,
        )
        for candidate in manifest["candidates"]
        if candidate["candidate_id"] in selected
    }
    for row in read_json(run_dir / "holdout_results.json"):
        if row.get("evaluator_id") != manifest["evaluator_id"]:
            raise ValueError("held-out evaluator identity mismatch")
        if row["candidate_id"] in selected:
            source_candidate = next(
                c for c in manifest["candidates"] if c["candidate_id"] == row["candidate_id"]
            )
            if row.get("source_sha256") != source_candidate["source_sha256"]:
                raise ValueError("held-out source identity mismatch")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("export destination must be empty")
    verify_blobs(trace_input, run_dir / "dataset")
    license_text = project_license_text()
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "LICENSE").write_text(license_text, encoding="utf-8")
    files = {}
    for candidate in manifest["candidates"]:
        if candidate["candidate_id"] not in selected:
            continue
        source = source_paths[candidate["candidate_id"]]
        target = destination / f"{candidate['candidate_id']}.cu"
        shutil.copyfile(source, target)
        files[target.name] = candidate["source_sha256"]
    for name in (
        "portfolio.json",
        "report.json",
        "search_results.json",
        "holdout_results.json",
    ):
        shutil.copyfile(run_dir / name, destination / name)
    snapshot_input(trace_input, run_dir / "dataset", destination)
    for candidate in manifest["candidates"]:
        if candidate["candidate_id"] in selected:
            shutil.copyfile(
                run_dir / candidate["solution"],
                destination / "dataset" / "solutions" / f"{candidate['candidate_id']}.json",
            )
    traces = destination / "dataset" / "traces"
    traces.mkdir(exist_ok=True)
    import json

    for index, path in enumerate(sorted((run_dir / "evaluations").rglob("result.json"))):
        exported_names = {"entelechy_" + candidate_id for candidate_id in selected}
        exported_names.add(trace_input.to_dict()["baseline_solution"]["name"])
        native = [
            t for t in read_json(path).get("native_traces", []) if t["solution"] in exported_names
        ]
        if native:
            (traces / f"{index}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in native))
    exported = {
        "schema_version": 1,
        "package_version": __version__,
        "license": "Apache-2.0",
        "source_repository": "https://github.com/huangzhilin-hzl/entelechy",
        "input_id": trace_input.input_id,
        "compiler_id": manifest["compiler_id"],
        "evaluator_id": manifest["evaluator_id"],
        "sources": files,
        "status": "review_artifact",
        "deployment_qualified": False,
        "source_runtime_dependency": "CUDA; ENTELECHY_STANDALONE omits the PyTorch binding",
    }
    write_json(destination / "export.json", exported)
    (destination / "README.md").write_text(
        "# Entelechy measured kernel export\n\n"
        "This is a review artifact, not an installed backend. The JSON portfolio assigns the "
        "session policy's explicit buckets; it is not an arbitrary-shape production dispatcher.\n\n"
        "Sources do not import Entelechy. Define ENTELECHY_STANDALONE to omit the PyTorch "
        "binding when integrating the CUDA kernel in another library.\n\n"
        "Run sanitizers, graph replay, adapter tests and end-to-end measurements "
        "before deployment.\n\n"
        "Generated by [Entelechy](https://github.com/huangzhilin-hzl/entelechy). "
        "The generated sources are licensed under [Apache-2.0](LICENSE). "
        "Preserve the license and source notices when redistributing them.\n",
        encoding="utf-8",
    )
    return exported
