# SPDX-License-Identifier: Apache-2.0
"""Installed, queryable instructions for agents owning the optimization loop."""

from __future__ import annotations

from dataclasses import fields

from . import __version__
from .compiler import Schedule
from .compiler.ir import Barrier, SharedResource
from .compiler.verify import OPERATORS
from .interaction import PROTOCOL
from .search import seed_schedules

SPEC_TOPICS = ("trace", "ir", "feedback", "evolution")


def tutorial(topic: str = "loop") -> dict:
    lessons = {
        "loop": {
            "purpose": (
                "The external agent owns decisions and file edits. The CLI supplies "
                "executable steps, evidence and next actions."
            ),
            "task_ownership": {
                "person": "Provides tasks, dataset, remote access and any specific requirements.",
                "agent": "Resolves the requested tasks and executes the CLI loop and source edits.",
                "scope": (
                    "Use the user's selected workloads and priorities. Bundled fixtures are "
                    "regression inputs, not an optimization task list."
                ),
            },
            "task_setup": {
                "selection": (
                    "Inspect the dataset with trace inspect DATASET, then --definition NAME. "
                    "Resolve the requested workloads and create one session per Definition "
                    "in the user's order."
                ),
                "baseline": (
                    "Honor the requested baseline or selection rule. If omitted, use the "
                    "sole matching Solution only when unambiguous; ask the user if the "
                    "baseline is missing or ambiguous. Pass its exact name to --baseline."
                ),
                "defaults": (
                    "Honor supplied budget, benchmark and policy settings. Otherwise omit "
                    "--budget, --bench-config and --policy to use CLI defaults; inspect "
                    "loop init --help and spec trace. Keep reference and policy fixed "
                    "throughout the experiment."
                ),
                "directories": (
                    "Use requested output paths. If omitted, choose unused persistent remote "
                    "experiment and local results directories and report both locations."
                ),
            },
            "execution_environment": {
                "agent": "Runs locally; local CLI discovery does not require a GPU.",
                "experiments": (
                    "Use the supplied Pod, Modal or other remote tools to invoke Entelechy "
                    "in the GPU environment. Run doctor there. The CLI executes on its "
                    "current host; the agent's tools handle remote jobs and file transfer."
                ),
                "state": (
                    "Keep loop/evolve commands and authoritative session state in one "
                    "persistent remote workspace. Interpret next_actions paths there. "
                    "Run evolve begin before syncing compiler edits; then validate and "
                    "measure that revision remotely."
                ),
                "results": (
                    "Read remote JSON feedback and retrieve logs and artifacts. Copy final "
                    "experiments and qualified exports to the user's local results directory."
                ),
            },
            "session_files": {
                "experiment.json": "Session state, budget, candidates and code identities.",
                "search_results.json": "Candidate correctness and measurements used during search.",
                "holdout_results.json": "Validation results after routing is frozen.",
            },
            "steps": [
                {
                    "step": "describe",
                    "command": (
                        "loop init DATASET --definition NAME --baseline SOLUTION --output SESSION"
                    ),
                    "result": (
                        "Snapshot native Definition, Workloads and a baseline Solution; "
                        "create draft.ir.json with schedule and bindings. "
                        "Read spec trace and spec ir."
                    ),
                },
                {
                    "step": "inspect",
                    "command": "check SESSION --ir IR; analyze SESSION --ir IR",
                    "result": (
                        "Inspect static legality and resource facts. Unknown performance "
                        "stays unknown."
                    ),
                },
                {
                    "step": "generate",
                    "command": "loop submit SESSION --ir IR",
                    "result": (
                        "Validate and snapshot the IR, then use our codegen to emit CUDA. No "
                        "hand-authored runtime twin is needed."
                    ),
                },
                {
                    "step": "evaluate",
                    "command": "loop evaluate SESSION --candidate ID",
                    "result": (
                        "Build native Solutions, check both against Definition.reference using "
                        "FlashInfer's evaluator and actual Workload inputs, then collect paired "
                        "baseline/candidate measurements and native result Traces."
                    ),
                },
                {
                    "step": "repair",
                    "command": (
                        "Edit the draft IR and repeat loop submit/evaluate, or use evolve "
                        "begin before compiler edits."
                    ),
                    "result": (
                        "Candidate repairs use the same compiler. Compiler repairs create a "
                        "new implementation identity and trial. Read tutorial compiler "
                        "before editing compiler source."
                    ),
                },
                {
                    "step": "select",
                    "command": "loop freeze SESSION",
                    "result": (
                        "Select routes using training evidence. Stop candidate changes before"
                        " held-out evaluation."
                    ),
                },
                {
                    "step": "validate",
                    "command": "loop holdout SESSION",
                    "result": (
                        "Evaluate fixed routes once; a failed holdout cannot be used to "
                        "reselect routes in this session."
                    ),
                },
                {
                    "step": "deliver",
                    "command": "loop export SESSION --output EXPORT",
                    "result": (
                        "Recheck raw evidence and export source, hashes, native Trace data and "
                        "license. Production integration still needs its own gates."
                    ),
                },
            ],
            "first_commands": ["entelechy spec", "entelechy spec ir", "entelechy loop init --help"],
            "delivery": (
                "Return experiment locations, correctness results, measured baseline "
                "comparisons and reports. Export qualified results. For blocked or "
                "unqualified tasks, retain evidence and explain the outcome."
            ),
            "stopping_rules": [
                (
                    "An unavailable GPU, baseline or timer requires an environment repair, "
                    "not an IR mutation."
                ),
                (
                    "An exhausted candidate budget stops new submissions. Use measured "
                    "evidence to freeze or retain the unsuccessful experiment."
                ),
                (
                    "A failed holdout closes this validation attempt. Fresh unseen cases are "
                    "needed for a new generalization claim."
                ),
            ],
        },
        "compiler": {
            "steps": [
                (
                    "Use a loop submit or loop evaluate event to state a capability gap or "
                    "suspected compiler defect."
                ),
                (
                    "Run evolve begin SESSION --evidence EVENT --hypothesis TEXT --output "
                    "PROPOSAL before editing package code."
                ),
                (
                    "Minimize the issue; edit IR, verifier, CUDA lowering or analysis and add"
                    " focused tests. Update spec ir/feedback when capabilities change."
                ),
                (
                    "Run evolve validate PROPOSAL. It executes frozen regression tests, "
                    "current tests and a frozen static corpus against the changed "
                    "implementation."
                ),
                (
                    "Follow its loop fork action. Re-submit IR and measure a fresh training "
                    "trial with the same Trace input, baseline and budget."
                ),
                (
                    "Run evolve compare PROPOSAL --after NEW_SESSION before either trial uses"
                    " holdouts. Review per-case effects and retained failures."
                ),
                (
                    "Iterate the revision if needed. Once chosen, finish the new session with"
                    " loop freeze and loop holdout."
                ),
            ],
            "ownership": (
                "The agent edits source. The CLI does not invoke an LLM, write compiler "
                "patches, switch git branches, commit, or promote revisions automatically."
            ),
            "limits": (
                "Validation is scoped to local trusted source. It is not a sandbox or a proof"
                " that arbitrary edited Python preserves every semantic property."
            ),
        },
    }
    return {"protocol": PROTOCOL, "package_version": __version__, "topic": topic, **lessons[topic]}


def spec(topic: str | None = None) -> dict:
    if topic is None:
        return {"protocol": PROTOCOL, "topics": {t: f"entelechy spec {t}" for t in SPEC_TOPICS}}
    documents = {
        "ir": {
            "owner": "entelechy/compiler/ir.py:Schedule",
            "supported_operators": sorted(OPERATORS),
            "fields": {
                kind.__name__: [f.name for f in fields(kind)]
                for kind in (Schedule, SharedResource, Barrier)
            },
            "mappings": ["warp_per_row", "cta_per_row"],
            "threads": {"minimum": 32, "maximum": 1024, "multiple_of": 32},
            "items_per_thread": [1, 2, 4, 8],
            "resources": (
                "CTA reductions require one float32 warp_partials element per warp; other "
                "paths declare no shared resources."
            ),
            "barriers": (
                "CTA reductions publish then release warp_partials; every CTA thread "
                "participates. Barrier.role is a synchronization role, not warp "
                "specialization."
            ),
            "schedule_examples": {
                op: [s.to_dict() for s in seed_schedules(op)[:2]] for op in sorted(OPERATORS)
            },
            "lowering": (
                "Our deterministic codegen emits CUDA with a PyTorch entry point matching "
                "the ordered Definition inputs followed by outputs. Edit IR, then regenerate; "
                "direct CUDA edits invalidate source identity."
            ),
            "not_implemented": [
                "composable operation-level IR",
                "tensor-core MMA",
                "TMA",
                "TMEM",
                "warp specialization",
                "multistage pipelines",
                "calibrated performance prediction",
            ],
        },
        "feedback": {
            "owner": "entelechy/interaction.py",
            "envelope": ["protocol", "status", "phase", "findings", "next_actions"],
            "next_actions": (
                "argv arrays are exact command arguments; editable names the source the agent"
                " may repair. reason explains the evidence behind the action."
            ),
            "routing": {
                "static_rejection": (
                    "Inspect the diagnostic path and spec ir. Repair the candidate or "
                    "snapshot a compiler investigation."
                ),
                "numerical_mismatch": (
                    "Minimize the case; inspect IR and generated CUDA. Candidate versus "
                    "compiler root cause is not yet established."
                ),
                "compile_error": (
                    "Inspect the retained worker log and generated source; minimize the "
                    "failed lowering before changing compiler rules."
                ),
                "unavailable": (
                    "Fix GPU, toolkit, baseline or timer; only unavailable training attempts "
                    "can retry unchanged source."
                ),
                "slow": (
                    "Compare measured case latencies and existing resource facts. Propose one"
                    " explicit scheduling hypothesis; no calibrated bottleneck diagnosis is "
                    "available yet."
                ),
            },
            "gates": (
                "check/analyze are static. submit generates source. evaluate performs runtime"
                " correctness before timing. Only frozen holdout evidence can qualify export."
            ),
        },
        "evolution": {
            "owner": "entelechy/evolution.py",
            "begin": (
                "Bind a hypothesis to an unchanged training event; snapshot package source, "
                "tests, native Trace inputs and build configuration BEFORE source edits."
            ),
            "validate": (
                "Reject protected evaluator/oracle changes. Run frozen and current tests and "
                "regenerate the frozen input corpus. Missing GPU evidence stays "
                "unqualified."
            ),
            "compare": (
                "Require validated current source, identical Trace input, baseline/environment, "
                "candidate budget, candidate count and GPU attempt count; compare "
                "training-only selected routes. If the before trial admitted no candidate, "
                "compare the new implementation to its measured baseline Solution and label "
                "the result new_capability_vs_baseline; no old-kernel speedup is claimed."
            ),
            "allowed_package_changes": [
                "entelechy/compiler/*.py",
                "entelechy/search.py",
                "entelechy/guide.py",
            ],
            "scope_limit": (
                "Search and compiler edits are trusted code. Equal counters cannot enforce "
                "equal agent reasoning or prove workload equivalence; tests and source review"
                " remain necessary."
            ),
            "version_policy": (
                "Package version stays 0.0.1. Content fingerprints identify each "
                "implementation independently."
            ),
            "acceptance": (
                "A matched training comparison informs revision selection; it is not "
                "automatic promotion or a held-out performance claim."
            ),
        },
    }
    from .trace_input import schema_bundle

    documents["trace"] = {
        "schema_revision": schema_bundle()["upstream_revision"],
        "objects": {
            "Definition": "Native interface, axes, constraints and authoritative reference code.",
            "Workload": "Variable axes, UUID and random/scalar/safetensors input descriptors.",
            "Solution": "Native baseline or our generated CUDA and ordered DPS entry point.",
            "Trace": "Workload-only input or a native evaluation result.",
        },
        "discovery": "trace inspect DATASET [--definition NAME]",
        "selection": (
            "loop init DATASET --definition NAME --baseline SOLUTION "
            "--output SESSION [--workloads UUID ...]"
        ),
        "policy": {
            "file": "Optional --policy JSON; no custom workload trace_input is required.",
            "fields": [
                "train",
                "holdout",
                "buckets",
                "seed",
                "acceptance",
                "timing",
                "cold_l2",
                "target_sm",
            ],
            "defaults": (
                "Sorted workload UUIDs: first 75% train, rest holdout; one "
                "bucket; seed 0; strict CUPTI."
            ),
            "partition": (
                "train/holdout contain disjoint UUID lists covering the "
                "selected workloads. Duplicate configurations are refused."
            ),
        },
        "benchmark": (
            "Optional --bench-config accepts native BenchmarkConfig YAML "
            "with per-op/per-definition overrides. At least four trials; "
            "default eight."
        ),
        "input_data": (
            "Local, materialized safetensors files are copied and hashed. "
            "Missing files and LFS pointers are refused, never replaced by "
            "random tensors."
        ),
        "comparison": (
            "Native Trace speedup_factor compares to mathematical "
            "reference. Entelechy separately scores the exact baseline "
            "Solution on matched inputs."
        ),
        "scope": (
            "One user-selected Definition per session. Query spec ir for installed "
            "operations and interface limits. Inspection does not imply compiler support."
        ),
    }
    documents["ir"]["trace_program"] = {
        "schedule": (
            "The Schedule object defined above; schedule_examples are not complete CLI inputs."
        ),
        "example_source": "SESSION/draft.ir.json, generated for the user's selected Definition.",
        "bindings": {
            "input": "Definition matrix input name",
            "output": "Definition output name",
            "weight": "Normalization vector input name, otherwise null",
            "epsilon": (
                "Normalization scalar input name or explicit constant; defaults to "
                "the candidate hypothesis 1e-6"
            ),
        },
        "semantics": (
            "Seeds are hypotheses, not translations of arbitrary Python. "
            "Definition.reference remains authoritative; numerical "
            "agreement is checked on GPU."
        ),
        "abi": (
            "Input dictionary order followed by output dictionary order; "
            "CUDA Solution uses binding=torch and "
            "destination_passing_style=true."
        ),
    }
    documents["evolution"]["trial"] = (
        "loop fork BEFORE_SESSION --output NEW_SESSION reuses frozen native input and "
        "policy after changing the compiler."
    )
    return {"protocol": PROTOCOL, "topic": topic, **documents[topic]}
