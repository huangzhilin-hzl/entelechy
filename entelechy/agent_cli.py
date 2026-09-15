# SPDX-License-Identifier: Apache-2.0
"""Discoverable CLI verbs for an external coding agent."""

from __future__ import annotations

import argparse
from pathlib import Path


def register(commands: argparse._SubParsersAction) -> None:
    tutorial = commands.add_parser("tutorial", help="learn the complete agent-driven loop")
    tutorial.add_argument("topic", nargs="?", choices=("loop", "compiler"), default="loop")
    spec = commands.add_parser("spec", help="query Trace schemas and current IR capabilities")
    spec.add_argument("topic", nargs="?", choices=("trace", "ir", "feedback", "evolution"))
    trace = commands.add_parser("trace", help="inspect a native FlashInfer TraceSet")
    trace.add_argument("verb", choices=("inspect",))
    trace.add_argument("dataset", type=Path)
    trace.add_argument("--definition")
    for name in ("check", "analyze"):
        command = commands.add_parser(
            name, help="inspect IR legality" if name == "check" else "inspect static resource facts"
        )
        command.add_argument("session", type=Path)
        command.add_argument("--ir", type=Path, required=True)
    loop = commands.add_parser(
        "loop",
        help="drive one immutable compiler's candidate search step by step",
        epilog="init -> submit -> evaluate -> revise -> freeze -> holdout -> export. See tutorial.",
    )
    loop.set_defaults(group_parser=loop)
    verbs = loop.add_subparsers(dest="verb")
    init = verbs.add_parser("init", help="freeze a workload and scaffold editable IR")
    init.add_argument("dataset", type=Path)
    init.add_argument("--definition", required=True)
    init.add_argument("--baseline", required=True, help="exact baseline Solution name")
    init.add_argument("--workloads", nargs="+", help="selected workload UUIDs (defaults to all)")
    init.add_argument("--policy", type=Path, help="optional JSON search/holdout policy")
    init.add_argument("--bench-config", type=Path, help="native BenchmarkConfig YAML")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument(
        "--budget", type=int, default=12, help="maximum accepted candidates (default: %(default)s)"
    )
    fork = verbs.add_parser("fork", help="start a new compiler trial from frozen Trace inputs")
    fork.add_argument("session", type=Path)
    fork.add_argument("--output", type=Path, required=True)
    for name in ("status", "submit", "evaluate", "freeze", "holdout", "export"):
        command = verbs.add_parser(name)
        command.add_argument("session", type=Path)
        if name == "submit":
            command.add_argument("--ir", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--candidate", required=True)
        if name in {"evaluate", "holdout"}:
            command.add_argument("--device", type=int, help="inherit session device (initially 0)")
            command.add_argument("--timeout", type=float, default=300)
        if name == "export":
            command.add_argument("--output", type=Path, required=True)
    evolve = commands.add_parser(
        "evolve",
        help="snapshot, validate and compare compiler changes",
        epilog=(
            "begin BEFORE editing -> edit source/tests/spec -> validate -> fresh loop -> compare."
        ),
    )
    evolve.set_defaults(group_parser=evolve)
    verbs = evolve.add_subparsers(dest="verb")
    begin = verbs.add_parser(
        "begin", help="freeze evidence and regression gates before source edits"
    )
    begin.add_argument("session", type=Path)
    begin.add_argument("--evidence", type=Path, required=True)
    begin.add_argument("--hypothesis", required=True)
    begin.add_argument("--output", type=Path, required=True)
    validate = verbs.add_parser(
        "validate", help="execute frozen/current tests and frozen static corpus"
    )
    validate.add_argument("proposal", type=Path)
    validate.add_argument("--timeout", type=float, default=300)
    compare = verbs.add_parser(
        "compare", help="compare matched training trials after revision validation"
    )
    compare.add_argument("proposal", type=Path)
    compare.add_argument("--after", type=Path, required=True)


def dispatch(args: argparse.Namespace) -> dict | None:
    from . import evolution, guide, interaction
    from .trace_input import inspect_dataset, prepare_input

    if args.command == "tutorial":
        return guide.tutorial(args.topic)
    if args.command == "spec":
        return guide.spec(args.topic)
    if args.command == "trace":
        return inspect_dataset(args.dataset, args.definition)
    if args.command in {"check", "analyze"}:
        return interaction.check_program(
            args.session / "input.json", args.ir, analyze=args.command == "analyze"
        )
    if args.verb is None:
        args.group_parser.print_help()
        return None
    if args.command == "loop":
        if args.verb == "init":
            value = prepare_input(
                args.dataset,
                args.definition,
                args.baseline,
                policy_path=args.policy,
                config_path=args.bench_config,
                workload_ids=args.workloads,
            )
            return interaction.init_loop(args.dataset, args.output, budget=args.budget, value=value)
        if args.verb == "fork":
            return interaction.fork_loop(args.session, args.output)
        if args.verb == "status":
            return interaction.loop_status(args.session)
        if args.verb == "submit":
            return interaction.submit(args.session, args.ir)
        if args.verb == "evaluate":
            return interaction.evaluate(
                args.session, args.candidate, device=args.device, timeout_s=args.timeout
            )
        if args.verb == "freeze":
            return interaction.freeze(args.session)
        if args.verb == "holdout":
            return interaction.holdout(args.session, device=args.device, timeout_s=args.timeout)
        if args.verb == "export":
            from .export import export_portfolio

            result = export_portfolio(args.session, args.output)
            return {
                "protocol": interaction.PROTOCOL,
                "status": "exported",
                "phase": "complete",
                "export": result,
                "output": str(args.output.resolve()),
                "next_actions": [],
            }
    if args.verb == "begin":
        return evolution.begin(args.session, args.evidence, args.hypothesis, args.output)
    if args.verb == "validate":
        return evolution.validate(args.proposal, timeout_s=args.timeout)
    if args.verb == "compare":
        return evolution.compare(args.proposal, args.after)
    raise ValueError("unknown agent command")
