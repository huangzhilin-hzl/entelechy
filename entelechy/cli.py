# SPDX-License-Identifier: Apache-2.0
"""Public CLI for native Trace input and agent-driven compiler evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .agent_cli import dispatch, register
from .artifacts import read_json


def _print(value: object) -> None:
    # Definition input/output mapping order defines the ABI.
    print(json.dumps(value, indent=2, allow_nan=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Agent-driven IR and CUDA compiler evolution from FlashInfer Trace.",
        epilog="Start: tutorial -> trace inspect -> loop init -> check/analyze -> submit -> "
        "evaluate -> revise -> freeze -> holdout -> export. Compiler edits: tutorial compiler.",
    )
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command")
    register(commands)
    doctor = commands.add_parser("doctor", help="inspect GPU and optional runtime dependencies")
    doctor.add_argument("--device", type=int, default=0)
    report = commands.add_parser("report", help="read a completed experiment report")
    report.add_argument("experiment", type=Path)
    return root


def main(argv: list[str] | None = None) -> int:
    root = parser()
    args = root.parse_args(argv)
    if args.command is None:
        root.print_help()
        return 0
    try:
        if args.command == "doctor":
            from .runtime.worker import doctor

            result = doctor(args.device)
        elif args.command == "report":
            result = read_json(args.experiment / "report.json")
        else:
            result = dispatch(args)
        if result is None:
            return 0
        _print(result)
        if result.get("status") == "unavailable":
            return 2
        return 1 if result.get("status") in {"rejected", "failed", "error"} else 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        _print(
            {
                "protocol": "entelechy.agent.v1",
                "status": "error",
                "error": {"code": type(error).__name__, "message": str(error)},
                "next_actions": [
                    {
                        "argv": [
                            "entelechy",
                            "tutorial",
                            "compiler" if args.command == "evolve" else "loop",
                        ],
                        "reason": "Inspect the workflow and repair the refused operation.",
                    }
                ],
            }
        )
        return 1
