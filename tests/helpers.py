# SPDX-License-Identifier: Apache-2.0
"""Native fixtures and explicitly synthetic observations for control-plane tests."""

from pathlib import Path

from entelechy import interaction
from entelechy.trace_input import prepare_input

ROOT = Path(__file__).resolve().parents[1]


def prepare_example(operator="softmax"):
    name = operator + "_example"
    return prepare_input(ROOT / "examples" / "trace", name, name + "_torch_baseline")


def init_example(root, operator="softmax", *, budget=2):
    value = prepare_example(operator)
    interaction.init_loop(ROOT / "examples" / "trace", root, value=value, budget=budget)
    return value


def records(trace_input, candidate, split, *, latency=1.0, status="ok"):
    return [
        dict(
            input_id=trace_input.input_id,
            candidate_id=candidate["candidate_id"],
            source_sha256=candidate["source_sha256"],
            evaluator_id=candidate["evaluator_id"],
            case_id=case["id"],
            split=split,
            baseline_id="synthetic-baseline",
            environment={"device": "synthetic-test-only", "timing": "cupti"},
            kernel_qualified=True,
            status=status,
            correct=status == "ok",
            baseline_samples_ms=[2.0] * 8 if status == "ok" else [],
            candidate_samples_ms=[latency] * 8 if status == "ok" else [],
            diagnostics=[]
            if status == "ok"
            else [{"code": status, "path": "worker", "message": status}],
        )
        for case in trace_input.cases(split)
    ]


def admit(session):
    return interaction.submit(session, session / "draft.ir.json")["candidate"]["candidate_id"]
