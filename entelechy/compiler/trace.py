# SPDX-License-Identifier: Apache-2.0
"""Row IR bindings and CUDA entry points for native FlashInfer Definitions.

Bindings describe a candidate algorithm. Only execution against Definition.reference
can establish agreement; op_type and interface checks are not a semantic proof.
"""

from __future__ import annotations

import ast
import math

from ..artifacts import canonical_hash
from .cuda import compile_cuda
from .ir import Schedule
from .verify import verify


def definition_id(definition: dict) -> str:
    return canonical_hash(
        {
            "definition": definition,
            "inputs": list(definition["inputs"]),
            "outputs": list(definition["outputs"]),
        }
    )


def seed_program(definition: dict) -> dict:
    inputs, outputs = definition["inputs"], definition["outputs"]
    matrices = [k for k, v in inputs.items() if v["shape"] is not None and len(v["shape"]) == 2]
    vectors = [k for k, v in inputs.items() if v["shape"] is not None and len(v["shape"]) == 1]
    scalars = [k for k, v in inputs.items() if v["shape"] is None]
    if len(matrices) != 1 or len(outputs) != 1:
        raise ValueError(
            "unsupported interface: initial compiler needs one matrix input and one output"
        )
    norm = definition["op_type"] == "rmsnorm"
    program = {
        "schedule": Schedule.for_mapping("warp_per_row", operator=definition["op_type"]).to_dict(),
        "bindings": {
            "input": matrices[0],
            "output": next(iter(outputs)),
            "weight": vectors[0] if norm and len(vectors) == 1 else None,
            "epsilon": scalars[0] if norm and len(scalars) == 1 else 1e-6,
        },
    }
    validate_program(program, definition)
    return program


def draft_program(definition: dict) -> dict:
    """Keep unsupported definitions reachable through submit feedback and evolve begin."""
    try:
        return seed_program(definition)
    except ValueError:
        return {
            "schedule": Schedule().to_dict(),
            "bindings": {
                "input": next(iter(definition["inputs"]), ""),
                "output": next(iter(definition["outputs"]), ""),
                "weight": None,
                "epsilon": 1e-6,
            },
        }


def validate_program(program: dict, definition: dict) -> Schedule:
    if not isinstance(program, dict) or set(program) != {"schedule", "bindings"}:
        raise ValueError("trace IR requires exactly schedule and bindings")
    binding = program["bindings"]
    if not isinstance(binding, dict) or set(binding) != {"input", "output", "weight", "epsilon"}:
        raise ValueError("bindings require input, output, weight and epsilon")
    inputs, outputs = definition["inputs"], definition["outputs"]
    if len(outputs) != 1 or binding["output"] not in outputs or binding["input"] not in inputs:
        raise ValueError("bindings must name Definition inputs and its single output")
    operator = definition["op_type"]
    schedule = Schedule.from_dict(program["schedule"])
    dtype = outputs[binding["output"]]["dtype"]
    report = verify(schedule, operator=operator, dtype=dtype)
    if not report.valid:
        raise ValueError(str([vars(d) for d in report.diagnostics]))
    x, out = inputs[binding["input"]], outputs[binding["output"]]
    if (
        x["dtype"] != dtype
        or x["shape"] is None
        or out["shape"] is None
        or len(x["shape"]) != 2
        or len(out["shape"]) != 2
    ):
        raise ValueError("row compiler requires two-dimensional input/output with matching dtype")
    consumed = {binding["input"]}
    if operator == "rmsnorm":
        weight = binding["weight"]
        if (
            weight not in inputs
            or inputs[weight]["dtype"] != dtype
            or len(inputs[weight]["shape"] or []) != 1
        ):
            raise ValueError("rmsnorm requires a one-dimensional weight with matching dtype")
        consumed.add(weight)
        eps = binding["epsilon"]
        if isinstance(eps, str):
            if (
                eps not in inputs
                or inputs[eps]["shape"] is not None
                or inputs[eps]["dtype"] not in {"float16", "bfloat16", "float32"}
            ):
                raise ValueError("epsilon binding must name a floating scalar input")
            consumed.add(eps)
        elif (
            isinstance(eps, bool)
            or not isinstance(eps, (float, int))
            or not math.isfinite(eps)
            or not 0 <= eps <= 1
        ):
            raise ValueError("epsilon constant must be finite and in [0, 1]")
    elif binding["weight"] is not None or binding["epsilon"] != 1e-6:
        raise ValueError("non-normalization bindings use weight=null and epsilon=1e-6")
    if consumed != set(inputs):
        raise ValueError("every Definition input must have a supported binding")
    return schedule


def dimensions(definition: dict, workload: dict, program: dict) -> tuple[int, int]:
    validate_program(program, definition)
    axes = {k: v["value"] for k, v in definition["axes"].items() if v["type"] == "const"}
    axes.update(workload["axes"])
    if any(type(v) is not int or not 1 <= v <= 2**31 - 1 for v in axes.values()):
        raise ValueError("initial row compiler requires positive 32-bit dimensions")
    allowed = (
        ast.Expression,
        ast.Compare,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.FloorDiv,
        ast.Mod,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.And,
        ast.Or,
        ast.Not,
        ast.USub,
        ast.UAdd,
    )
    for expression in definition.get("constraints", []):
        tree = ast.parse(expression, mode="eval")
        if any(not isinstance(n, allowed) for n in ast.walk(tree)):
            raise ValueError("unsupported axis constraint syntax")
        if not eval(compile(tree, "<axis constraint>", "eval"), {"__builtins__": {}}, axes):
            raise ValueError(f"workload violates Definition constraint: {expression}")
    binding = program["bindings"]
    shape = lambda spec: [axes[k] for k in spec["shape"]]
    rows, cols = shape(definition["outputs"][binding["output"]])
    multiplier = 2 if definition["op_type"] == "silu_mul" else 1
    if shape(definition["inputs"][binding["input"]]) != [rows, cols * multiplier]:
        raise ValueError("input and output shapes do not match the bound row algorithm")
    if binding["weight"] and shape(definition["inputs"][binding["weight"]]) != [cols]:
        raise ValueError("weight must match the output width")
    if rows * cols * multiplier >= 2**31:
        raise ValueError("input exceeds the current row compiler indexing range")
    for key, spec in definition["inputs"].items():
        descriptor = workload["inputs"][key]
        if (spec["shape"] is None) != (descriptor["type"] == "scalar"):
            raise ValueError("scalar/tensor input descriptor disagrees with Definition")
    return rows, cols


def compile_trace(program: dict, definition: dict) -> str:
    schedule = validate_program(program, definition)
    binding = program["bindings"]
    dtype = definition["outputs"][binding["output"]]["dtype"]
    source = compile_cuda(schedule, definition["op_type"], dtype)
    # Keep the existing kernel/launcher; supply the native ordered DPS interface.
    source = source[: source.index("PYBIND11_MODULE(")]
    arguments, variables, guards = [], {}, []
    first_axis = {}
    for index, (name, spec) in enumerate(
        [*definition["inputs"].items(), *definition["outputs"].items()]
    ):
        var = f"argument_{index}"
        variables[name] = var
        arguments.append(f"{'double' if spec['shape'] is None else 'at::Tensor'} {var}")
        if spec["shape"] is None:
            continue
        guards.append(
            f'TORCH_CHECK({var}.dim() == {len(spec["shape"])}, "Definition rank mismatch");'
        )
        for dim, axis in enumerate(spec["shape"]):
            extent = f"{var}.size({dim})"
            value = definition["axes"][axis]
            expected = str(value["value"]) if value["type"] == "const" else first_axis.get(axis)
            if expected:
                guards.append(f'TORCH_CHECK({extent} == {expected}, "Definition axis mismatch");')
            first_axis.setdefault(axis, extent)
    x, out = variables[binding["input"]], variables[binding["output"]]
    weight = (
        variables[binding["weight"]] if binding["weight"] else f"at::empty({{0}}, {x}.options())"
    )
    eps = binding["epsilon"]
    epsilon = variables[eps] if isinstance(eps, str) else repr(float(eps))
    source += f"\nvoid trace_run({', '.join(arguments)}) {{\n" + "\n".join(guards)
    source += f"\nrun({x}, {weight}, {out}, {epsilon});\n}}\n"
    source += (
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("run", &trace_run); }\n#endif\n'
    )
    return (
        f"// definition_id={definition_id(definition)}; program_id={canonical_hash(program)}\n"
        + source
    )


def solution_for(source: str, definition: dict, *, target_sm: int | None = None) -> dict:
    from ..artifacts import source_hash

    return {
        "name": "entelechy_" + source_hash(source)[:24],
        "definition": definition["name"],
        "author": "entelechy",
        "spec": {
            "language": "cuda",
            "binding": "torch",
            "target_hardware": [f"sm_{target_sm}" if target_sm else "cuda"],
            "entry_point": "kernel.cu::run",
            "destination_passing_style": True,
        },
        "sources": [{"path": "kernel.cu", "content": source}],
    }
