# SPDX-License-Identifier: Apache-2.0
"""CPU compatibility check against an explicit local flashinfer-bench source tree.

Only data-model modules are loaded in this standalone process. The upstream
package initializer and runtime dtype resolver are deliberately not executed.
Production GPU workers use ordinary upstream imports instead.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    for name, relative in (
        ("flashinfer_bench", "flashinfer_bench"),
        ("flashinfer_bench.data", "flashinfer_bench/data"),
        ("flashinfer_bench.bench", "flashinfer_bench/bench"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str((args.upstream / relative).resolve())]
        sys.modules[name] = module
    utilities = types.ModuleType("flashinfer_bench.utils")

    def no_runtime_dtype(*args):
        raise RuntimeError("runtime dtype resolution is not part of data-schema checking")

    utilities.dtype_str_to_torch_dtype = no_runtime_dtype
    sys.modules[utilities.__name__] = utilities
    Definition = importlib.import_module("flashinfer_bench.data.definition").Definition
    Solution = importlib.import_module("flashinfer_bench.data.solution").Solution
    Trace = importlib.import_module("flashinfer_bench.data.trace").Trace
    BenchmarkConfig = importlib.import_module("flashinfer_bench.bench.config").BenchmarkConfig

    from entelechy.compiler.trace import compile_trace, seed_program, solution_for
    from entelechy.trace_input import load_dataset, prepare_input, resolve_config

    checked = []
    dataset = repo / "examples" / "trace"
    for operator in ("rmsnorm", "softmax", "silu_mul"):
        name = operator + "_example"
        definition, workloads, solutions = load_dataset(dataset, name)
        native = Definition.model_validate(definition)
        assert list(native.inputs) == list(definition["inputs"])
        for workload in workloads:
            Trace.model_validate(workload)
        for solution in solutions:
            Solution.model_validate(solution)
        program = seed_program(definition)
        generated = solution_for(compile_trace(program, definition), definition)
        Solution.model_validate(generated)
        value = prepare_input(dataset, name, name + "_torch_baseline")
        config = value.to_dict()["benchmark_config"]
        assert BenchmarkConfig.model_validate(config).resolve_eval_config(
            native
        ).model_dump() == resolve_config(config, definition)
        checked.append(name)
    print(json.dumps({"native_models_validated": checked, "gpu_executed": False}, indent=2))


if __name__ == "__main__":
    main()
