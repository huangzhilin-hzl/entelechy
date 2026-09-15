# Development notes

This guide is for maintaining the tool. For optimization tasks, start with the
[human quick start](../README.md#quick-start-for-people) and let the coding agent
follow the [agent workflow](AGENT_LOOP.md).

Install the base and development dependencies with `pip install -e '.[dev]'`.
Run `pytest`, `ruff check .`, `ruff format --check .`, and `python -m build`.

## Execution environment

The local coding agent needs the CLI and development tools. GPU dependencies belong
in the execution environment supplied by the user, such as a Pod or Modal runtime.
The agent uses that environment's execution and file-transfer tools to run the CLI
remotely and retrieve results. Entelechy does not provide a remote provisioning or
job-submission backend.

Prepare the remote checkout with Python 3.10 or newer, Linux, an NVIDIA GPU, a
compatible CUDA toolkit including NVCC, and CUDA-enabled PyTorch. Install the
project's development and native evaluation dependencies there:

```bash
python -m pip install -e '.[dev,trace]'
entelechy doctor
```

Dependency versions are declared in [pyproject.toml](../pyproject.toml).
The doctor command checks the environment in which it runs; use `--device` when
selecting a non-default GPU within that environment. Strict CUPTI must be available
for performance qualification. GPU integration tests skip when prerequisites are
unavailable.

Make the selected native TraceSet and referenced blobs available remotely. Keep the
source revision, working directory, and session storage consistent across calls;
use persistent storage when the execution environment is recreated between jobs.
The agent retrieves experiment directories, logs, and exports to the local results
directory. Local GPU development can use the same setup on the local machine.

## Native Trace changes

The `examples/trace` directory contains independently authored regression fixtures
in native Definition, Workload and Solution formats. CLI and integration tests use
these fixtures. User experiments use the user's own TraceSet and selected tasks
through the same input format and CLI. Static compiler regression gates cover the
frozen fixtures and the frozen session input; they do not choose the user's targets.

Preserve native input/output dictionary order and keep the original reference and
input descriptors. Test missing blobs, scalar values, argument reordering, native
Solution packaging, and candidate/runtime identity drift. Never substitute random
inputs for unavailable real data.

Use `input_id` for the frozen input identity and `draft.ir.json` for the editable
program. CLI `--ir` inputs contain both `schedule` and `bindings`; the low-level
`Schedule` remains useful for compiler unit tests.

The bundled schema JSON is generated from the upstream Pydantic models at the
revision recorded in `entelechy/schemas/NOTICE.md`. Schema changes are evaluator
changes and require independent review; they cannot be promoted as compiler-only
optimizations through `evolve`. See `scripts/check_flashinfer_schema.py` for a
CPU compatibility check against a local upstream source checkout.

## Compiler changes

Use `evolve begin` on a submission or training event before editing. Add a focused
regression, change the IR/verifier/lowering/spec together where needed, then run
`evolve validate`. Follow the returned `loop fork` action, measure the new trial,
and compare before holdout. Keep package version fixed at 0.0.1.

Mocked measurements validate control flow only. A package build, syntax check,
static corpus or passing CPU test must not be described as GPU correctness or
measured performance improvement.
