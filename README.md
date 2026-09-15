# Entelechy

Give your coding agent a kernel optimization task. Entelechy supplies the IR,
CUDA compiler, measurements, and feedback it uses to carry out that task.
Inputs use native [FlashInfer Trace](https://bench.flashinfer.ai/docs/flashinfer-trace)
Definitions, Workloads, and baseline Solutions. The agent can improve candidate
programs and extend the IR or compiler when the selected workload requires it.

**Version: 0.0.1. Status: experimental.** No measured GPU speedup is claimed by
this repository yet.

## Quick start for people

### 1. Install locally and start your agent

Install the local CLI from a source checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
entelechy
```

Start your coding agent locally in this checkout. Give it your dataset location
and access to a remote GPU environment, such as a Kubernetes Pod or Modal, with
instructions for connecting or invoking jobs. The agent prepares the remote
workspace, runs experiments through Entelechy, and retrieves the results.

Your local machine needs Python 3.10 or newer; it does not need a GPU. Remote runtime
setup is covered in the [execution environment guide](docs/DEVELOPMENT.md#execution-environment).

### 2. Give the agent your task

Give your coding agent a short task prompt:

```text
Use Entelechy to optimize <my tasks, in priority order>.
Dataset: <TraceSet location>
Remote environment: <Pod / Modal access instructions>

Start with `entelechy tutorial` and follow its CLI guidance.
```

Add any task-specific requirements, such as a baseline or budget, in plain language.
The CLI teaches the agent the workflow, defaults, and constraints. The
[agent workflow](docs/AGENT_LOOP.md) is the corresponding execution reference.

### 3. Review the results

The agent retrieves results to your local results directory, reports the remote
experiment locations, and explains which targets were met. The main files are:

| Artifact | What to review |
| --- | --- |
| `experiment.json` | Selected input identity, candidate budget and experiment state |
| `search_results.json` | Correctness, timing samples and diagnostics collected during search |
| `holdout_results.json` | Validation results for the frozen selection |
| `report.json` | Coverage, performance comparison and qualification result |
| Qualified export | Generated CUDA, native Solutions, Trace data and supporting evidence |

Completed validation writes a report. Blocked experiments retain their status,
events, and available measurements; they may not have holdout results or an export.

## What the agent does automatically

After receiving the task, the coding agent uses its remote execution and file-transfer
tools to invoke Entelechy in the supplied environment. The CLI runs commands where
invoked; the agent's tools handle remote connections and job submission.

| Stage | Agent action | Commands and tools |
| --- | --- | --- |
| Discover | Read the workflow locally, prepare the remote workspace, inspect your dataset and check the remote runtime | `tutorial`, `spec`, `trace inspect`, `doctor` |
| Initialize | Select your Definition, Workloads and baseline; freeze the policy | `loop init` |
| Develop | Edit the task's IR, inspect legality and resource facts, generate CUDA | `check`, `analyze`, `loop submit` |
| Measure and revise | Check reference agreement, measure against the baseline and use feedback to revise the candidate | `loop evaluate` |
| Extend the compiler when needed | Bind a hypothesis to evidence, edit IR/compiler source, run regressions and measure a fresh trial | `evolve begin`, `evolve validate`, `loop fork`, `evolve compare` |
| Validate and deliver | Freeze selection, evaluate reserved workloads, export qualified results and retrieve artifacts locally | `loop freeze`, `loop holdout`, `loop export`, agent file-transfer tools |

The CLI returns structured diagnostics, evidence paths, and executable
`next_actions`. The agent decides which action fits the evidence and your task.
Compiler changes and candidate search are separate loops so measurements remain
bound to a specific implementation.

```text
Local coding agent
  -> remote execution tools -> Entelechy CLI in your Pod / Modal environment
  -> IR + CUDA codegen -> reference checks + GPU measurements
  -> feedback to the agent -> revisions -> holdout validation
  -> results retrieved locally for review
```

## Capabilities and evidence

Your task determines which operators to optimize. The agent queries
`entelechy spec ir` for the installed compiler's capabilities and checks them against
the selected Definition. Current lowering is bounded to row kernels; accepting a
native Trace definition does not establish that it can already be compiled.
Missing capabilities require an implementation change and validation. The fixtures
in `examples/trace` exercise the tool and its regressions; they do not set your
optimization scope or execution order.

The native reference defines correctness. Performance comparisons use the exact
baseline Solution selected for the task. Native Trace speedup retains its
mathematical-reference meaning; Entelechy separately records paired comparisons
against the selected baseline. Strict CUPTI measurements are required for kernel
performance qualification. Static checks, CPU tests and generated CUDA alone do
not establish GPU correctness or a speedup.

See [architecture](docs/ARCHITECTURE.md) for implementation boundaries and
[development notes](docs/DEVELOPMENT.md) for regression and packaging checks.

## License and attribution

Entelechy and its generated CUDA are licensed under [Apache-2.0](LICENSE).
The FlashInfer schema snapshot retains upstream attribution in
[NOTICE.md](entelechy/schemas/NOTICE.md). Imported datasets and baseline Solutions
retain their own provenance and applicable licenses.

The design draws on [CAKE](https://arxiv.org/pdf/2608.12629),
[TileFoundry](https://github.com/tile-ai/TileFoundry), and
[FlashInfer-Bench](https://github.com/flashinfer-ai/flashinfer-bench).
This is an independent implementation and does not claim reproduction of their
published performance results.
