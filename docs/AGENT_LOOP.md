# Agent CLI workflow

This is the execution reference for the coding agent after a person provides the
[task prompt](../README.md#2-give-the-agent-your-task). The person supplies the
tasks, dataset, remote access, and any task-specific requirements. The agent
executes the commands below and makes the IR/compiler edits. Entelechy supplies
native Trace input, inspection, CUDA generation, evaluation, and revision gates.
The agent runs locally, outside the CLI, and uses the supplied remote execution
tools to invoke experiment commands in the GPU environment.

## Local agent and remote experiments

Read `entelechy tutorial` and `entelechy spec` locally. Use the user's Pod access,
Modal invocation instructions, or other remote tooling to prepare a workspace and
invoke Entelechy there. The CLI executes on its current host; remote job submission
and file transfer belong to the agent's tools. See the
[execution environment guide](DEVELOPMENT.md#execution-environment) for runtime setup.

Keep one authoritative experiment directory in the remote workspace. Run `loop`
and `evolve` commands there, and interpret returned `next_actions[].argv` and file
paths in that environment. Run `doctor` there to check the target GPU runtime;
a local machine without a GPU does not block a remote experiment.

Keep the remote source revision synchronized with the code being tested. If editing
compiler source locally, run `evolve begin` remotely before synchronizing the edits,
then validate and measure the changed revision remotely. Preserve the remote
workspace and session paths across job invocations. Retrieve CLI JSON, logs, and
artifacts to inspect feedback; after validation, copy the completed experiment and
any qualified export to the user's local results directory. Keep retrieved copies
for review while continuing state-changing commands in the remote workspace.

## Task inputs and command notation

Resolve the user's requested tasks against their dataset. Preserve the requested
order and workload scope; create one session per selected Definition. Use the
bundled fixtures only when the user asks for a demonstration or regression check.
Query `spec ir` for current capabilities rather than choosing a task because it
appears in an example.

The task prompt only needs tasks, dataset, remote access, and the `entelechy tutorial`
entry point. Query the CLI for workflow and constraints. Honor any explicit baseline,
budget, configuration, or output paths. If no baseline is specified, use the sole
matching Solution when unambiguous; resolve missing or ambiguous baselines with the
user. Omit `--budget`, `--bench-config`, and `--policy` to use CLI defaults when no
overrides are requested. Choose unused remote experiment and local results
directories when unspecified, and report their locations. These are agent setup
decisions; `loop init` still requires an explicit baseline and output path.

Commands containing the following uppercase names are templates. Substitute actual
values from the task or CLI output before executing them:

| Placeholder | Meaning |
| --- | --- |
| `DATASET` | The user's native TraceSet directory in the remote workspace |
| `DEFINITION` | The exact Definition name resolved from the requested task |
| `BASELINE` | The exact baseline Solution name selected under the user's requirements |
| `SESSION` | The persistent remote experiment directory passed to `loop init --output` |
| `IR` | The editable `SESSION/draft.ir.json` file, or another candidate program |
| `ID` | A candidate ID returned by `loop submit` |
| `DEVICE` | The GPU device index visible inside the remote execution environment |
| `EVENT`, `TEXT`, `PROPOSAL` | Recorded evidence path, optimization hypothesis, and proposal directory |
| `BEFORE_SESSION`, `NEW_SESSION` | Experiment directories before and after a compiler revision |
| `EXPORT` | The remote output directory for a qualified export, retrieved locally afterward |

The agent can inspect the dataset to resolve names from a workload description.
Missing required task details should be resolved before selecting unrelated tasks
or changing the baseline.

## Discover the workflow

```bash
entelechy
entelechy tutorial
entelechy tutorial compiler
entelechy spec trace
entelechy spec ir
entelechy spec feedback
entelechy loop init --help
```

Run these checks in the remote workspace:

```text
entelechy trace inspect DATASET
entelechy trace inspect DATASET --definition DEFINITION
entelechy doctor --device DEVICE
```

Root, `loop`, and `evolve` without a subcommand print help and exit 0. Operational
responses use JSON. Exit 1 means an error or rejection. Exit 2 with JSON status
`unavailable` means missing execution prerequisites; argument errors also use
exit 2 but print usage to stderr. Exit 0 alone is not a performance qualification.

## Native input and policy

The input directory uses FlashInfer's `definitions/`, `workloads/`, `solutions/`,
and optional input blob paths. Definitions describe ordered inputs and outputs,
axes, constraints, and authoritative Python reference code. Workload-only Trace
records bind variable axes and carry random, scalar, or safetensors inputs.

```text
entelechy loop init DATASET --definition DEFINITION --baseline BASELINE --output SESSION
```

Pass `--budget`, `--bench-config`, or `--policy` when the task supplies overrides.
The default candidate budget is 12; benchmark and policy defaults are described below.

`--baseline` names an exact native Solution. The compiler never replaces it with
another implementation or treats the mathematical reference as the target kernel.
`--workloads UUID UUID` optionally selects a subset. At least two distinct workload
configurations are needed. One Definition is selected per session.

Session state and evaluation results have separate files:

| File | Purpose |
| --- | --- |
| `experiment.json` | Current phase, candidate budget, submitted candidates, implementation identities and event index |
| `search_results.json` | Correctness, paired timing samples and diagnostics used to select candidates during search; initially empty |
| `holdout_results.json` | Validation results written after evaluating the frozen routing |

The two result files are also included in a qualified export. The workload split
names remain `train` and `holdout`; `train` means candidate search, not model training.

`--bench-config` reads native BenchmarkConfig YAML. Supported controls are
`warmup_runs`, `iterations`, `num_trials`, `rtol`, `atol`,
`required_matched_ratio`, `profile_baseline`, `op_type_config`, and
`definition_config`. Definition overrides replace op-type overrides; top-level
non-null values win. Custom evaluator extras are currently refused. Reference
profiling must remain enabled. At least four trials are required; without a config
file, Entelechy explicitly requests eight trials and retains upstream parameter
defaults for the other fields. Runner selection and timeouts belong to Entelechy's
isolated CLI worker, so unrelated BenchmarkConfig controls are refused.

`--policy` optionally reads JSON containing:

- `train` and `holdout`: disjoint workload UUID lists covering the selection.
- `buckets`: a UUID-to-bucket mapping; every bucket must have both splits.
- `seed`: an unsigned 32-bit seed for reproducible native random input generation.
- `acceptance`: `min_speedup`, `max_regression`, and `confidence`.
- `timing`: `cupti` or diagnostic `cuda_event`; `cold_l2`: a boolean.
- `target_sm`: an optional required SM architecture.

Without a policy, sorted workload UUIDs are partitioned approximately 75/25, with
at least one workload on each side, one bucket, seed 0, strict CUPTI, cold L2,
minimum speedup 1.03, regression limit 0.02, and confidence 0.95. The default split
is reproducible, not a guarantee of representative coverage. Inspect it and use
an explicit policy for production studies. Duplicate configurations cannot stand
in for independent holdouts.

Local safetensors paths must be relative to the dataset root. Selected blobs are
copied into the session and hashed. Missing files, external paths, and Git LFS
pointers are refused; the tool does not substitute random tensors or silently
fetch external data. CPU inspection does not execute reference or Solution code.
GPU evaluation executes the selected dataset's trusted code in the worker.

## Candidate IR and CUDA Solution

```text
entelechy check SESSION --ir IR
entelechy analyze SESSION --ir IR
entelechy loop submit SESSION --ir IR
```

The complete CLI IR contains a `schedule` and `bindings`. A low-level `Schedule`
alone is not a complete CLI input. The current schedule controls
warp/CTA row assignment, thread count, unrolling, shared memory, and barriers.
Bindings name the Definition's matrix input, output, any required vector input, and
an epsilon scalar input or explicit constant. When epsilon is not an input, the
seed uses 1e-6 as a candidate hypothesis; the agent must read the reference and
adjust the IR when necessary. op_type and interface compatibility are not a proof
of mathematical equivalence.

The installed `spec ir` defines supported operations, dtypes, mappings, and bindings.
Current lowering supports bounded row interfaces. A broader Definition can be
inspected and snapshotted; an unsupported draft is rejected on submission with an
evidence-bound compiler investigation action. Extend the compiler through `evolve`
when the user's selected task needs a missing capability, keeping the problem
definition fixed.

`check` and `analyze` are static. There is no calibrated latency predictor.
`submit` freezes the accepted program and generates `kernel.cu`, `candidate.json`,
and a native `solution.json`. CUDA uses a PyTorch binding and destination-passing
style: Definition inputs in their original order, then outputs in their original
order. Native JSON order is preserved, and the explicit argument order participates
in the input fingerprint. Direct edits to saved CUDA or Solution files invalidate
the candidate; edit IR and resubmit instead.

## Feedback and measurement

The submit result supplies `candidate.candidate_id` and executable
`next_actions[].argv`. Follow the evaluate action or run:

```text
entelechy loop evaluate SESSION --candidate ID --device DEVICE
```

The isolated worker validates native objects with the installed flashinfer-bench
models, builds the reference and Solutions through its builder registry, and uses
its default evaluator for reference agreement. Both baseline and candidate are
checked on native Workload inputs. Extra checks detect input mutation, incomplete
output writes, and output guard corruption. Only then are timings collected.

Entelechy collects paired ABBA rounds on the same native input configuration and
retains the sample arrays. Strict CUPTI refuses event-timer fallback. Diagnostic
CUDA-event timings cannot qualify a kernel for export. This measurement protocol
is intentionally explicit; it does not claim identical statistics to an arbitrary
`flashinfer-bench run` invocation.

Each result has two distinct measurements:

- Native `Trace.evaluation.performance.speedup_factor` compares a Solution to the
  mathematical reference, following FlashInfer's schema.
- Entelechy's paired summary compares the candidate to the selected baseline
  Solution. This is the optimization objective.

Native evaluation Traces are saved alongside paired samples in each worker's
`result.json`. No GPU means `unavailable`, never a synthesized speedup. Definition
or Workload changes are part of the input identity. Software, hardware, baseline,
and timer identities accompany measurements.

| Feedback | Next action |
| --- | --- |
| Static IR rejection | Read `spec ir`, repair bindings/schedule, or begin compiler investigation |
| CUDA build failure | Inspect generated source and worker logs; minimize the failed lowering |
| Numerical mismatch | Inspect the native reference and failing workload; localize candidate versus compiler fault |
| Correct but slow | Compare per-workload timings and resource facts; propose a specific change |
| Missing GPU, baseline dependency, or timer | Repair the environment; retry the unavailable attempt |

Results include diagnostic paths, evidence paths, and concrete next actions.
`root_cause_proven` remains false: a failure does not prove a compiler defect.
The agent chooses the hypothesis and edits; CLI commands do not invoke a model.

## Selection and state

```text
entelechy loop status SESSION
entelechy loop freeze SESSION
entelechy loop holdout SESSION
entelechy loop export SESSION --output EXPORT
```

The state progression is `search -> frozen -> holdout_running -> complete`.
Interrupted holdouts enter `interrupted`. Search edits stop at freeze. Every
admitted candidate needs training records before freeze; baseline fallback remains
available. Holdout is single-use and cannot repair routes. Export rechecks raw
qualification and includes selected CUDA, native Solutions, the native dataset,
result Traces, routing, and evidence. It does not install a production dispatcher.

Mutating commands acquire an exclusive session lock. A host-local GPU lease
serializes Entelechy workers. Only unavailable training attempts can retry;
attempts retain separate logs. Device choice is inherited and becomes fixed after
successful training measurements. These are trusted-local integrity gates, not a
sandbox against an agent that rewrites all artifacts or reads holdout files.

## Compiler evolution

Rejected submissions retain the submitted IR bytes and their hash. `evolve begin`
copies that IR into the proposal so later edits cannot change the failure evidence.

Use the actual `evidence_path` returned by a submit or training evaluation:

```text
entelechy evolve begin SESSION --evidence EVENT --hypothesis TEXT --output PROPOSAL
```

Begin before editing implementation. It freezes source, schemas, tests, examples,
the selected native input, baseline, policy, and blobs. Then the agent edits IR,
verification, CUDA lowering, analysis, focused tests, and the queryable spec.
The workload reference, evaluators, acceptance code and data schemas remain
protected within this compiler-change workflow.

```text
entelechy evolve validate PROPOSAL
entelechy loop fork BEFORE_SESSION --output NEW_SESSION
```

Validation runs frozen and current tests against the changed implementation,
generates candidates for the frozen regression fixtures, and generates
native candidates for the frozen session input. Identical inputs are deduplicated.
Regression fixtures test existing compiler behavior; they do not add optimization
tasks to the user's plan. `static_validated` does not establish GPU correctness or speed.
The returned `loop fork` action reuses the exact native input and policy with the
new compiler identity. Submit and evaluate candidates in that fresh session.

```text
entelechy evolve compare PROPOSAL --after NEW_SESSION
```

Comparison checks identical inputs, baseline, environment, candidate budget,
submitted count, attempt count and the exact validated revision. It uses training
data only; the descriptive cross-revision aggregate is not a paired confidence
bound or a held-out claim. Choose a revision before its final holdout. Package
version remains 0.0.1; content fingerprints distinguish revisions.

If the before trial admitted no candidate and the proposal was bound to a rejected
submission, comparison uses `comparison_kind=new_capability_vs_baseline`. It
compares new measured routes to their baseline Solution in the new trial. There is
no old generated-kernel timing, and candidate/attempt counts cannot be matched to
zero. The input and candidate budget remain fixed. This result does not prove the
old compiler could never express the rejected program.

## Design sources and compatibility

The CLI discovery pattern follows
[TileFoundry](https://tile-ai.github.io/TileFoundry.github.io/spec/cli/), and
IR/compiler evolution follows the direction of [CAKE](https://arxiv.org/pdf/2608.12629).
Input and Solution formats follow the
[FlashInfer Trace schema](https://bench.flashinfer.ai/docs/flashinfer-trace).
The packaged CPU schemas are generated from flashinfer-bench commit
`40e6ca7844b514eb4b1c7edba6d6a7377df57870`; their provenance is recorded in
[the schema notice](../entelechy/schemas/NOTICE.md). Installed upstream models and
resolved benchmark settings are checked again on the GPU path.

`input_id` identifies the immutable native input snapshot throughout session,
candidate, evaluation, comparison, and export artifacts. The CLI and runtime use
one native Trace input path.
