# Changelog

Changes are recorded here before a release is tagged. This file does not imply
that a package has been published to PyPI.

## Unreleased

### Added

- Discoverable agent CLI tutorials and specifications, stepwise candidate search,
  evidence-linked repair actions, and compiler revision snapshot/validation/comparison.
- Typed schedules, legality checks, and CUDA generation for supported row kernels.
- Native FlashInfer TraceSet selection, immutable input snapshots, ordered CUDA
  Solutions, native reference evaluation, and explicit baseline Solutions.
- Isolated workers and strict CUPTI measurement.
- Paired performance statistics, training-only routing selection, independent
  holdout evaluation, and source exports with reproducibility records.
- Frozen input forks and native Trace regression coverage for compiler evolution.
- Contributor documentation, security reporting guidance, community standards,
  and CPU compatibility checks in CI.

### Changed

- Reduced the task prompt to task inputs and a CLI entry point; agent setup,
  defaults, workflow constraints, and delivery guidance live in the CLI tutorial.
- Documented local agents driving remote GPU experiments, with runtime setup in
  the execution environment and results retrieved locally.
- Separated the human quick start and task prompt from the agent execution workflow;
  documentation uses user-selected workloads and labels bundled data as regression fixtures.
- Reoriented the package around compiler-guided kernel optimization.
- Made GPU dependencies optional so the control plane can run without CUDA.
- Unified input identity as `input_id` and candidate files as `draft.ir.json`.
  CLI programs contain both `schedule` and `bindings` and are submitted with `--ir`.
- Named evaluation artifacts `search_results.json` and `holdout_results.json`;
  `experiment.json` continues to store session state.
- Pinned the reviewed FlashInfer-Bench revision in the `trace` optional dependencies.

### Removed

- The initial standalone vector-add example and its benchmark.
- Custom workload contracts, batch search and external proposer interfaces,
  compatibility aliases, and the separate legacy runtime.

### Validation status

- GPU correctness and performance claims require evidence from a target device.
  CPU tests and generated source alone do not establish a speedup.
