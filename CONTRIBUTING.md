# Contributing

Contributions to kernels, compiler analysis, runtime tooling, tests, and
documentation are welcome. Follow the [code of conduct](CODE_OF_CONDUCT.md).
Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).

Write repository documentation, comments, docstrings, CLI messages, and
contribution materials in English. See the [development guide](docs/DEVELOPMENT.md)
for setup, tooling versions, and release checks.

The project version is fixed at `0.0.1`. Do not bump it for routine changes;
change it only when a maintainer explicitly requests a new version.

Install the CPU control plane and development tools with `pip install -e '.[dev]'`.
Run `ruff check .`, `ruff format --check .`, `pytest`, and `python -m build`.
GPU tests are marked `cuda`; a skipped GPU test is not evidence of correctness or speed.

## Proposing a change

For a new operator or a substantial IR change, open an issue describing the
workload, target device, baseline, and validation plan. Keep changes focused and
use the pull request template to explain behavior and validation. Documentation
fixes can be submitted directly.

Include a regression test when fixing a behavioral defect. Document checks you
could not run. Do not include credentials, private workloads, profiler dumps, or
generated experiment directories in a pull request.

New project source files should carry the Apache-2.0 SPDX identifier. Contributions
are provided under the project's [Apache-2.0 license](LICENSE); preserve applicable
third-party notices when reusing code.

## Compiler changes

Add vocabulary, resource/effect semantics, verifier rules, lowering, and a minimal
regression case together. Diagnose unsupported operations rather than guessing a
lowering. Follow `entelechy tutorial compiler`: snapshot evidence before editing,
validate the revision, then measure a fresh trial from `loop fork`.

Keep each experiment on one compiler fingerprint. Compare old/new revisions with
the same frozen native Trace inputs, device, baseline identity, timing method, and budget.
Check formerly valid programs as well as malformed schedules.

## Performance evidence

- Freeze workload, oracle, tolerances and acceptance criteria before search.
- Record failures and regressions; never average them away.
- Choose routing on training data, freeze it, then evaluate all required holdouts.
- Report absolute latency, exact environment/baseline, raw samples and timing scope.
- CUDA event diagnostics cannot qualify a kernel performance result.
- Per-kernel measurements are not a dispatcher-inclusive benchmark.
- Run sanitizers, graph replay, adapter and end-to-end tests before deployment.
- Attribute imported code and preserve its license and exact revision.

The CPU control plane uses JSON Schema and YAML parsing. Importing `entelechy`
must not import PyTorch, initialize CUDA, or compile a kernel. Use `.[trace]` for
the native evaluation runtime and `.[gpu]` for low-level CUDA compiler tests.
