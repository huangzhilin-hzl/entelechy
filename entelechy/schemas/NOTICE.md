# FlashInfer Trace schema provenance

`flashinfer_trace.json` is generated from the upstream Pydantic models at
[flashinfer-bench commit 40e6ca7844b514eb4b1c7edba6d6a7377df57870](https://github.com/flashinfer-ai/flashinfer-bench/tree/40e6ca7844b514eb4b1c7edba6d6a7377df57870).

Source models: `data/definition.py`, `data/workload.py`, `data/trace.py`,
`data/solution.py`, and `bench/config.py`. They are distributed under Apache-2.0.
See the project's accompanying Apache-2.0 LICENSE. Generated schemas retain
upstream descriptions. Entelechy adds cross-object and supported-lowering checks
in Python; GPU execution also validates objects with the installed upstream models.

Bundling these data schemas allows inspection and CUDA source generation without
importing the GPU dependencies imported by the upstream package initializer.
Updating this snapshot requires compatibility tests and a source review.
