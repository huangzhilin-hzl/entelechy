# Entelechy

> Actualizing the theoretical peak.

**Entelechy** is a CuTe DSL-first NVIDIA kernel library for learning, reproducing, and exploring LLM inference kernels by hand.

In Aristotle's philosophy, *entelechy* is the realization of potential. Here, it means turning a GPU's theoretical bandwidth and compute into measured kernel performance. The distribution is `entelechy-kernels`; the Python package is `entelechy`.

## Scope

- Fundamentals: copy, reduction, softmax, RMSNorm, RoPE.
- Linear: GEMV/GEMM, grouped GEMM, FP8/FP4/W4A16.
- Attention: prefill, paged decode, split-KV, MLA, sparse attention.
- MoE: routing, permutation, grouped GEMM, fused finalize.
- Systems: sampling, collectives, compute-communication fusion.
- Research: paper implementations remain explicit experimental kernels until validated.

CuTe DSL is preferred. CUDA C++ or Triton is added only when it provides a necessary capability or a useful comparison.

## Layout

```text
entelechy/
├── entelechy/       # kernels and public Python API
├── tests/           # correctness against PyTorch
├── benchmarks/      # reproducible per-kernel benchmarks
├── .github/         # continuous integration
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── pyproject.toml
```

Do not add architecture, registry, or documentation layers before a real kernel needs them. New operators start as one module, one test, and one benchmark.

## Install

Requirements: Linux, Python 3.10+, CUDA 12.9 or CUDA 13.x, and an NVIDIA GPU.

```bash
# CUDA 12.9
pip install -e '.[dev]'

# CUDA 13.x
pip install -e '.[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

## Use

```python
import torch
from entelechy import vector_add

x = torch.randn(4096, device="cuda", dtype=torch.float16)
y = torch.randn_like(x)
out = vector_add(x, y)
```

```bash
pytest
python benchmarks/benchmark_vector_add.py
```

## Kernel standard

Every kernel contribution must include:

1. a clear public API and PyTorch correctness test;
2. exact supported SM, dtype, shape, layout, and numerical tolerance;
3. a reproducible benchmark with GPU/CUDA/CuTe DSL versions;
4. before/after performance data for optimization changes;
5. upstream attribution when code is ported rather than independently implemented.

The implementation style is informed by [QuACK](https://github.com/Dao-AILab/quack); operator coverage, testing discipline, and contribution rules are informed by [FlashInfer](https://github.com/flashinfer-ai/flashinfer). Entelechy is an independent implementation, not a source-code fork.
