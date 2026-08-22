# Contributing

## Development

```bash
pip install -e '.[dev]'
pre-commit install
pytest
```

For CUDA 13.x:

```bash
pip install -e '.[dev,cu13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

## Adding a kernel

Keep the change local and reviewable:

1. add `entelechy/<op>.py` with the public function and CuTe DSL kernel;
2. add `tests/test_<op>.py` against a PyTorch reference;
3. add `benchmarks/benchmark_<op>.py`;
4. export the public function from `entelechy/__init__.py`;
5. update the README scope only when support is real and tested.

Separate algorithm, layout, and scheduling changes when practical. Architecture-specific variants may share an operator module until their size justifies a subpackage.

## Pull requests

- Explain the kernel, supported SM/dtypes/shapes, and known limitations.
- Include correctness coverage for boundary and non-tile-aligned shapes.
- Performance changes must report before/after numbers, GPU, CUDA, CuTe DSL version, shapes, warmup, and iteration count.
- Keep imported code under its original license and record the exact source revision.
- Run `ruff check .`, `ruff format --check .`, and relevant tests before submission.

Be respectful and technical in project discussions. Report security issues through GitHub's private security advisory instead of a public issue.
