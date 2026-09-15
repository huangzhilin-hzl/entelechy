# SPDX-License-Identifier: Apache-2.0
"""Strict GPU timing with paired ABBA trials and explicit cache policy."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import importlib
import importlib.metadata
import inspect
import math
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any


class TimingUnavailable(RuntimeError):
    """The requested GPU timing protocol cannot be executed faithfully."""


def _l2_cache_bytes(torch: Any, device: int) -> int:
    props = torch.cuda.get_device_properties(device)
    for name in ("L2_cache_size", "l2_cache_size"):
        value = getattr(props, name, 0)
        if value:
            return int(value)
    candidates = [ctypes.util.find_library("cudart"), "libcudart.so"]
    torch_root = Path(torch.__file__).resolve().parent
    candidates.extend(str(path) for path in torch_root.glob("lib/libcudart.so*"))
    for directory in sys.path:
        candidates.extend(
            str(path) for path in Path(directory).glob("nvidia/cuda_runtime/lib/libcudart.so*")
        )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            library = ctypes.CDLL(candidate)
            query = library.cudaDeviceGetAttribute
            query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            query.restype = ctypes.c_int
            value = ctypes.c_int()
            # cudaDevAttrL2CacheSize is 38 in CUDA's public runtime ABI.
            if query(ctypes.byref(value), 38, device) == 0 and value.value > 0:
                return value.value
        except (OSError, AttributeError):
            continue
    raise TimingUnavailable("cold_l2 requested but the actual L2 cache capacity cannot be queried")


def _timed(torch: Any, run: Callable, repeats: int, flush: Any | None) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    elapsed = 0.0
    if flush is None:
        start.record()
        for _ in range(repeats):
            run()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / repeats
    # Every timed invocation starts after the eviction buffer write. Timing an
    # entire loop after one flush would make all but its first call warm-L2.
    for _ in range(repeats):
        flush.add_(1)
        start.record()
        run()
        end.record()
        end.synchronize()
        elapsed += float(start.elapsed_time(end))
    return elapsed / repeats


def _require_cupti() -> tuple[Callable, dict]:
    """Resolve CUPTI explicitly before invoking a helper which permits fallback."""
    try:
        importlib.import_module("cupti.cupti")
        version = importlib.metadata.version("cupti-python")
        if int(version.split(".")[0]) < 13:
            raise ValueError(f"cupti-python >= 13 is required, installed: {version}")
        utilities = importlib.import_module("flashinfer.testing.utils")
        function = utilities.bench_gpu_time_with_cupti
        flashinfer = importlib.import_module("flashinfer")
        source_hash = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    except Exception as error:
        raise TimingUnavailable(f"strict CUPTI timer unavailable: {error}") from error
    return function, {
        "cupti": version,
        "timer_flashinfer": str(getattr(flashinfer, "__version__", "unknown")),
        "timer_source_sha256": source_hash,
        "kernel_qualified": True,
    }


def _timed_cupti(function: Callable, run: Callable, repeats: int, cold_l2: bool) -> float:
    def reject_fallback(*args, **kwargs):
        raise TimingUnavailable("CUPTI requested: substitution with an event timer is forbidden")

    # The inspected FlashInfer helper has these two fallback routes. Guard both
    # entry points as well as warning text so a removed warning cannot silently
    # turn a kernel-qualified experiment into an event measurement.
    namespace = getattr(function, "__globals__", {})
    saved = {
        name: namespace[name]
        for name in ("bench_gpu_time_with_cuda_event", "bench_gpu_time_with_cudagraph")
        if name in namespace
    }
    namespace.update(dict.fromkeys(saved, reject_fallback))
    try:
        with warnings.catch_warnings():
            # The helper permits fallback normally. A selected timer must not.
            warnings.filterwarnings("error", message=r"(?i).*fall.*back.*")
            samples = function(
                fn=run,
                dry_run_iters=0,
                repeat_iters=repeats,
                use_cuda_graph=True,
                cold_l2_cache=cold_l2,
            )
    except Exception as error:
        raise TimingUnavailable(f"strict CUPTI measurement failed: {error}") from error
    finally:
        namespace.update(saved)
    if len(samples) != repeats:
        raise TimingUnavailable("CUPTI did not return one activity measurement per iteration")
    if any(not math.isfinite(float(value)) or value <= 0 for value in samples):
        raise TimingUnavailable("CUPTI returned nonfinite or nonpositive activity timing")
    return sum(float(value) for value in samples) / len(samples)


def paired_benchmark(
    torch: Any,
    baseline: Callable,
    candidate: Callable,
    settings: dict,
    flush: Any | None = None,
    cupti_function: Callable | None = None,
) -> tuple[list[float], list[float]]:
    """Each sample is an ABBA trial: mean(A1,A2), mean(B1,B2)."""
    for _ in range(settings["warmup"]):
        baseline()
        candidate()
    torch.cuda.synchronize()
    baseline_samples, candidate_samples = [], []
    if settings.get("timing", "cuda_event") == "cupti":
        if cupti_function is None:
            raise TimingUnavailable("strict CUPTI timer was not initialized")
        measure = lambda run: _timed_cupti(
            cupti_function, run, settings["repeats"], settings.get("cold_l2", False)
        )
    else:
        measure = lambda run: _timed(torch, run, settings["repeats"], flush)
    for _ in range(settings["trials"]):
        a1 = measure(baseline)
        b1 = measure(candidate)
        b2 = measure(candidate)
        a2 = measure(baseline)
        baseline_samples.append((a1 + a2) / 2)
        candidate_samples.append((b1 + b2) / 2)
    if any(
        not math.isfinite(value) or value <= 0 for value in baseline_samples + candidate_samples
    ):
        raise RuntimeError("CUDA event measurement was not finite and strictly positive")
    return baseline_samples, candidate_samples
