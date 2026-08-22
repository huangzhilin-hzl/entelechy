# SPDX-License-Identifier: Apache-2.0
"""Minimal CuTe DSL kernel used to validate the project workflow."""

from __future__ import annotations

from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _vector_add_kernel(
    x: cute.Tensor,
    y: cute.Tensor,
    out: cute.Tensor,
    size: cutlass.Int32,
    threads: cutlass.Constexpr,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    idx = block_idx * threads + thread_idx
    if idx < size:
        out[idx] = x[idx] + y[idx]


@cute.jit
def _launch(
    x: cute.Tensor,
    y: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
    threads: cutlass.Constexpr = 256,
):
    size = x.shape[0]
    _vector_add_kernel(x, y, out, size, threads).launch(
        grid=(cute.ceil_div(size, threads), 1, 1),
        block=(threads, 1, 1),
        stream=stream,
    )


def vector_add(x: Any, y: Any, *, out: Any | None = None) -> Any:
    """Add two contiguous CUDA tensors with identical shape and dtype."""

    import torch

    if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
        raise ValueError("x and y must have identical shape, dtype, and device")
    if x.device.type != "cuda":
        raise ValueError("Entelechy requires CUDA tensors")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("x and y must be contiguous")
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != x.dtype or out.device != x.device:
        raise ValueError("out must match x in shape, dtype, and device")
    elif not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if x.numel() == 0:
        return out

    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    _launch(
        from_dlpack(x.view(-1), assumed_align=16),
        from_dlpack(y.view(-1), assumed_align=16),
        from_dlpack(out.view(-1), assumed_align=16),
        stream,
    )
    return out
