# SPDX-License-Identifier: Apache-2.0
"""Deterministic CUDA lowering for validated contiguous row schedules."""

from __future__ import annotations

from textwrap import dedent

from .ir import Schedule
from .verify import Diagnostic, ScheduleValidationError, VerificationReport, verify


def _loop(schedule: Schedule, body: str) -> str:
    return dedent(
        f"""\
        for (int64_t base = column_lane; base < cols; base += ROW_THREADS * ITEMS) {{
        #pragma unroll
          for (int item = 0; item < ITEMS; ++item) {{
            const int64_t col = base + static_cast<int64_t>(item) * ROW_THREADS;
            if (col < cols) {{
        {body}
            }}
          }}
        }}
        """
    )


def compile_cuda(schedule: Schedule, operator: str, dtype: str) -> str:
    """Return a self-contained .cu source with ``run(x, weight, out, eps)``.

    Define ENTELECHY_STANDALONE to compile the CUDA kernel without PyTorch or
    pybind headers. The exported kernel's columns always mean output columns;
    silu_mul's input row contains twice that many columns.
    """
    # Unlike verify's exploratory mode, lowering requires a complete workload.
    missing = tuple(
        Diagnostic(f"UNSUPPORTED_{field.upper()}", f"{field} must be specified", field)
        for field, value in (("operator", operator), ("dtype", dtype))
        if value is None
    )
    if missing:
        raise ScheduleValidationError(VerificationReport(missing))
    report = verify(schedule, operator=operator, dtype=dtype)
    if not report.valid:
        raise ScheduleValidationError(report)
    scalar, torch_dtype, read, write = {
        "float16": ("__half", "at::kHalf", "__half2float(value)", "__float2half_rn(value)"),
        "bfloat16": (
            "__nv_bfloat16",
            "at::kBFloat16",
            "__bfloat162float(value)",
            "__float2bfloat16_rn(value)",
        ),
        "float32": ("float", "at::kFloat", "value", "value"),
    }[dtype]
    cta = schedule.mapping == "cta_per_row"
    reduction = operator != "silu_mul"
    rows_per_cta = 1 if cta else schedule.threads // 32
    row_threads = schedule.threads if cta else 32
    lane = "threadIdx.x" if cta else "(threadIdx.x & 31)"
    row = (
        "static_cast<int64_t>(blockIdx.x)"
        if cta
        else "static_cast<int64_t>(blockIdx.x) * ROWS_PER_CTA + threadIdx.x / 32"
    )
    shared = ""
    helpers = ""
    reduce_arg = ""
    if reduction:
        helpers = dedent(
            """\
            template <bool Maximum>
            __device__ __forceinline__ float warp_reduce(float value) {
              #pragma unroll
              for (int offset = 16; offset > 0; offset /= 2) {
                const float peer = __shfl_down_sync(0xffffffffu, value, offset);
                value = Maximum ? fmaxf(value, peer) : value + peer;
              }
              return value;
            }
            """
        )
        if cta:
            shared = f"__shared__ float warp_partials[{schedule.resources[0].elements}];"
            ready, consumed = schedule.barriers
            reduce_arg = ", warp_partials"
            helpers += dedent(
                f"""\
                template <bool Maximum>
                __device__ __forceinline__ float row_reduce(float value, float* warp_partials) {{
                  const int lane = threadIdx.x & 31;
                  const int warp = threadIdx.x / 32;
                  value = warp_reduce<Maximum>(value);
                  if (lane == 0) warp_partials[warp] = value;
                  __syncthreads(); // {ready.name}: publish warp_partials to all warps
                  value = lane < THREADS / 32 ? warp_partials[lane] : (Maximum ? -INFINITY : 0.0f);
                  __syncthreads(); // {consumed.name}: release warp_partials before reuse
                  value = warp_reduce<Maximum>(value);
                  return __shfl_sync(0xffffffffu, value, 0);
                }}
                """
            )
        else:
            helpers += dedent(
                """\
                template <bool Maximum>
                __device__ __forceinline__ float row_reduce(float value) {
                  value = warp_reduce<Maximum>(value);
                  return __shfl_sync(0xffffffffu, value, 0);
                }
                """
            )

    if operator == "rmsnorm":
        body = "float sum_squares = 0.0f;\n"
        body += _loop(
            schedule,
            "      const float value = load_scalar(x[row * cols + col]);\n"
            "      sum_squares += value * value;",
        )
        body += f"sum_squares = row_reduce<false>(sum_squares{reduce_arg});\n"
        body += "const float inverse_rms = rsqrtf(sum_squares / static_cast<float>(cols) + eps);\n"
        body += _loop(
            schedule,
            "      const float value = load_scalar(x[row * cols + col]);\n"
            "      const float scale = load_scalar(weight[col]);\n"
            "      out[row * cols + col] = store_scalar((value * inverse_rms) * scale);",
        )
    elif operator == "softmax":
        body = "float row_max = -INFINITY;\n"
        body += _loop(schedule, "      row_max = fmaxf(row_max, load_scalar(x[row * cols + col]));")
        body += f"row_max = row_reduce<true>(row_max{reduce_arg});\nfloat denominator = 0.0f;\n"
        body += _loop(
            schedule, "      denominator += expf(load_scalar(x[row * cols + col]) - row_max);"
        )
        body += f"denominator = row_reduce<false>(denominator{reduce_arg});\n"
        body += _loop(
            schedule,
            "      const float numerator = expf(load_scalar(x[row * cols + col]) - row_max);\n"
            "      out[row * cols + col] = store_scalar(numerator / denominator);",
        )
    else:
        body = _loop(
            schedule,
            "      const float gate = load_scalar(x[row * (2 * cols) + col]);\n"
            "      const float value = load_scalar(x[row * (2 * cols) + cols + col]);\n"
            "      const float activation = gate * (1.0f / (1.0f + expf(-gate)));\n"
            "      out[row * cols + col] = store_scalar(activation * value);",
        )

    weight_guard = (
        dedent(
            """\
          TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
          TORCH_CHECK(weight.device() == x.device(), "weight device must match x");
          TORCH_CHECK(weight.scalar_type() == x.scalar_type(), "weight dtype must match x");
          TORCH_CHECK(weight.is_contiguous() && weight.dim() == 1 && weight.numel() == cols,
                      "rmsnorm weight must be contiguous with shape [cols]");
          TORCH_CHECK(!overlaps(out, weight), "out must not overlap weight");
        """
        )
        if operator == "rmsnorm"
        else '  TORCH_CHECK(weight.numel() == 0, "weight tensor must be empty");\n'
    )
    weight_ptr = (
        "reinterpret_cast<const scalar_t*>(weight.data_ptr())"
        if operator == "rmsnorm"
        else "nullptr"
    )
    input_multiplier = 2 if operator == "silu_mul" else 1
    bf16_guard = (
        dedent(
            """\
          cudaDeviceProp properties;
          C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, x.get_device()));
          TORCH_CHECK(properties.major >= 8, "bfloat16 requires sm_80 or newer");
        """
        )
        if dtype == "bfloat16"
        else ""
    )
    return dedent(
        f"""\
        // SPDX-License-Identifier: Apache-2.0
        // Generated by Entelechy's bounded row-schedule compiler.
        // schedule_id={schedule.schedule_id}; operator={operator}; dtype={dtype}
        // Define ENTELECHY_STANDALONE to compile the kernel without PyTorch.
        #include <cuda_runtime.h>
        #include <cuda_fp16.h>
        #include <cuda_bf16.h>
        #include <cstdint>
        #include <cmath>
        #include <climits>
        #ifndef ENTELECHY_STANDALONE
        #include <torch/extension.h>
        #include <c10/cuda/CUDAGuard.h>
        #include <c10/cuda/CUDAStream.h>
        #include <c10/cuda/CUDAException.h>
        #endif

        using scalar_t = {scalar};
        constexpr int THREADS = {schedule.threads};
        constexpr int ITEMS = {schedule.items_per_thread};
        constexpr int ROW_THREADS = {row_threads};
        constexpr int ROWS_PER_CTA = {rows_per_cta};

        __device__ __forceinline__ float load_scalar(scalar_t value) {{ return {read}; }}
        __device__ __forceinline__ scalar_t store_scalar(float value) {{ return {write}; }}

        {helpers}
        extern "C" __global__ void entelechy_kernel(
            const scalar_t* __restrict__ x,
            const scalar_t* __restrict__ weight,
            scalar_t* __restrict__ out,
            int64_t rows, int64_t cols, float eps) {{
          const int64_t row = {row};
          // Row validity is uniform across the synchronization group.
          if (row >= rows) return;
          const int column_lane = {lane};
          {shared}
          {body}
        }}

        #ifndef ENTELECHY_STANDALONE
        namespace {{
        bool overlaps(const at::Tensor& left, const at::Tensor& right) {{
          if (left.numel() == 0 || right.numel() == 0) return false;
          const uintptr_t l = reinterpret_cast<uintptr_t>(left.data_ptr());
          const uintptr_t r = reinterpret_cast<uintptr_t>(right.data_ptr());
          const uint64_t lb = static_cast<uint64_t>(left.numel()) * left.element_size();
          const uint64_t rb = static_cast<uint64_t>(right.numel()) * right.element_size();
          return l <= r ? static_cast<uint64_t>(r - l) < lb : static_cast<uint64_t>(l - r) < rb;
        }}
        }}

        void run(const at::Tensor& x, const at::Tensor& weight, at::Tensor out, double eps) {{
          TORCH_CHECK(x.is_cuda() && out.is_cuda(), "x and out must be CUDA tensors");
          TORCH_CHECK(x.device() == out.device(), "out device must match x");
          TORCH_CHECK(x.scalar_type() == {torch_dtype}, "x dtype does not match compiled kernel");
          TORCH_CHECK(out.scalar_type() == x.scalar_type(), "out dtype must match x");
          TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "x and out must be 2D");
          TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "x and out must be contiguous");
          const int64_t rows = out.size(0);
          const int64_t cols = out.size(1);
          TORCH_CHECK(cols > 0 && cols <= INT_MAX, "cols must be in [1, INT_MAX]");
          TORCH_CHECK(x.size(0) == rows && x.size(1) == {input_multiplier} * cols,
                      "input shape does not match output shape for {operator}");
          TORCH_CHECK(std::isfinite(eps) && eps >= 0.0 && eps <= 3.4028234663852886e38,
                      "eps must be finite, nonnegative and representable in float32");
          TORCH_CHECK(!overlaps(x, out), "out must not overlap x");
        {weight_guard}
          c10::cuda::CUDAGuard device_guard(x.device());
        {bf16_guard}
          if (rows == 0) return;
          const int64_t blocks = rows / ROWS_PER_CTA + (rows % ROWS_PER_CTA != 0);
          TORCH_CHECK(blocks <= INT_MAX, "row count exceeds CUDA grid.x limit");
          const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
          entelechy_kernel<<<static_cast<unsigned int>(blocks), THREADS, 0, stream.stream()>>>(
              reinterpret_cast<const scalar_t*>(x.data_ptr()),
              {weight_ptr},
              reinterpret_cast<scalar_t*>(out.data_ptr()), rows, cols, static_cast<float>(eps));
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }}

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {{
          module.def("run", &run, "Entelechy {operator} ({schedule.schedule_id})");
        }}
        #endif
        """
    )
