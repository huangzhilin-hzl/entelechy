# SPDX-License-Identifier: Apache-2.0
"""CUDA-event benchmark for the first Entelechy kernel."""

import argparse
import json
import statistics

import torch

from entelechy import vector_add


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=1 << 24)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.elements, device="cuda", dtype=dtype)
    y = torch.randn_like(x)
    out = torch.empty_like(x)

    for _ in range(args.warmup):
        vector_add(x, y, out=out)
    torch.cuda.synchronize()

    samples_us = []
    for _ in range(args.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        vector_add(x, y, out=out)
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1_000)

    torch.testing.assert_close(out, x + y)
    median_us = statistics.median(samples_us)
    print(
        json.dumps(
            {
                "kernel": "vector_add",
                "gpu": torch.cuda.get_device_name(),
                "sm": "".join(map(str, torch.cuda.get_device_capability())),
                "dtype": args.dtype,
                "elements": args.elements,
                "median_us": median_us,
                "effective_bandwidth_gbps": args.elements
                * x.element_size()
                * 3
                / median_us
                / 1_000,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
