# SPDX-License-Identifier: Apache-2.0
"""Isolated GPU execution; importing this package does not import PyTorch."""


def doctor(device: int = 0) -> dict:
    """Inspect CUDA availability without making CUDA a package dependency."""
    from .worker import doctor as inspect_device

    return inspect_device(device)


__all__ = ["doctor"]
