# SPDX-License-Identifier: Apache-2.0

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.cuda
@pytest.mark.parametrize("shape", [(0,), (17,), (3, 5), (4096,)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_vector_add(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    from entelechy import vector_add

    x = torch.randn(shape, device="cuda", dtype=dtype)
    y = torch.randn_like(x)
    torch.testing.assert_close(vector_add(x, y), x + y)
