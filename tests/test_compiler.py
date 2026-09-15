# SPDX-License-Identifier: Apache-2.0
"""CPU tests for schedule legality and the CUDA compiler's public trace_input."""

import dataclasses
import json
import re

import pytest

from entelechy.compiler import (
    Barrier,
    Schedule,
    ScheduleSchemaError,
    ScheduleValidationError,
    SharedResource,
    compile_cuda,
    verify,
)


def codes(schedule, **kwargs):
    return {item.code for item in verify(schedule, **kwargs).diagnostics}


@pytest.mark.parametrize("operator", ["rmsnorm", "softmax", "silu_mul"])
@pytest.mark.parametrize("mapping", ["warp_per_row", "cta_per_row"])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_supported_schedules_roundtrip_and_lower(operator, mapping, dtype):
    schedule = Schedule.for_mapping(mapping, threads=128, items_per_thread=4, operator=operator)
    restored = Schedule.from_dict(json.loads(json.dumps(schedule.to_dict())))
    assert restored == schedule
    assert restored.schedule_id == schedule.schedule_id
    assert verify(schedule, operator, 129, dtype, "sm_80").valid
    source = compile_cuda(schedule, operator, dtype)
    assert source == compile_cuda(restored, operator, dtype)
    assert "void run(const at::Tensor& x" in source
    assert "#ifndef ENTELECHY_STANDALONE" in source
    assert "CUDAGuard device_guard(x.device())" in source
    assert "getCurrentCUDAStream(x.get_device())" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK()" in source
    assert "if (col < cols)" in source
    assert source.count("{") == source.count("}")
    if mapping == "cta_per_row" and operator != "silu_mul":
        assert "__shared__ float warp_partials[4]" in source
        assert source.index("publish warp_partials") < source.index("release warp_partials")
        assert verify(schedule, operator).shared_memory_bytes == 16
    else:
        assert "__syncthreads" not in source
        assert verify(schedule, operator).shared_memory_bytes == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("threads", 33),
        ("threads", True),
        ("threads", 0),
        ("threads", 1056),
        ("items_per_thread", 3),
        ("items_per_thread", False),
        ("mapping", "async_warp"),
    ],
)
def test_invalid_hardware_config_cannot_lower(field, value):
    schedule = dataclasses.replace(Schedule(), **{field: value})
    assert not verify(schedule).valid
    with pytest.raises(ScheduleValidationError):
        compile_cuda(schedule, "softmax", "float32")


def test_unknown_serialized_fields_are_never_silently_ignored():
    with pytest.raises(ScheduleSchemaError, match="unknown fields"):
        Schedule.from_dict({"mapping": "warp_per_row", "cache_policy": "cg"})
    with pytest.raises(ScheduleSchemaError, match="unknown fields"):
        Schedule.from_dict({"resources": [{"name": "warp_partials", "elements": 4, "bytes": 16}]})
    with pytest.raises(ScheduleSchemaError, match="unknown fields"):
        Schedule.from_dict(
            {
                "barriers": [
                    {"name": "ready", "resource": "warp_partials", "role": "publish", "skip": True}
                ]
            }
        )
    with pytest.raises(ScheduleSchemaError):
        Schedule.from_dict({"resources": None})


def test_missing_release_is_a_dataflow_error():
    schedule = Schedule.for_mapping("cta_per_row")
    schedule = dataclasses.replace(schedule, barriers=schedule.barriers[:1])
    assert "REDUCTION_DATAFLOW" in codes(schedule, operator="rmsnorm")
    with pytest.raises(ScheduleValidationError, match="REDUCTION_DATAFLOW"):
        compile_cuda(schedule, "rmsnorm", "float32")


def test_release_before_publish_is_rejected():
    schedule = Schedule.for_mapping("cta_per_row")
    assert "REDUCTION_DATAFLOW" in codes(
        dataclasses.replace(schedule, barriers=tuple(reversed(schedule.barriers))),
        operator="softmax",
    )


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("scope", "warp", "BARRIER_SCOPE"),
        ("participants", "warp_0", "BARRIER_DIVERGENCE"),
        ("resource", "unknown", "UNKNOWN_RESOURCE"),
        ("role", "wait", "BARRIER_ROLE"),
        ("name", "bad;code", "INVALID_BARRIER_NAME"),
    ],
)
def test_illegal_sync_declarations(field, value, code):
    schedule = Schedule.for_mapping("cta_per_row")
    barrier = dataclasses.replace(schedule.barriers[0], **{field: value})
    schedule = dataclasses.replace(schedule, barriers=(barrier, schedule.barriers[1]))
    assert code in codes(schedule, operator="softmax")


def test_missing_and_undersized_shared_storage():
    base = Schedule.for_mapping("cta_per_row", threads=256)
    missing = dataclasses.replace(base, resources=())
    assert {"MISSING_REDUCTION_RESOURCE", "UNKNOWN_RESOURCE"} <= codes(missing, operator="rmsnorm")
    short = dataclasses.replace(base, resources=(SharedResource("warp_partials", 7),))
    assert "RESOURCE_SIZE" in codes(short, operator="rmsnorm")
    low_precision = dataclasses.replace(
        base, resources=(SharedResource("warp_partials", 8, "float16"),)
    )
    assert "RESOURCE_DTYPE" in codes(low_precision, operator="rmsnorm")


def test_protocol_cannot_be_attached_to_an_unused_mapping_or_operator():
    cta = Schedule.for_mapping("cta_per_row")
    assert "UNUSED_PROTOCOL" in codes(cta, operator="silu_mul")
    assert "UNUSED_PROTOCOL" in codes(
        dataclasses.replace(cta, mapping="warp_per_row"), operator="rmsnorm"
    )
    assert "MISSING_REDUCTION_RESOURCE" in codes(
        Schedule(mapping="cta_per_row"), operator="rmsnorm"
    )


def test_duplicate_and_excessive_resources():
    schedule = Schedule.for_mapping("cta_per_row")
    duplicate = dataclasses.replace(schedule, resources=schedule.resources * 2)
    assert "DUPLICATE_RESOURCE" in codes(duplicate, operator="rmsnorm")
    oversized = dataclasses.replace(schedule, resources=(SharedResource("warp_partials", 16384),))
    assert "SHARED_MEMORY_LIMIT" in codes(oversized, operator="rmsnorm")
    duplicate_barrier = dataclasses.replace(
        schedule,
        barriers=(schedule.barriers[0], Barrier("partials_ready", "warp_partials", "release")),
    )
    assert "DUPLICATE_BARRIER" in codes(duplicate_barrier, operator="rmsnorm")


def test_workload_and_target_constraints():
    schedule = Schedule()
    assert "UNSUPPORTED_OPERATOR" in codes(schedule, operator="matmul")
    assert "UNSUPPORTED_DTYPE" in codes(schedule, dtype="fp16")
    for cols in (0, -1, 1.5, True, 2**31):
        assert "INVALID_SHAPE" in codes(schedule, cols=cols)
    assert verify(schedule, dtype="float16", target="sm_70").valid
    assert verify(schedule, dtype="bfloat16", target="sm_100a").valid
    assert "UNSUPPORTED_TARGET" in codes(schedule, dtype="bfloat16", target="sm_75")
    assert "UNSUPPORTED_TARGET" in codes(schedule, target="cpu")
    with pytest.raises(ScheduleValidationError):
        compile_cuda(schedule, "matmul", "float32")


def test_changes_to_physical_schedule_change_source_and_identity():
    a = Schedule(threads=128, items_per_thread=1)
    b = Schedule(threads=256, items_per_thread=4)
    assert a.schedule_id != b.schedule_id
    source = compile_cuda(b, "softmax", "float32")
    assert "constexpr int THREADS = 256;" in source
    assert "constexpr int ITEMS = 4;" in source
    assert "constexpr int ROWS_PER_CTA = 8;" in source
    assert source != compile_cuda(a, "softmax", "float32")


def test_host_boundary_validates_shape_dtype_aliases_and_empty_rows():
    source = compile_cuda(Schedule(), "rmsnorm", "float16")
    for fragment in (
        "x.scalar_type() == at::kHalf",
        "out.scalar_type() == x.scalar_type()",
        "x.device() == out.device()",
        "x.is_contiguous() && out.is_contiguous()",
        "x.dim() == 2 && out.dim() == 2",
        "!overlaps(x, out)",
        "!overlaps(out, weight)",
        "weight.device() == x.device()",
        "weight.numel() == cols",
        "if (rows == 0) return",
    ):
        assert fragment in source
    silu = compile_cuda(Schedule(), "silu_mul", "float16")
    assert "x.size(1) == 2 * cols" in silu
    assert "weight.numel() == 0" in silu


def test_softmax_subtracts_row_max_before_every_exponential():
    source = compile_cuda(Schedule(), "softmax", "float32")
    exponentials = re.findall(r"expf\(([^;]+)", source)
    assert len(exponentials) == 2
    assert all("- row_max" in expression for expression in exponentials)
    assert source.index("row_max = row_reduce<true>") < source.index("float denominator")
    assert source.index("denominator = row_reduce<false>") < source.index("numerator / denominator")


def test_diagnostics_are_machine_readable_and_stable():
    report = verify(Schedule(threads=17), operator="rmsnorm")
    data = json.loads(json.dumps(report.to_dict()))
    assert data["valid"] is False
    assert data["diagnostics"][0] == {
        "code": "INVALID_THREADS",
        "path": "threads",
        "message": "threads must be a multiple of 32 in [32, 1024]",
        "severity": "error",
    }


@pytest.mark.parametrize("operator,dtype", [(None, "float32"), ("softmax", None)])
def test_lowering_never_infers_a_missing_workload(operator, dtype):
    with pytest.raises(ScheduleValidationError):
        compile_cuda(Schedule(), operator, dtype)


def test_malformed_typed_objects_produce_diagnostics_instead_of_lowering():
    assert "SCHEMA_TYPE" in codes({"mapping": "warp_per_row"})
    assert "SCHEMA_TYPE" in codes(Schedule(resources=[{"name": "warp_partials"}]))
    assert "SCHEMA_TYPE" in codes(Schedule(barriers=("all",)))
    malformed = Schedule.for_mapping("cta_per_row")
    malformed = dataclasses.replace(malformed, resources=(SharedResource(["warp_partials"], True),))
    assert {"INVALID_RESOURCE_NAME", "RESOURCE_SIZE"} <= codes(malformed, operator="softmax")


@pytest.mark.cuda
@pytest.mark.parametrize("operator", ["rmsnorm", "softmax", "silu_mul"])
@pytest.mark.parametrize("mapping", ["warp_per_row", "cta_per_row"])
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
def test_generated_cuda_executes_tails_on_current_stream(
    tmp_path, monkeypatch, operator, mapping, dtype_name
):
    """Compile the actual generated extension; no timing assertions or mocks.

    A three-warp block tests non-power-of-two CTA reductions. Boundary row and
    column counts exercise inactive warps and masked, unrolled loads/stores.
    """
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires an NVIDIA GPU")
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        pytest.skip("requires a CUDA toolkit for generated-source compilation")
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    if major < 7 or (dtype_name == "bfloat16" and major < 8):
        pytest.skip("device does not support this compiler target")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
    schedule = Schedule.for_mapping(mapping, threads=96, items_per_thread=4, operator=operator)
    source = tmp_path / "kernel.cu"
    source.write_text(compile_cuda(schedule, operator, dtype_name))
    name = f"entelechy_smoke_{operator}_{mapping}_{dtype_name}"
    module = load(
        name=name,
        sources=[str(source)],
        build_directory=str(tmp_path),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
        verbose=False,
    )
    dtype = getattr(torch, dtype_name)
    tolerance = {"float16": 2e-3, "bfloat16": 2e-2, "float32": 3e-5}[dtype_name]
    generator = torch.Generator().manual_seed(1531)
    stream = torch.cuda.Stream(device=device)
    for rows, cols in [(1, 1), (7, 17), (9, 33), (13, 129), (2, 769), (5, 4097)]:
        input_cols = cols * (2 if operator == "silu_mul" else 1)
        with torch.cuda.stream(stream):
            x = torch.randn(rows, input_cols, generator=generator).to(
                device=f"cuda:{device}", dtype=dtype
            )
            weight = (
                torch.randn(cols, generator=generator).to(device=x.device, dtype=dtype)
                if operator == "rmsnorm"
                else torch.empty(0, device=x.device, dtype=dtype)
            )
            # The reference uses CPU FP64 values reconstructed from quantized
            # inputs, independently of the generated reduction schedule.
            values = x.cpu().double()
            if operator == "rmsnorm":
                expected = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6)
                expected = expected * weight.cpu().double()
            elif operator == "softmax":
                expected = torch.softmax(values, dim=-1)
            else:
                gate, up = values.chunk(2, dim=-1)
                expected = torch.nn.functional.silu(gate) * up
            storage = torch.full((rows * cols + 128,), 123.0, device=x.device, dtype=dtype)
            out = storage[64:-64].view(rows, cols)
            out.fill_(float("nan"))
            original_x, original_weight = x.clone(), weight.clone()
            for _ in range(3):
                module.run(x, weight, out, 1e-6)
        stream.synchronize()
        assert torch.isfinite(out).all()
        assert torch.all(storage[:64] == 123.0)
        assert torch.all(storage[-64:] == 123.0)
        assert torch.equal(x, original_x)
        assert torch.equal(weight, original_weight)
        torch.testing.assert_close(out.cpu().double(), expected, atol=tolerance, rtol=tolerance)

    x = torch.empty(2, 16, dtype=dtype, device=device)
    weight = (
        torch.ones(16, dtype=dtype, device=device)
        if operator == "rmsnorm"
        else torch.empty(0, dtype=dtype, device=device)
    )
    with pytest.raises(RuntimeError, match="overlap"):
        module.run(x, weight, x if operator != "silu_mul" else x.view(4, 8)[:2], 1e-6)
    if operator == "rmsnorm":
        with pytest.raises(RuntimeError, match="overlap weight"):
            module.run(x[:1], weight, weight.view(1, 16), 1e-6)
    output_cols = 8 if operator == "silu_mul" else 16
    module.run(x[:0], weight, torch.empty(0, output_cols, dtype=dtype, device=device), 1e-6)
