"""Lower fixed E4M3FN vectors to FP16 or FP32 through inline assembly."""

from ._mlir_util import walk_ops

# With nonzero rd and rs1=zero, vsetvli sets VL to VLMAX.
# e8 selects the input element width; ta/ma select tail/mask agnostic policies.
_FP8_ASM = (
    "vsetvli t0, zero, e8, {lmul}, ta, ma\n\t"
    "vfwcvt.f.f.v $0, $1"
)
_FP8_CONSTRAINTS = "=&^vr,^vr,~{t0},~{vl},~{vtype},~{fflags}"

def _fp8_lanes_and_lmul(n, vlen):
    """Return the scalable vector's base element count and FP8 input LMUL."""
    # VLMAX = k * vscale = k * (VLEN / 64) = n
    vscale = vlen // 64     # LLVM scalable-vector multiplier
    k = n // vscale

    if n % vscale != 0:
        raise ValueError("n must be divisible by vscale")
    if k not in (1, 2, 4, 8, 16, 32, 64):
        raise ValueError(f"Unsupported scalable-vector element count: {k}")

    lmul = k / 8.0
    if(lmul < 1.0):
        lmul = f"f{int(1 / lmul)}"
    else:
        lmul = f"{int(lmul)}"
    return k, f"m{lmul}"


def _fp8_convert(ip, val, n, vlen):
    """Convert a fixed i8 carrier vector to a fixed FP16 vector."""
    from mlir.ir import IntegerType, F16Type, VectorType
    from mlir.dialects import llvm, vector

    k, lmul = _fp8_lanes_and_lmul(n, vlen)
    i8 = IntegerType.get_signless(8)
    f16 = F16Type.get()
    src_ty = VectorType.get([k], i8, scalable=[True])
    dst_ty = VectorType.get([k], f16, scalable=[True])

    with ip:
        big = vector.ScalableInsertOp(val, llvm.UndefOp(src_ty).result, 0).result

        got = llvm.InlineAsmOp(
            dst_ty, [big], _FP8_ASM.format(lmul=lmul), _FP8_CONSTRAINTS,
            has_side_effects=True, asm_dialect=0,).res

        small = VectorType.get([n], f16)
        return vector.ScalableExtractOp(small, got, 0).result


def _convert(ip, source, result_type, vlen):
    """Convert a fixed FP8 vector to FP16, extending to FP32 when requested."""
    from mlir.ir import Operation, IntegerType, VectorType

    shape = VectorType(source.type).shape
    n = shape[0]
    i8_type = VectorType.get(shape, IntegerType.get_signless(8))

    with ip:
        # Preserve the FP8 bits; numerical conversion happens in Spike.
        raw = Operation.create(
            "arith.bitcast", results=[i8_type], operands=[source]
        ).results[0]

    half = _fp8_convert(ip, raw, n, vlen)

    # If the result type is FP16, we can return early.
    if half.type == result_type:
        return half

    with ip:
        # Extend FP16 -> FP32.
        return Operation.create(
            "arith.extf", results=[result_type], operands=[half]
        ).results[0]


def run(module, vlen=256, **_):
    """Replace fixed-vector E4M3FN extensions to FP16 or FP32."""
    from mlir.ir import InsertionPoint
    import re

    targets = []
    for region in module.operation.regions:
        for block in region.blocks:
            for op in walk_ops(block):
                if op.operation.name != "arith.extf":
                    continue

                source_type = str(op.operation.operands[0].type)
                result_type = str(op.operation.results[0].type)
                match = re.fullmatch(r"vector<(\d+)xf8E4M3FN>", source_type)

                if match:
                    n = int(match.group(1))
                    if result_type in (f"vector<{n}xf16>", f"vector<{n}xf32>"):
                        targets.append(op.operation)

    for op in targets:
        converted = _convert(InsertionPoint(op), op.operands[0],
                             op.results[0].type, vlen)

        op.results[0].replace_all_uses_with(converted)
        op.erase()
