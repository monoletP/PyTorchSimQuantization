"""Expand scalar E4M3FN-to-FP32 conversions before LLVM lowering."""

from ._mlir_util import walk_ops

def run(module, **_):
    """Replace scalar E4M3FN-to-FP32 extf operations in place."""
    # %9 = arith.extf %8 : vector<64xf8E4M3FN> to vector<64xf32>
    from mlir.ir import InsertionPoint
    
    def _decode(source):
        """Decode E4M3FN value to FP32."""
        
        from mlir.ir import Operation, IntegerType, IntegerAttr, F32Type, FloatAttr, VectorType, DenseElementsAttr
        
        is_vector = VectorType.isinstance(source.type)
        shape = VectorType(source.type).shape if is_vector else None

        def same_shape(element_type):
            return VectorType.get(shape, element_type) if is_vector else element_type

        i32_scalar = IntegerType.get_signless(32)
        f32_scalar = F32Type.get()

        i1 = same_shape(IntegerType.get_signless(1))
        i8 = same_shape(IntegerType.get_signless(8))
        i32 = same_shape(i32_scalar)
        f32 = same_shape(f32_scalar)
        i64 = IntegerType.get_signless(64)
        
        def _convert_i32(value):
            """Create a 32-bit integer constant."""
            element = IntegerAttr.get(i32_scalar, value)
            if is_vector:
                return Operation.create("arith.constant", results=[i32],
                                        attributes={"value": DenseElementsAttr.get_splat(i32, element)}).results[0]
            else :
                return Operation.create("arith.constant", results=[i32], attributes={"value": element}).results[0]
            
        def _convert_f32(value):
            """Create a 32-bit float constant."""
            element = FloatAttr.get(f32_scalar, value)
            if is_vector:
                return Operation.create("arith.constant", results=[f32],
                                        attributes={"value": DenseElementsAttr.get_splat(f32, element)}).results[0]
            else :
                return Operation.create("arith.constant", results=[f32], attributes={"value": element}).results[0]
            
        def _andi(value, mask):
            """Bitwise AND a 32-bit integer value with `mask`."""
            return Operation.create("arith.andi", results=[i32], operands=[value, _convert_i32(mask)]).results[0]
        
        def _shift_left(value, shift):
            """Shift a 32-bit integer value left by `shift` bits."""
            if(shift == 0) :
                return value
            elif(shift > 0) :
                return Operation.create("arith.shli", results=[i32], operands=[value, _convert_i32(shift)]).results[0]
            else :
                return Operation.create("arith.shrui", results=[i32], operands=[value, _convert_i32(-shift)]).results[0]
        
        raw8 = Operation.create("arith.bitcast", results=[i8], operands=[source]).results[0]
        raw = Operation.create("arith.extui", results=[i32], operands=[raw8]).results[0]
        
        # E4M3FN format: 1 sign bit, 4 exponent bits, 3 mantissa bits``
        sign_bit = _shift_left(_andi(raw, 0b10000000), 24)
        exponent = _shift_left(_andi(raw, 0b01111000), -3)
        mantissa = _andi(raw, 0b00000111)
        
        # fp8E4M3FN to fp32 conversion:
        # fp32 : 1 sign bit / 8 exponent bits / 23 mantissa bits
        # if exponent is non-zero, (-1)^sign x (1 + mantissa / 8) x 2^ (exponent - 7)
        # just add 120 to exponent and shift left by 23 bits
        exponent_plus_120 = Operation.create("arith.addi", results=[i32], operands=[exponent, _convert_i32(120)]).results[0]
        exponent_shifted = _shift_left(exponent_plus_120, 23)
        mantissa_shifted = _shift_left(mantissa, 20)
        normal_bits = Operation.create("arith.ori", results=[i32], operands=[exponent_shifted, mantissa_shifted]).results[0]
        
        # if exponent is zero, (-1)^sign x (mantissa) x 2^(-9)
        # just divide mantissa by 512 (2^9)
        mantissa_fp32 = Operation.create("arith.uitofp", results=[f32], operands=[mantissa]).results[0]
        mantissa_div512 = Operation.create("arith.divf", results=[f32], operands=[mantissa_fp32, _convert_f32(512.0)]).results[0]
        denormal_bits = Operation.create("arith.bitcast", results=[i32], operands=[mantissa_div512]).results[0]
        
        is_zero_exponent = Operation.create("arith.cmpi", results=[i1], operands=[exponent, _convert_i32(0)], attributes={"predicate": IntegerAttr.get(i64, 0)}).results[0]
        result_bit = Operation.create("arith.select", results=[i32], operands=[is_zero_exponent, denormal_bits, normal_bits]).results[0]
        
        # 0b01111111 -> Nan, 0b11111111 -> -Nan
        magnitude = _andi(raw, 0b01111111)
        is_nan = Operation.create("arith.cmpi", results=[i1], operands=[magnitude, _convert_i32(0b01111111)], attributes={"predicate": IntegerAttr.get(i64, 0)}).results[0]
        result_bit = Operation.create("arith.select", results=[i32], operands=[is_nan, _convert_i32(0x7fc00000), result_bit]).results[0]
        
        result_bit = Operation.create("arith.ori", results=[i32], operands=[result_bit, sign_bit]).results[0]
        
        return Operation.create("arith.bitcast", results=[f32], operands=[result_bit]).results[0]

    targets = []
    for region in module.operation.regions:
        for block in region.blocks:
            for op in walk_ops(block):
                if op.operation.name != "arith.extf":
                    continue

                source_type = str(op.operation.operands[0].type)
                result_type = str(op.operation.results[0].type)
                if ((source_type == "f8E4M3FN" and result_type == "f32")
                    or (source_type == "vector<64xf8E4M3FN>" and result_type == "vector<64xf32>")):
                    targets.append(op.operation)

    for op in targets:
        with InsertionPoint(op):
            converted = _decode(op.operands[0])

        op.results[0].replace_all_uses_with(converted)
        op.erase()