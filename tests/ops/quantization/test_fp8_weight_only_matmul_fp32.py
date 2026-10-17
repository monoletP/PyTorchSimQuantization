import os
import sys
import torch
from pathlib import Path
from PyTorchSimFrontend.mlir.mlir_caller_codegen import MLIRKernelCallerCodeGen

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_fp8_weight_only_matmul(device, input_size=128, hidden_size=128, output_size=128):
    def fp32_matmul(a, b):
        return torch.matmul(a, b)

    def fp8_matmul(a, b, scale):
        return torch.matmul(a, b.float() * scale)

    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size)
    weight = torch.randn(hidden_size, output_size)
    fp8_dtype = torch.float8_e4m3fn
    scale = (weight.abs().max().clamp(min=1e-12) / torch.finfo(fp8_dtype).max).item()

    weight_fp8 = (weight / scale).to(fp8_dtype)

    x1 = input.to(device=device)
    w1 = weight_fp8.to(device=device)
    x2 = input.to("cpu")
    w2 = weight_fp8.to("cpu")

    weight_fp32 = w2.float() * scale
    y = fp32_matmul(x2, weight_fp32)

    print(
        f"FP32 baseline: M={input_size}, K={hidden_size}, N={output_size}",
        flush=True,
    )
    opt_fp32 = torch.compile(dynamic=False)(fp32_matmul)
    res_fp32 = opt_fp32(x1, weight_fp32.to(device=device))
    test_result("FP32 Matmul Forward", res_fp32, y)

    print(
        f"FP8 weight-only: M={input_size}, K={hidden_size}, N={output_size}",
        flush=True,
    )
    opt_fn = torch.compile(dynamic=False)(fp8_matmul)
    res = opt_fn(x1, w1, scale)
    test_result("FP8 Weight-Only Matmul Forward", res, y)


def main():
    device = torch.device("npu:0")
    test_fp8_weight_only_matmul(device, 32, 32, 32)
    test_fp8_weight_only_matmul(device, 128, 128, 128)
    test_fp8_weight_only_matmul(device, 256, 256, 256)
    test_fp8_weight_only_matmul(device, 128, 256, 256)
    test_fp8_weight_only_matmul(device, 128, 63, 56)

    reader = MLIRKernelCallerCodeGen(validation=True, arg_attributes=[])
    dump_path = Path(os.environ["TORCHSIM_DUMP_PATH"])

    for binary in sorted(dump_path.glob("*/validation_binary")):
        size = reader.get_spad_size(str(binary))
        if size <= 0:
            raise RuntimeError(f"Cannot measure SRAM allocation: {binary}")
        print(f"kernel {binary.parent.name}: SRAM = {size} bytes/lane")


if __name__ == "__main__":
    main()
