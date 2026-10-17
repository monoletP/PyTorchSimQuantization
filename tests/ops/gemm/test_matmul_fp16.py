"""Compare FP16 matmul against CPU without FP8 conversion or scaling."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", "/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result


def test_matmul_fp16(device, input_size=128, hidden_size=128, output_size=128):
    def matmul(a, b):
        return torch.matmul(a, b)

    torch.manual_seed(0)
    input = torch.randn(input_size, hidden_size).half()
    weight = torch.randn(hidden_size, output_size).half()
    opt_fn = torch.compile(dynamic=False)(matmul)
    result = opt_fn(input.to(device), weight.to(device))
    expected = matmul(input, weight)

    assert result.dtype == torch.float16, f"Expected FP16 output, got {result.dtype}"
    actual = result.cpu()
    max_diff = (actual.float() - expected.float()).abs().max().item()
    mismatches = (~torch.isclose(actual, expected, rtol=1e-3, atol=1e-3)).sum().item()
    shape = (input_size, hidden_size, output_size)
    print(f"FP16 Matmul (M, K, N)={shape}: max_abs_diff={max_diff}, "
          f"mismatches={mismatches}/{expected.numel()}", flush=True)
    test_result(f"FP16 Matmul {shape}", actual, expected, rtol=1e-3, atol=1e-3)


def main():
    device = torch.device("npu:0")
    shapes = [(32, 32, 32), (128, 128, 128), (256, 256, 256),
              (128, 256, 256), (128, 63, 56)]
    failed = []
    for shape in shapes:
        try:
            test_matmul_fp16(device, *shape)
        except SystemExit as error:
            if error.code != 1:
                raise
            failed.append(shape)

    print(f"FP16 Matmul: {len(shapes) - len(failed)}/{len(shapes)} cases passed", flush=True)
    if failed:
        print(f"Failed (M, K, N): {failed}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
