import os
import sys
import torch

sys.path.insert(0, os.path.join(os.environ.get("TORCHSIM_DIR", default="/workspace/PyTorchSim"), "tests"))
from _pytorchsim_utils import test_result

def test_fp8_weight_only_matmul(device, input_size=128, hidden_size=128, output_size=128):
    def custom_matmul(a, b, scale):
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

    opt_fn = torch.compile(dynamic=False)(custom_matmul)
    res = opt_fn(x1, w1, scale)
    y = custom_matmul(x2, w2, scale)
    test_result("FP8 Weight-Only Matmul Forward", res, y)


def main():
    device = torch.device("npu:0")
    test_fp8_weight_only_matmul(device, 32, 32, 32)


if __name__ == "__main__":
    main()
