"""Validate E4M3FN -> FP16 alone and report simulated kernel costs.

The cycle reports include loads/stores and DMA, not just vfwcvt latency.
"""

import os
from pathlib import Path
import re
import tempfile

import torch


def test_fp8_to_fp16(device, numel=4096):
    from PyTorchSimFrontend import extension_config

    def convert(x):
        return x.half()

    if numel <= 0 or numel % 256:
        raise ValueError("numel must be a positive multiple of 256")

    dump_path = Path(extension_config.get_dump_path())
    # Reinterpret bytes to include every encoding, both signed zeros, and NaNs.
    bits = torch.tensor(range(256), dtype=torch.uint8).repeat(numel // 256)
    source = bits.view(torch.float8_e4m3fn)
    expected = convert(source)
    compiled = torch.compile(dynamic=False)(convert)
    actual = compiled(source.to(device)).cpu()

    assert actual.dtype == torch.float16, f"Expected FP16, got {actual.dtype}"
    assert actual.shape == expected.shape
    nan_mask = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan_mask), "NaN locations differ"
    finite_mask = ~nan_mask
    # Compare bits so that +0 and -0 must also be preserved. NaN payloads may differ.
    mismatches = finite_mask & (actual.view(torch.int16) != expected.view(torch.int16))
    if mismatches.any():
        indices = mismatches.nonzero().flatten()[:8].tolist()
        details = [(f"0x{int(bits[i]):02x}", actual[i].item(), expected[i].item())
                   for i in indices]
        raise AssertionError(f"FP8 -> FP16 mismatches (bits, actual, CPU): {details}")

    ir_files = [p for p in dump_path.rglob("*_llvm.mlir") if "_sample_" not in p.name]
    conversion_files = [p for p in ir_files if "vfwcvt.f.f.v" in p.read_text()]
    assert conversion_files, f"No inline vector conversion found under {dump_path}"
    print(f"FP8 -> FP16: {numel} elements, all 256 encodings checked -- PASS", flush=True)
    print(f"  Exact finite values: {int(finite_mask.sum())}; NaNs: {int(nan_mask.sum())}")
    for path in conversion_files:
        print(f"  LLVM IR: {path}")
        print(f"  Assembly: {path.with_name(path.name.removesuffix('_llvm.mlir') + '.s')}")
    for path in sorted(dump_path.rglob("trace_cycles.tsv")):
        rows = [line for line in path.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        print(f"  Gem5 compute-tile cycles (cycles, overlap): {rows}")
        print(f"  Cycle table: {path}")

    log_path = Path(extension_config.CONFIG_TORCHSIM_LOG_PATH)
    for path in sorted(log_path.glob("*.log")):
        cycles = re.findall(
            r"(?:Total execution cycles:|\[TOGSim-trace\] Total cycles:)\s*(\d+)",
            path.read_text(),
        )
        if cycles:
            print(f"  TOGSim kernel cycles: {cycles[-1]} (includes memory transfers)")
            print(f"  TOGSim log: {path}")


def main():
    from PyTorchSimFrontend import extension_config

    if not extension_config.pytorchsim_functional_mode:
        raise RuntimeError("Enable pytorchsim_functional_mode in the hardware YAML to check Spike results")
    if not extension_config.pytorchsim_timing_mode:
        raise RuntimeError("Enable pytorchsim_timing_mode in the hardware YAML to measure kernel cycles")

    root = Path(os.environ.get("TORCHSIM_DUMP_PATH") or tempfile.mkdtemp(prefix="fp8-to-fp16-"))
    print(f"FP8 -> FP16 artifacts: {root}", flush=True)
    old_dump = os.environ.get("TORCHSIM_DUMP_PATH")
    old_log = os.environ.get("TORCHSIM_LOG_PATH")
    try:
        for numel in (512, 1024, 4096, 65536):
            case_path = root / f"n{numel}"
            os.environ["TORCHSIM_DUMP_PATH"] = str(case_path)
            os.environ["TORCHSIM_LOG_PATH"] = str(case_path / "togsim_results")
            test_fp8_to_fp16(torch.device("npu:0"), numel)
    finally:
        for name, value in (("TORCHSIM_DUMP_PATH", old_dump), ("TORCHSIM_LOG_PATH", old_log)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    main()
