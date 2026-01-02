import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

# Compile just-in-time
module_path = os.path.dirname(__file__)
w8a16_gemm = load(
    name="w8a16_gemm",
    sources=[
        os.path.join(module_path, "binding.cpp"),
        os.path.join(module_path, "w8a16_gemm.cu"),
    ],
    extra_cuda_cflags=["-O3", "-arch=sm_80", "--use_fast_math"],
    verbose=True
)

def test_correctness():
    M = 128
    N = 4096
    K = 4096
    
    device = "cuda"
    dtype = torch.bfloat16
    
    # Inputs
    X = torch.randn(M, K, device=device, dtype=dtype)
    W_fp = torch.randn(N, K, device=device, dtype=torch.float32)
    
    # Quantize W
    scales = W_fp.abs().amax(dim=1) / 127.0
    scales = scales.float()
    W_int8 = torch.clamp((W_fp / scales[:, None]).round(), -127, 127).to(torch.int8)
    
    # Reference (De-quantize)
    # Match kernel logic: int8 -> float -> * scale(float) -> bf16
    W_deq = (W_int8.float() * scales[:, None]).to(dtype)
    Y_ref = F.linear(X, W_deq)
    
    # Kernel
    Y_out = torch.zeros(M, N, device=device, dtype=dtype)
    
    # Warmup
    w8a16_gemm.forward(X, W_int8, scales, Y_out)
    

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    w8a16_gemm.forward(X, W_int8, scales, Y_out)
    end.record()
    torch.cuda.synchronize()
    
    print(f"Kernel time: {start.elapsed_time(end):.3f} ms")
    
    # Verify
    diff = (Y_out - Y_ref).abs()
    print(f"Max diff: {diff.max().item()}")
    print(f"Mean diff: {diff.mean().item()}")
    print(f"Y_ref mean: {Y_ref.abs().mean().item()}")
    
    # Find mismatches
    mismatch_mask = diff > 2.0 # Allow slightly larger diff due to bf16 rounding
    if mismatch_mask.any():
        indices = torch.nonzero(mismatch_mask)
        print("Mismatches found:")
        for idx in indices[:5]:
            r, c = idx.tolist()
            print(f"Y[{r},{c}]: Ref={Y_ref[r,c].item()}, Kernel={Y_out[r,c].item()}, Diff={diff[r,c].item()}")
            

            acc = 0.0
            x_row = X[r].float()
            w_row = W_int8[c].float()
            scale = scales[c].item()
            w_deq = w_row * scale
            dot = (x_row * w_deq).sum()
            print(f"  Manual Dot (float): {dot.item()}")
            

    rel_err = diff / (Y_ref.abs() + 1e-5)
    print(f"Mean rel err: {rel_err.mean().item()}")

    
    if diff.max() <= 2.0:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":
    torch.manual_seed(0)
    test_correctness()
