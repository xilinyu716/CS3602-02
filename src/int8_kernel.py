import triton
import triton.language as tl
import torch

@triton.jit
def int8_linear_kernel(
    X_ptr,          # fp16 [B, K]
    Wq_ptr,         # int8  [M, K]
    Scale_ptr,      # fp16  [M]
    Y_ptr,          # fp16  [B, M]
    B, M, K,
    stride_xb, stride_xk,
    stride_wm, stride_wk,
    stride_yb, stride_ym,
    BLOCK_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_B, BLOCK_M), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_idx = k + offs_k

        x = tl.load(
            X_ptr
            + offs_b[:, None] * stride_xb
            + k_idx[None, :] * stride_xk,
            mask=(offs_b[:, None] < B) & (k_idx[None, :] < K),
            other=0.0,
        )

        wq = tl.load(
            Wq_ptr
            + offs_m[None, :] * stride_wm
            + k_idx[:, None] * stride_wk,
            mask=(offs_m[None, :] < M) & (k_idx[:, None] < K),
            other=0,
        )

        wq_fp16 = wq.to(tl.float16)

        acc += tl.dot(x, wq_fp16)

    scale = tl.load(
        Scale_ptr + offs_m,
        mask=offs_m < M,
        other=0.0,
    )

    acc = acc * scale[None, :]
    acc = acc.to(tl.float16)

    tl.store(
        Y_ptr
        + offs_b[:, None] * stride_yb
        + offs_m[None, :] * stride_ym,
        acc,
        mask=(offs_b[:, None] < B) & (offs_m[None, :] < M),
    )



def int8_linear_triton(x, Wq, scale):
    """
    x: [B, T, K] fp16
    Wq: [M, K] int8
    scale: [M] fp16
    """
    B, T, K = x.shape
    M, _ = Wq.shape

    x2d = x.reshape(B * T, K)
    y2d = torch.empty((B * T, M), device=x.device, dtype=torch.float16)

    BLOCK_B = 64
    BLOCK_M = 64
    BLOCK_K = 32

    grid = (
        triton.cdiv(B * T, BLOCK_B),
        triton.cdiv(M, BLOCK_M),
    )

    int8_linear_kernel[grid](
        x2d, Wq, scale, y2d,
        B * T, M, K,
        x2d.stride(0), x2d.stride(1),
        Wq.stride(0), Wq.stride(1),
        y2d.stride(0), y2d.stride(1),
        BLOCK_B=BLOCK_B,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    return y2d.reshape(B, T, M)


