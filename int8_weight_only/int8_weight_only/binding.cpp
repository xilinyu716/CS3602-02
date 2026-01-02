#include <torch/extension.h>
#include <vector>

// Forward declaration
void w8a16_gemm_forward_cuda(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor scales,
    torch::Tensor Bias,
    torch::Tensor Y,
    int act_type
);

// C++ interface
void w8a16_gemm_forward(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor scales,
    torch::Tensor Bias,
    torch::Tensor Y,
    int act_type
) {
    w8a16_gemm_forward_cuda(X, W, scales, Bias, Y, act_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &w8a16_gemm_forward, "W8A16 GEMM Forward (CUDA)");
}
