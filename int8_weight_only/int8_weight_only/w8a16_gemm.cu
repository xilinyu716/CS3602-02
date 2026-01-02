#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cassert>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;

#define BN 128
#define BK 32

typedef __device_builtin__ __attribute__((aligned(16))) int4 int4_a16;

template <int BM>
__global__ void w8a16_gemm_kernel(
    const __nv_bfloat16* __restrict__ X,
    const int8_t* __restrict__ W,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ Bias,
    __nv_bfloat16* __restrict__ Y,
    int M, int N, int K,
    int act_type // 0=None, 1=GELU
) {
    // Block index
    int bx = blockIdx.x; // M dimension
    int by = blockIdx.y; // N dimension
    
    // Thread index
    int tid = threadIdx.x;
    int warp_id = tid / 32;

    // Shared memory for double buffering
    extern __shared__ __nv_bfloat16 smem[];
    __nv_bfloat16 (*smem_X)[BM][BK] = (__nv_bfloat16 (*)[BM][BK])smem;
    __nv_bfloat16 (*smem_W)[BN][BK] = (__nv_bfloat16 (*)[BN][BK])(smem + 2 * BM * BK);

    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag[4][2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][4];
    
    // Initialize accumulator
    #pragma unroll
    for(int i=0; i<4; ++i) {
        #pragma unroll
        for(int j=0; j<4; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // Warp Mapping
    int warp_row, warp_col;
    if (BM == 128) {
        // 4 warps. 2x2.
        warp_row = (warp_id / 2) * 64;
        warp_col = (warp_id % 2) * 64;
    } else { 
        // BM=64 (2 warps) or BM=32 (2 warps)
        // 1x2 grid.
        warp_row = 0;
        warp_col = warp_id * 64;
    }

    const __nv_bfloat16* X_base = X + bx * BM * K;
    const int8_t* W_base = W + by * BN * K;

    constexpr int threads_per_block = (BM == 128) ? 128 : 64;
    constexpr int rows_per_thread_w = BN / threads_per_block;
    
    float my_scales[2] = {0.0f, 0.0f};
    
    #pragma unroll
    for(int r=0; r<rows_per_thread_w; ++r) {
        int w_row_local = tid + r * blockDim.x;
        int w_row_global = by * BN + w_row_local;
        if (w_row_global < N) {
             my_scales[r] = scales[w_row_global];
        }
    }
    
    int4 ldg_W_arr[2][2];

    const __nv_bfloat16* src_x = X_base + tid * K; 
    bool valid_x_row = (tid < BM) && ((bx * BM + tid) < M);
    
    uint32_t smem_x_ptr = __cvta_generic_to_shared(&smem_X[0][tid][0]);
    
    if (valid_x_row) {
        // Load 64 bytes (BK=32)
        // 4 x 16 bytes
        #pragma unroll
        for(int i=0; i<4; ++i) {
            // cp.async requires byte-aligned pointers.
            const void* src = (const void*)(src_x + i * 8); // 8 bf16 = 16 bytes
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_x_ptr + i * 16), "l"(src));
        }
    } else {
        if (tid < BM) {
            // Zero out 64 bytes
             int4* dst = (int4*)&smem_X[0][tid][0];
             #pragma unroll
             for(int i=0; i<4; ++i) dst[i] = make_int4(0,0,0,0);
        }
    }

    // Load W (Synchronous)
    #pragma unroll
    for(int r=0; r<rows_per_thread_w; ++r) {
        int w_row_local = tid + r * blockDim.x;
        const int8_t* src_w = W_base + w_row_local * K;
        #pragma unroll
        for(int i=0; i<2; ++i) {
             ldg_W_arr[r][i] = *((const int4*)(src_w + i * 16));
        }
    }
    
    // Commit X load
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;"); // Wait for X

    __syncthreads(); 

    #pragma unroll
    for(int r=0; r<rows_per_thread_w; ++r) {
        int w_row_local = tid + r * blockDim.x;
        __nv_bfloat16* dst_w = &smem_W[0][w_row_local][0];
        int8_t* w_int8 = (int8_t*)&ldg_W_arr[r]; 
        
        #pragma unroll
        for(int i=0; i<32; ++i) {
            float val = (float)w_int8[i];
            dst_w[i] = __float2bfloat16(val * my_scales[r]);
        }
    }
    
    __syncthreads();

    int num_tiles = K / BK;
    

    constexpr int num_row_tiles = (BM == 32) ? 2 : 4;

    for (int k_tile = 0; k_tile < num_tiles; ++k_tile) {
        int buf_compute = k_tile % 2;
        int buf_load = (k_tile + 1) % 2;
        
    
        if (k_tile < num_tiles - 1) {
            int next_k = k_tile + 1;
            
            const __nv_bfloat16* sx = X_base + tid * K + next_k * BK;
            uint32_t smem_x_ptr_next = __cvta_generic_to_shared(&smem_X[buf_load][tid][0]);
            
            if (valid_x_row) {
                #pragma unroll
                for(int i=0; i<4; ++i) {
                    const void* src = (const void*)(sx + i * 8);
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "r"(smem_x_ptr_next + i * 16), "l"(src));
                }
            } else {
                 // Zeroing logic if needed
            }
            
            asm volatile("cp.async.commit_group;");
            
            #pragma unroll
            for(int r=0; r<rows_per_thread_w; ++r) {
                int w_row_local = tid + r * blockDim.x;
                const int8_t* sw = W_base + w_row_local * K + next_k * BK;
                #pragma unroll
                for(int i=0; i<2; ++i) {
                    ldg_W_arr[r][i] = *((const int4*)(sw + i * 16));
                }
            }
        }
        
        // 2. Compute Tile k
        #pragma unroll
        for(int k_step = 0; k_step < 2; ++k_step) {
            int k_curr = k_step * 16;
            
            // Load A (X)
            #pragma unroll
            for(int i=0; i<num_row_tiles; ++i) {
                wmma::load_matrix_sync(a_frag[i][k_step], &smem_X[buf_compute][warp_row + i*16][k_curr], BK);
            }
            
            // Load B (W^T)
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
            #pragma unroll
            for(int j=0; j<4; ++j) {
                wmma::load_matrix_sync(b_frag, &smem_W[buf_compute][warp_col + j*16][k_curr], BK);
                
                #pragma unroll
                for(int i=0; i<num_row_tiles; ++i) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i][k_step], b_frag, c_frag[i][j]);
                }
            }
        }
        
        // 3. Sync
        // Wait for X load
        if (k_tile < num_tiles - 1) {
             asm volatile("cp.async.wait_group 0;");
             __syncthreads(); 
             
             // Store W (Dequantize)
             #pragma unroll
             for(int r=0; r<rows_per_thread_w; ++r) {
                int w_row_local = tid + r * blockDim.x;
                __nv_bfloat16* dw = &smem_W[buf_load][w_row_local][0];
                int8_t* wi = (int8_t*)&ldg_W_arr[r];
                #pragma unroll
                for(int i=0; i<32; ++i) {
                    float val = (float)wi[i];
                    dw[i] = __float2bfloat16(val * my_scales[r]);
                }
            }
            __syncthreads();
        }
    }

    // Store Output
    float* smem_f = (float*)smem + warp_id * 256;
    int lane_id = tid % 32;

    #pragma unroll
    for(int i=0; i<num_row_tiles; ++i) {
        #pragma unroll
        for(int j=0; j<4; ++j) {
            // Store tile to smem
            wmma::store_matrix_sync(smem_f, c_frag[i][j], 16, wmma::mem_row_major);
            __syncwarp();
            
            #pragma unroll
            for(int k=0; k<8; ++k) {
                int idx = lane_id + k * 32; // 0..255
                int r = idx / 16;
                int c = idx % 16;
                float val = smem_f[idx];
                
                int global_row = bx * BM + warp_row + i * 16 + r;
                int global_col = by * BN + warp_col + j * 16 + c;
                
                if (global_row < M && global_col < N) {
                    __nv_bfloat16 val_bf = __float2bfloat16(val);
                    if (Bias != nullptr) {
                        val_bf = __hadd(val_bf, Bias[global_col]);
                    }
                    if (act_type == 1) { // GELU

                        float x = __bfloat162float(val_bf);
                        // Standard tanh approximation
                        float c = 0.7978845608028654f; // sqrt(2/pi)
                        float tanh_in = c * (x + 0.044715f * x * x * x);
                        float tanh_out = tanhf(tanh_in);
                        float gelu = 0.5f * x * (1.0f + tanh_out);
                        val_bf = __float2bfloat16(gelu);
                    }
                    Y[global_row * N + global_col] = val_bf;
                }
            }
            __syncwarp();
        }
    }
}

__global__ void w8a16_gemv_kernel(
    const __nv_bfloat16* __restrict__ X,
    const int8_t* __restrict__ W,
    const float* __restrict__ scales,
    const __nv_bfloat16* __restrict__ Bias,
    __nv_bfloat16* __restrict__ Y,
    int N, int K,
    int act_type
) {
    int warp_id = threadIdx.y; 
    int lane_id = threadIdx.x; 
    
    
    int n_idx = blockIdx.x * 4 + warp_id;
    
    if (n_idx >= N) return;
    
    float scale = scales[n_idx];
    float sum = 0.0f;
    
    const int8_t* w_ptr = W + n_idx * K;
    const __nv_bfloat16* x_ptr = X;
    
    
    int k_loop = K / 16; 
    
    const int4* x_vec = (const int4*)x_ptr;
    const int4* w_vec = (const int4*)w_ptr;
    
    
    for (int k = lane_id; k < k_loop; k += 32) {
        int4 w_val = w_vec[k];

        int4 x_val_0 = x_vec[k * 2];
        int4 x_val_1 = x_vec[k * 2 + 1];
        
        int8_t* w_i8 = (int8_t*)&w_val;
        __nv_bfloat16* x_bf16_0 = (__nv_bfloat16*)&x_val_0;
        __nv_bfloat16* x_bf16_1 = (__nv_bfloat16*)&x_val_1;
        
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum += (float)w_i8[i] * __bfloat162float(x_bf16_0[i]);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            sum += (float)w_i8[i+8] * __bfloat162float(x_bf16_1[i]);
        }
    }
    
    
    sum *= scale;
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    if (lane_id == 0) {
        __nv_bfloat16 res = __float2bfloat16(sum);
        if (Bias != nullptr) {
            res = __hadd(res, Bias[n_idx]);
        }
        if (act_type == 1) { // GELU
             float x = __bfloat162float(res);
             float c = 0.7978845608028654f; 
             float tanh_in = c * (x + 0.044715f * x * x * x);
             float tanh_out = tanhf(tanh_in);
             float gelu = 0.5f * x * (1.0f + tanh_out);
             res = __float2bfloat16(gelu);
        }
        Y[n_idx] = res;
    }
}

void w8a16_gemm_forward_cuda(
    torch::Tensor X,
    torch::Tensor W,
    torch::Tensor scales,
    torch::Tensor Bias,
    torch::Tensor Y,
    int act_type
) {
    int M = X.size(0);
    int K = X.size(1);
    int N = W.size(0);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const __nv_bfloat16* bias_ptr = (Bias.defined() && Bias.numel() > 0) ? reinterpret_cast<const __nv_bfloat16*>(Bias.data_ptr<at::BFloat16>()) : nullptr;
    
    if (M == 1) {

        dim3 block(32, 4);
        dim3 grid((N + 3) / 4);
        
        w8a16_gemv_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
            reinterpret_cast<const int8_t*>(W.data_ptr<int8_t>()),
            scales.data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
            N, K, act_type
        );
    } else if (M <= 32) {
        // BM=32
        dim3 block32(64); // 2 warps
        dim3 grid((M + 31) / 32, (N + 127) / 128);
        int smem_size = 32768;
        
        w8a16_gemm_kernel<32><<<grid, block32, smem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
            reinterpret_cast<const int8_t*>(W.data_ptr<int8_t>()),
            scales.data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
            M, N, K, act_type
        );
    } else if (M <= 64) {
    
        dim3 block64(64); 
        dim3 grid((M + 63) / 64, (N + 127) / 128);
        int smem_size = 32768;
        
        w8a16_gemm_kernel<64><<<grid, block64, smem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
            reinterpret_cast<const int8_t*>(W.data_ptr<int8_t>()),
            scales.data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
            M, N, K, act_type
        );
    } else {
        dim3 block(128);
        dim3 grid((M + 127) / 128, (N + 127) / 128);
        int smem_size = 32768;
        
        if (smem_size > 48 * 1024) {
            cudaFuncSetAttribute(w8a16_gemm_kernel<128>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        }
        
        w8a16_gemm_kernel<128><<<grid, block, smem_size, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(X.data_ptr<at::BFloat16>()),
            reinterpret_cast<const int8_t*>(W.data_ptr<int8_t>()),
            scales.data_ptr<float>(),
            bias_ptr,
            reinterpret_cast<__nv_bfloat16*>(Y.data_ptr<at::BFloat16>()),
            M, N, K, act_type
        );
    }
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
