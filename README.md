# Efficient Inference 项目指南

**概览**
- 提供基于H2O的attention优化、基于int8 weight-only量化的MLP优化，支持 BF16/FP16 activation、CUDA Graph。
- 目标架构：NVIDIA Ampere 及以上（sm_80+），不依赖 cuBLAS/CUTLASS，内核使用 WMMA/mma.sync。

**环境要求**
- GPU：NVIDIA Ampere 或更高（A100、RTX 30/40 系列），支持 BF16/Tensor Core
- CUDA 工具链：与 PyTorch 对齐（示例使用 CUDA 12.8）
- Python：3.10 及以上
- 依赖包：
  - torch>=2.9.1（CUDA 12.x 对应的官方轮子）
  - transformers、datasets、tqdm
  - 可选：nsight-systems（性能分析）


```bash
conda create -n opt python=3.10 -y
conda activate opt
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets tqdm
```

**构建 INT8 CUDA 扩展**

```bash
cd int8_weight_only
pip install . --no-build-isolation
```



**准备模型与数据（离线）**
- 模型缓存：建议将 Pythia-2.8B 离线缓存到本地，例如：`/home/xlyu/models/models--EleutherAI--pythia-2.8b`。
- 数据集：脚本使用本地 Wikitext 路径 `/home/xlyu/datasets/wikitext`。确保该目录存在并包含标准拆分（`split='test'`）。

**生成 INT8 Quantization Checkpoint**
- 使用提供的脚本将 Linear 权重量化为 weight-only INT8（per-channel scale），输出到 `checkpoints/pythia-2.8b-int8`：

```bash
python src/quantize_int8.py \
  --model_path /home/xlyu/models/models--EleutherAI--pythia-2.8b \
  --out_dir checkpoints/pythia-2.8b-int8 \
  --dtype float16
```

- 生成文件：
  - `weights_int8.pt`（所有 Linear 的 INT8 权重）
  - `scales_fp32.pt`（每输出通道 FP32 缩放）
  - `bias_fp16.pt`（若存在 bias，保存为 FP16）

**运行benchmark**
- 脚本会顺序跑三种模式（FP16 全注意力、INT8 全注意力、INT8+H2O）：

```bash
python src/benchmark_int8.py
```

- 关键参数：
  - `--model_path`：本地模型路径（默认：`/home/xlyu/models/models--EleutherAI--pythia-2.8b`）
  - `--ckpt_dir`：INT8 检查点目录（默认：`checkpoints/pythia-2.8b-int8` 或相邻目录，视脚本版本而定）
  - `--prefill_len`、`--decode_len`：预填充与解码长度（默认：500/1000）
  - `--window`、`--h2o_hh`：H2O 窗口与 Heavy Hitter 大小（默认：252/252）
  - `--batch_size`：批大小（默认：1；CUDA Graph 针对解码仅在 batch_size=1 时启用）

- 示例（覆盖路径与长度）：

```bash
python src/benchmark_int8.py \
  --model_path /home/xlyu/models/models--EleutherAI--pythia-2.8b \
  --ckpt_dir checkpoints/pythia-2.8b-int8 \
  --prefill_len 128 --decode_len 256 --batch_size 1
```

**CUDA Graph 使用说明**
- 在 `int8_full` 与 `int8_h2o` 模式且 `batch_size=1` 时，脚本会在预填充后对所有 Int8Linear 进行图化，解码阶段优先走图化的前向：
  - 图化入口：启用于 [benchmark_int8.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/src/benchmark_int8.py#L203-L221)
  - 模块实现：见 [int8_linear.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/int8_weight_only/int8_linear.py#L34-L50)

**性能分析（可选）**
- 使用 Nsight Systems：

```bash
nsys profile -o profile_fp16_full --force-overwrite true --stats=true \
  python src/benchmark_int8.py
```

**目录结构**
- `src/`：基准、量化脚本与 KV 缓存管理
  - [benchmark_int8.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/src/benchmark_int8.py)
  - [quantize_int8.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/src/quantize_int8.py)
  - [kv_cache.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/src/kv_cache.py)
- `int8_weight_only/`：PyTorch 扩展与 Python 封装
  - [setup.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/setup.py)
  - [binding.cpp](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/int8_weight_only/binding.cpp)
  - [w8a16_gemm.cu](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/int8_weight_only/w8a16_gemm.cu)
  - [int8_linear.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/int8_weight_only/int8_linear.py)
- `checkpoints/pythia-2.8b-int8/`：INT8 weight-only 检查点（需提前生成或准备）


