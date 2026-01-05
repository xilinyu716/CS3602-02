# SJTU CS3602-02 课程大作业个人部分

**概览**

- KV Press: 实现了H2O (Heavy Hitter Oracle)优化
- Linear: 实现了基于W8A16 Quantization的自定义cuda kernel，并采用了与GeLU Activation的算子融合减少自定义kernel引入的额外 Kernel Launch 开销，与原版hugginface model对齐
- 目标架构：NVIDIA Ampere 及以上（sm_80+），不依赖 cuBLAS/CUTLASS，内核使用 WMMA/mma.sync。

**环境要求**
- GPU：NVIDIA Ampere 或更高（A100、RTX 30/40 系列），支持 BF16/Tensor Core
- CUDA 工具链：与 PyTorch 对齐（示例使用 CUDA 12.8）
- Python：3.10 及以上
- 依赖包：
  - torch
  - transformers、datasets、tqdm
  - 可选：nsight-systems（性能分析）


```bash
conda create -n opt python=3.10 -y
conda activate opt
pip install torch
pip install transformers datasets tqdm
pip install accelerate
```

**编译自定义的cuda kernel**

```bash
cd int8_weight_only
pip install . --no-build-isolation
```



**准备模型与数据**
- 模型缓存：建议将 Pythia-2.8B 离线缓存到本地，例如：`/home/xlyu/models/models--EleutherAI--pythia-2.8b`。
- 数据集：脚本使用本地 Wikitext 路径，例如： `/home/xlyu/datasets/wikitext`。确保该目录存在并包含标准拆分（`split='test'`）。

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
  - `--prefill_len`、`--decode_len`：预填充与解码长度（默认：128/2048）
  - `--window`、`--h2o_hh`：H2O 窗口与 Heavy Hitter 大小（默认：252/252）
  - `--batch_size`：批处理大小（默认：1；CUDA Graph 针对解码仅在 batch_size=1 时启用）
  - `--dataset`: 数据集路径 （默认：`/home/xlyu/datasets/wikitext`）

- 示例（覆盖路径与长度）：

```bash
python src/benchmark.py \
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
  - [benchmark.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/src/benchmark_int8.py)
  - [quantize_int8.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/src/quantize_int8.py)
  - [kv_cache.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/src/kv_cache.py)
- `int8_weight_only/`：PyTorch 扩展与 Python 封装
  - [setup.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/setup.py)
  - [binding.cpp](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/int8_weight_only/binding.cpp)
  - [w8a16_gemm.cu](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/int8_weight_only/w8a16_gemm.cu)
  - [int8_linear.py](file:///home/xlyu/AI_HW/%E5%A4%A7%E4%BD%9C%E4%B8%9A/FlashAttention/FlashAttention-Optimization/int8_weight_only/int8_weight_only/int8_linear.py)
- `checkpoints/pythia-2.8b-int8/`：INT8 weight-only 检查点（需提前生成或准备）


**部分实验结果展示**
Pythia-2.8B 在 A100 GPU 上的基准测试结果（prefill_len=128, decode_len=2048）：

**batch size = 1**
| Mode | Throughput (tokens/sec) | Peak GPU Memory (MB) | PPL | Total Time (s) |
|------|------------------------|---------------------|-----|----------------|
| FP16 Full | 49.80 | 6674.16 | 7.8750 | 43.70 |
| INT8 Full | 44.85 | 4306.36 | 8.0000 | 48.51 |
| INT8 H2O | 36.72 | 6017.51 | 10.4375 | 59.25 |

**batch size = 16**
| Mode | Throughput (tokens/sec) | Peak GPU Memory (MB) | PPL | Total Time (s) |
|------|------------------------|---------------------|-----|----------------|
| FP16 Full | 249.58 | 27242.63 | 7.8750 | 139.50 |
| INT8 Full | 200.19 | 25065.00 | 8.0000 | 173.91 |
| INT8 H2O | 279.14 | 10778.59 | 10.4375 | 123.73 |


## 实验结果总结

1. **权重量化效果显著**
   - INT8 权重量化相比 FP16 在单批次场景（batch_size=1）下节省约 35% GPU 内存（4306 MB vs 6674 MB）
   - 模型精度损失最小（PPL 从 7.875 增至 8.0，损失仅 1.6%）
   - 吞吐量略有下降（44.85 vs 49.80 tokens/sec，约 10% 性能开销）

2. **H2O 稀疏注意力的双重效应**

   **在单批次（batch_size=1）场景**：
   - H2O 虽然保持了内存占用（6017 MB），但由于需要维护重要性得分，内存优化效果有限
   - 吞吐量下降明显（36.72 tokens/sec），总耗时增加约 22%
   - 精度损失增大（PPL 升至 10.4375，约 32% 相对损失）
   - 本场景中 H2O 的优化空间受限

   **在高批次（batch_size=16）场景**：
   - H2O 展现出强大的内存优化能力：仅需 10778 MB 内存，相比 FP16 的 27243 MB 节省 60%
   - 吞吐量反而提升（279.14 vs 249.58 tokens/sec，约 12% 性能提升）
   - 总运行时间缩短（123.73 vs 139.50 秒）
   - INT8+H2O 组合达到最优的性能-内存平衡

3. **使用场景建议**
   - **单批次推理**（batch_size=1）：优先选用 INT8 Full，权衡内存与精度
   - **批量推理**（batch_size≥16）或**长序列生成**：强烈推荐 INT8+H2O 方案
   - **实时部署**（内存约束）：H2O 在大批量或长序列场景下是最优选择

