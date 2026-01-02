import os
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from kv_cache import KVCacheManager
from int8_weight_only import Int8Linear 

def load_quantized_checkpoint(ckpt_dir):
    wq = torch.load(os.path.join(ckpt_dir, "weights_int8.pt"), map_location="cpu")
    sc = torch.load(os.path.join(ckpt_dir, "scales_fp32.pt"), map_location="cpu")
    b = torch.load(os.path.join(ckpt_dir, "bias_fp16.pt"), map_location="cpu")
    return wq, sc, b


def replace_linear_with_int8(model, wq, sc, b, dtype, device):
    for name, m in list(model.named_modules()):
        if isinstance(m, torch.nn.Linear) and name in wq:
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[-1]
            parent = model.get_submodule(parent_name) if parent_name != '' else model
            new_m = Int8Linear(
                    in_features=m.in_features,  
                    out_features=m.out_features, 
                    bias=(m.bias is not None),   
                    device=device,
                    dtype=dtype
                )

            new_m.Wq = wq[name].to(device)
            new_m.scale = sc[name].to(device)
            if m.bias is not None and name in b:
                new_m.bias = b[name].to(device).to(dtype)
            setattr(parent, child_name, new_m)

def theoretical_param_bytes_fp16(model):
    total = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            total += m.weight.numel() * 2
            if m.bias is not None:
                total += m.bias.numel() * 2
    return total

def theoretical_param_bytes_int8(wq, sc, b, dtype):
    bytes_per_scale = 2 if dtype == torch.float16 else 4
    bytes_per_bias = 2 if dtype == torch.float16 else 2
    w_bytes = sum(t.numel() for t in wq.values())
    s_bytes = sum(t.numel() for t in sc.values()) * bytes_per_scale
    b_bytes = sum(t.numel() for t in b.values()) * bytes_per_bias if b is not None else 0
    return w_bytes + s_bytes + b_bytes

def get_tokenizer(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def run_mode(model_path, ckpt_dir, mode, prefill_len, decode_len, window, h2o_hh, batch_size=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    need_attn = (mode == "int8_h2o")
    attn_impl = ("eager" if need_attn else "sdpa")
    dtype = torch.bfloat16 # Changed to bfloat16 for INT8 kernel compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        device_map=device,
        local_files_only=True,
    )
    
    tok = get_tokenizer(model_path)
    wq, sc, b = load_quantized_checkpoint(ckpt_dir)
    if mode in ["int8_full", "int8_h2o"]:
        # First, find MLP blocks and replace dense_h_to_4h with fused activation
        # GPT-NeoX: layers -> mlp -> dense_h_to_4h (linear), dense_4h_to_h (linear), act (GELU)
        for name, m in list(model.named_modules()):
            if "mlp" in name and isinstance(m, torch.nn.Module):
                # Check if it has the standard GPT-NeoX MLP structure
                if hasattr(m, "dense_h_to_4h") and hasattr(m, "dense_4h_to_h") and hasattr(m, "act"):
                    # Fuse Activation into dense_h_to_4h
                    # 1. Replace dense_h_to_4h with Int8Linear(act_type="gelu")
                    linear_name = name + ".dense_h_to_4h"
                    if linear_name in wq:
                        old_linear = m.dense_h_to_4h
                        new_linear = Int8Linear(
                            in_features=old_linear.in_features,
                            out_features=old_linear.out_features,
                            bias=(old_linear.bias is not None),
                            device=device,
                            dtype=dtype,
                            act_type="gelu"
                        )
                        new_linear.Wq = wq[linear_name].to(device)
                        new_linear.scale = sc[linear_name].to(device)
                        if old_linear.bias is not None and linear_name in b:
                            new_linear.bias = b[linear_name].to(device).to(dtype)
                        
                        m.dense_h_to_4h = new_linear
                        # 2. Replace act with Identity
                        m.act = torch.nn.Identity()
                        
                        # 3. Replace dense_4h_to_h with standard Int8Linear
                        linear_name_2 = name + ".dense_4h_to_h"
                        if linear_name_2 in wq:
                            old_linear_2 = m.dense_4h_to_h
                            new_linear_2 = Int8Linear(
                                in_features=old_linear_2.in_features,
                                out_features=old_linear_2.out_features,
                                bias=(old_linear_2.bias is not None),
                                device=device,
                                dtype=dtype,
                                act_type="none"
                            )
                            new_linear_2.Wq = wq[linear_name_2].to(device)
                            new_linear_2.scale = sc[linear_name_2].to(device)
                            if old_linear_2.bias is not None and linear_name_2 in b:
                                new_linear_2.bias = b[linear_name_2].to(device).to(dtype)
                            m.dense_4h_to_h = new_linear_2
                            
        # Then replace remaining linear layers (Attention QKV, Output)
        replace_linear_with_int8(model, wq, sc, b, dtype, device)
    print(model)
    kv_mode = ("h2o" if mode == "int8_h2o" else "full")
    kv = KVCacheManager(mode=kv_mode, window_size=window, h2o_heavy_size=h2o_hh)
    

    ds = load_dataset("/home/xlyu/datasets/wikitext", split="test")
    text = "\n\n".join(ds["text"])
        
    enc = tok(text, return_tensors="pt")
    total_len = prefill_len + decode_len
    input_ids = enc.input_ids[:, :total_len].to(device)
    
    # Replicate for batch size
    input_ids = input_ids.expand(batch_size, -1)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Enable CUDA Graph for INT8 modes (linear layers only)
    # Note: We will enable it AFTER prefill
    use_cuda_graph = (mode in ["int8_full", "int8_h2o"]) and torch.cuda.is_available() and batch_size == 1
    
    lat_prefill = []
    lat_decode = []
    nlls = []
    
    # Warmup
    print("Warmup...")
    # For CUDA Graph, we need to run a few iterations to stabilize
    kv.reset()
    # Dummy run
    with torch.no_grad():
         _ = model(
            input_ids=input_ids[:, 0:1],
            past_key_values=kv.past_key_values,
            use_cache=True
         )
    kv.reset()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_total = time.time()
    
    # NVTX Range for the whole mode
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(f"RunMode_{mode}")

    # ==========================
    # 1. Prefill Phase
    # ==========================
    print(f"Running Prefill ({prefill_len} tokens)...")
    inp_prefill = input_ids[:, :prefill_len]
    
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(f"Prefill")
    
    t0 = time.time()
    with torch.no_grad():
        out = model(
            input_ids=inp_prefill,
            past_key_values=kv.past_key_values,
            use_cache=True,
            output_attentions=need_attn,
        )
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()
    lat_prefill.append((t1 - t0) * 1000)
    
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop() # Prefill
        
    kv.update(out.past_key_values, attentions=(out.attentions if need_attn else None))
    
    # ==========================
    # 2. CUDA Graph Capture (Optional)
    # ==========================
    if use_cuda_graph:
        print("Enabling CUDA Graph for Int8Linear modules (decode step shapes)...")
        
        # Find all Int8Linear modules
        modules_to_graph = []
        for m in model.modules():
            if isinstance(m, Int8Linear):
                modules_to_graph.append(m)
        
        if modules_to_graph:
            print(f"Graphing {len(modules_to_graph)} modules...")
            count = 0
            for m in modules_to_graph:
                inp_dim = m.in_features
                sample_input = torch.empty((batch_size, 1, inp_dim), dtype=dtype, device=device)
                m.enable_graph(sample_input)
                count += 1
            print(f"Graphed {count} Int8Linear modules.")

    # ==========================
    # 3. Decoding Phase
    # ==========================
    print(f"Running Decoding ({decode_len} steps)...")
    
    with torch.no_grad():
        for i in tqdm(range(decode_len)):
            
            curr_idx = prefill_len + i
            if curr_idx >= input_ids.size(1) - 1:
                break
                
            inp = input_ids[:, curr_idx : curr_idx+1]
            tgt = input_ids[:, curr_idx+1 : curr_idx+2]
            
            # Position IDs
            pos = torch.tensor([[curr_idx]], device=device).expand(batch_size, 1)
            
            # NVTX Range for single step
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(f"Step_{i}")

            t0 = time.time()
            
            out = model(
                input_ids=inp,
                past_key_values=kv.past_key_values,
                position_ids=pos,
                use_cache=True,
                output_attentions=need_attn,
                )
            torch.cuda.synchronize() if device == "cuda" else None
            t1 = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.nvtx.range_pop() # Step

            lat_decode.append((t1 - t0) * 1000)
            
            # Loss calc (sum over batch?)
            # logits: (B, 1, V)
            # tgt: (B, 1)
            loss = torch.nn.functional.cross_entropy(
                out.logits.view(-1, out.logits.size(-1)),
                tgt.reshape(-1),
                reduction='mean'
            )
            nlls.append(loss)
            attn = out.attentions if need_attn else None
            kv.update(out.past_key_values, attentions=attn)
            
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_pop() # Mode

    total_time = time.time() - start_total
    
    # Metrics
    prefill_latency_ms = sum(lat_prefill) / len(lat_prefill) if lat_prefill else 0
    prefill_tps = (batch_size * prefill_len * 1000.0) / prefill_latency_ms if prefill_latency_ms > 0 else 0
    
    decode_latency_ms = sum(lat_decode) / len(lat_decode) if lat_decode else 0
    decode_tps = (batch_size * 1000.0) / decode_latency_ms if decode_latency_ms > 0 else 0
    
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device == "cuda" else 0.0
    ppl = torch.exp(torch.stack(nlls).mean()).item() if nlls else 0.0
    
    total_tokens = batch_size * (prefill_len + decode_len)
    total_tps = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "mode": mode,
        "prefill_tps": prefill_tps,
        "prefill_lat_ms": prefill_latency_ms, # Total time for prefill batch
        "decode_tps": decode_tps,
        "decode_lat_ms": decode_latency_ms,
        "total_tps": total_tps,
        "peak_mb": peak_mb,
        "ppl": ppl,
        "total_s": total_time,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="/home/xlyu/models/models--EleutherAI--pythia-2.8b")
    ap.add_argument("--ckpt_dir", type=str, default="../checkpoints/pythia-2.8b-int8")
    ap.add_argument("--prefill_len", type=int, default=500)
    ap.add_argument("--decode_len", type=int, default=1000)
    ap.add_argument("--window", type=int, default=252)
    ap.add_argument("--h2o_hh", type=int, default=252)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()
    wq, sc, b = load_quantized_checkpoint(args.ckpt_dir)
    model_cpu = AutoModelForCausalLM.from_pretrained(
        args.model_path,
    )
    bytes_fp16 = theoretical_param_bytes_fp16(model_cpu)
    bytes_int8 = theoretical_param_bytes_int8(wq, sc, b, torch.float16)
    print("Param storage bytes FP16:", bytes_fp16)
    print("Param storage bytes INT8 weight-only:", bytes_int8)
    print("Reduction:", bytes_fp16 - bytes_int8)
    print(f"Batch Size: {args.batch_size}")
    
    print("Running FP16 baseline full-attention")
    r_fp16 = run_mode(args.model_path, args.ckpt_dir, "fp16_full", args.prefill_len, args.decode_len, args.window, args.h2o_hh, args.batch_size)
    print("Running INT8 weight-only full-attention")
    r_int8_full = run_mode(args.model_path, args.ckpt_dir, "int8_full", args.prefill_len, args.decode_len, args.window, args.h2o_hh, args.batch_size)
    print("Running INT8 weight-only + H2O KV cache")
    r_int8_h2o = run_mode(args.model_path, args.ckpt_dir, "int8_h2o", args.prefill_len, args.decode_len, args.window, args.h2o_hh, args.batch_size)
    print("==================================================")
    for r in [r_fp16, r_int8_full, r_int8_h2o]:
        print(f"Mode: {r['mode']}")
        print(f"Total Throughput: {r['total_tps']:.2f} tokens/sec")
        print(f"Prefill Throughput: {r['prefill_tps']:.2f} tokens/sec")
        print(f"Prefill Latency: {r['prefill_lat_ms']:.2f} ms")
        print(f"Decoding Throughput: {r['decode_tps']:.2f} tokens/sec")
        print(f"Decoding Latency: {r['decode_lat_ms']:.2f} ms/token")
        print(f"Peak GPU Memory: {r['peak_mb']:.2f} MB")
        print(f"PPL: {r['ppl']:.4f}")
        print(f"Total Time: {r['total_s']:.2f} s")
        print("------------------------------")

if __name__ == "__main__":
    main()
