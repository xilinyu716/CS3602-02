import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def quantize_linear_weight(W):
    Wf = W.float()
    s = torch.clamp(Wf.abs().amax(dim=1) / 127.0, min=1e-8)
    q = torch.clamp((Wf / s[:, None]).round(), -128, 127).to(torch.int8)
    return q, s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default="/home/xlyu/models/models--EleutherAI--pythia-2.8b")
    ap.add_argument("--out_dir", type=str, default="../checkpoints/pythia-2.8b-int8")
    ap.add_argument("--dtype", type=str, default="float16")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    weights_q = {}
    scales = {}
    biases = {}
    modules = list(model.named_modules())
    for name, m in tqdm(modules, desc="Quantizing", total=len(modules)):
        if isinstance(m, torch.nn.Linear):
            W = m.weight.detach().to("cpu")
            q, s = quantize_linear_weight(W)
            weights_q[name] = q
            scales[name] = s.to(torch.float32)
            if m.bias is not None:
                biases[name] = m.bias.detach().to("cpu").half()
    torch.save(weights_q, os.path.join(args.out_dir, "weights_int8.pt"))
    torch.save(scales, os.path.join(args.out_dir, "scales_fp32.pt"))
    torch.save(biases, os.path.join(args.out_dir, "bias_fp16.pt"))
    print("DONE")
    print("Linear layers quantized:", len(weights_q))
    if len(scales) > 0:
        all_s = torch.cat([v.flatten() for v in scales.values()])
        print("Scale stats:", float(all_s.min()), float(all_s.mean()), float(all_s.max()))

if __name__ == "__main__":
    main()
