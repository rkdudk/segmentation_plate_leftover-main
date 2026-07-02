import argparse
import time
import numpy as np
from PIL import Image
import torch
from mmseg.apis import init_model, inference_model

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--image", required=True)
parser.add_argument("--device", default="cuda:0")
args = parser.parse_args()

print("[INFO] loading model...")
model = init_model(args.config, args.checkpoint, device=args.device)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

base_alloc = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
base_reserved = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0

img = np.array(Image.open(args.image).convert("RGB"))

# warmup
print("[INFO] warmup...")
_ = inference_model(model, img)

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

print("[INFO] measuring inference...")
t0 = time.time()
_ = inference_model(model, img)

if torch.cuda.is_available():
    torch.cuda.synchronize()

dt = time.time() - t0

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    peak = torch.cuda.max_memory_allocated() / 1024 / 1024

    print(f"[RESULT] inference_time_sec={dt:.3f}")
    print(f"[RESULT] base_allocated_after_load_MB={base_alloc:.1f}")
    print(f"[RESULT] base_reserved_after_load_MB={base_reserved:.1f}")
    print(f"[RESULT] allocated_after_infer_MB={alloc:.1f}")
    print(f"[RESULT] reserved_after_infer_MB={reserved:.1f}")
    print(f"[RESULT] peak_allocated_during_infer_MB={peak:.1f}")
else:
    print(f"[RESULT] inference_time_sec={dt:.3f}")
    print("[RESULT] CUDA not available")
