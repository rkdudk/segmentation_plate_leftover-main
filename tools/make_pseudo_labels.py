import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # segmentation_plate_leftover-main 루트
from mmseg.apis import init_model, inference_model

REPO_ROOT = "models/segmentation_plate_leftover-main"  
CONFIG = f"{REPO_ROOT}/configs/mask2former/mask2former_swin-s_food-dish.py"
CKPT = "models/best_mIoU_iter_12000.pth"              

IMG_DIR = "data/food_dish_finetune/images/train"
OUT_DIR = "data/food_dish_finetune/annotations/train"

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_model(CONFIG, CKPT, device=device)

    for p in Path(IMG_DIR).glob("*"):
        img = Image.open(p).convert("RGB")
        img_np = np.array(img)
        result = inference_model(model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)  # 0/1/2

        Image.fromarray(pred).save(Path(OUT_DIR) / (p.stem + ".png"))
        print("saved", p.name)

if __name__ == "__main__":
    main()