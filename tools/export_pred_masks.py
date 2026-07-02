import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from mmseg.apis import init_model, inference_model


def build_overlay(img_rgb: np.ndarray, pred: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    color = np.zeros((*pred.shape, 3), dtype=np.uint8)

    # 0: plate, 1: leftover, 2: background
    color[pred == 0] = (255, 0, 0)    # plate → red
    color[pred == 1] = (0, 255, 0)    # leftover → green
    color[pred == 2] = (0, 0, 255)    # background → blue

    overlay = (img_rgb * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--out-mask-dir", required=True)
    parser.add_argument("--out-overlay-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--opacity", type=float, default=0.4)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_mask_dir = Path(args.out_mask_dir)
    out_overlay_dir = Path(args.out_overlay_dir)

    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device=args.device)

    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for img_path in sorted(src_dir.iterdir()):
        if img_path.suffix.lower() not in valid_exts:
            continue

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        result = inference_model(model, img_np)
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

        mask_path = out_mask_dir / f"{img_path.stem}.png"
        Image.fromarray(pred).save(mask_path)

        overlay = build_overlay(img_np, pred, alpha=args.opacity)
        overlay_path = out_overlay_dir / f"{img_path.stem}.png"
        Image.fromarray(overlay).save(overlay_path)

        print(f"[OK] {img_path.name} -> {mask_path.name}, {overlay_path.name}")


if __name__ == "__main__":
    main()