from pathlib import Path
import numpy as np
from PIL import Image

color_dir = Path("finetuning/correction_set/masks_color")
out_dir = Path("finetuning/correction_set/masks_corrected")
out_dir.mkdir(parents=True, exist_ok=True)

for p in sorted(color_dir.glob("*.png")):
    img = np.array(Image.open(p).convert("RGB"))

    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # red = plate = 0
    red = (img[:, :, 0] > 200) & (img[:, :, 1] < 80) & (img[:, :, 2] < 80)
    # green = leftover = 1
    green = (img[:, :, 0] < 80) & (img[:, :, 1] > 200) & (img[:, :, 2] < 80)
    # blue = background = 2
    blue = (img[:, :, 0] < 80) & (img[:, :, 1] < 80) & (img[:, :, 2] > 200)

    mask[red] = 0
    mask[green] = 1
    mask[blue] = 2

    # 혹시 애매한 색이 있으면 가장 가까운 색으로 보정
    known = red | green | blue
    if not known.all():
        rgb = img.astype(np.int16)
        d_red = ((rgb - np.array([255, 0, 0])) ** 2).sum(axis=2)
        d_green = ((rgb - np.array([0, 255, 0])) ** 2).sum(axis=2)
        d_blue = ((rgb - np.array([0, 0, 255])) ** 2).sum(axis=2)
        nearest = np.argmin(np.stack([d_red, d_green, d_blue], axis=0), axis=0)
        mask[~known] = nearest[~known].astype(np.uint8)

    Image.fromarray(mask).save(out_dir / p.name)
    vals, counts = np.unique(mask, return_counts=True)
    print(p.name, dict(zip(vals.tolist(), counts.tolist())))

print("saved to:", out_dir)
