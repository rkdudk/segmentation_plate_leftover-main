from pathlib import Path
import numpy as np
from PIL import Image

mask_dir = Path("finetuning/correction_set/masks")
vis_dir = Path("finetuning/correction_set/mask_vis")
vis_dir.mkdir(exist_ok=True)

palette = {
    0: [255,0,0],   # plate → 빨강
    1: [0,255,0],   # leftover → 초록
    2: [0,0,255],   # bg → 파랑
}

for p in mask_dir.glob("*.png"):
    m = np.array(Image.open(p))
    color = np.zeros((*m.shape,3), dtype=np.uint8)

    for k,v in palette.items():
        color[m==k] = v

    Image.fromarray(color).save(vis_dir/p.name)

print("done")