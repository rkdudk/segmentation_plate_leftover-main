#!/usr/bin/env python3
"""Run inference on a folder and save side-by-side comparisons."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

from mmseg.apis import init_model, inference_model


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize inference comparisons')
    parser.add_argument('--config', required=True, help='Config file')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--src-dir', required=True, help='Directory with input images')
    parser.add_argument('--out-dir', required=True, help='Directory to store comparisons')
    parser.add_argument('--device', default='cuda:0', help='Device to run inference')
    parser.add_argument('--opacity', type=float, default=0.5, help='Overlay opacity')
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device=args.device)

    for img_path in sorted(src_dir.glob('*')):
        if not img_path.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp']:
            continue
        
        # PIL로 이미지 읽기 및 EXIF orientation 자동 처리
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)  # EXIF orientation 자동 보정
        image = image.convert('RGB')
        
        # Inference 수행
        result = inference_model(model, str(img_path))
        pred = result.pred_sem_seg.data.squeeze().cpu().numpy()
        
        # 디버그: 예측된 클래스 값들 확인
        unique_classes = np.unique(pred)
        print(f'{img_path.name}: 예측된 클래스 = {unique_classes}')

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Prediction')
        plt.imshow(image)
        
        # 모든 영역 색칠 (배경 포함)
        plt.imshow(pred, alpha=args.opacity, cmap='jet', interpolation='nearest')
        plt.axis('off')

        out_path = out_dir / f'{img_path.stem}_compare.jpg'
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
