#!/usr/bin/env python3
"""Simple side-by-side inference visualization."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from mmseg.apis import init_model, inference_model
from mmseg.apis import show_result_pyplot


def parse_args():
    parser = argparse.ArgumentParser(description='Compare inference outputs')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--src-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--opacity', type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device=args.device)

    for img_path in sorted(src_dir.glob('*')):
        if img_path.suffix.lower() not in ('.jpg', '.png', '.jpeg', '.bmp'):
            continue
        result = inference_model(model, str(img_path))
        drawn = show_result_pyplot(
            model, str(img_path), result,
            opacity=args.opacity, show=False)

        orig = Image.open(img_path).convert('RGB')

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.title('Original')
        plt.imshow(orig)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Prediction')
        plt.imshow(drawn)
        plt.axis('off')

        plt.tight_layout()
        out_file = out_dir / f'{img_path.stem}_compare.jpg'
        plt.savefig(out_file, dpi=150)
        plt.close()
        print(f'Saved {out_file}')


if __name__ == '__main__':
    main()
