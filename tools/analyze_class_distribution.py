#!/usr/bin/env python3
"""Analyze class distribution for the merged food/dish dataset."""

import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = {
    0: 'dishes',
    1: 'leftovers',
    2: 'background',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze class distribution')
    parser.add_argument(
        '--data-root',
        default='data/food_dish_final_bg',
        help='Root directory of the merged dataset')
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'val'],
        help='Dataset splits to analyze')
    return parser.parse_args()


def analyze_split(ann_dir):
    stats = {
        'images': 0,
        'class_pixels': defaultdict(int),
        'class_images': defaultdict(int),
    }
    ann_files = sorted(f for f in os.listdir(ann_dir) if f.endswith('.png'))
    for ann_file in tqdm(ann_files, desc=f'Analyzing {os.path.basename(ann_dir)}'):
        ann = np.array(Image.open(os.path.join(ann_dir, ann_file)))
        stats['images'] += 1
        unique, counts = np.unique(ann, return_counts=True)
        for label, count in zip(unique, counts):
            if label == 255:
                continue
            stats['class_pixels'][label] += int(count)
            stats['class_images'][label] += 1
    return stats


def main():
    args = parse_args()
    overall_pixels = defaultdict(int)
    total_images = 0

    for split in args.splits:
        ann_dir = os.path.join(args.data_root, 'annotations', split)
        if not os.path.isdir(ann_dir):
            print(f'⚠️  Missing split: {ann_dir}')
            continue

        stats = analyze_split(ann_dir)
        total_images += stats['images']
        for label, pixels in stats['class_pixels'].items():
            overall_pixels[label] += pixels

        print(f'\n[{split}] images: {stats["images"]}')
        total_pixels = sum(stats['class_pixels'].values())
        for label, pixels in stats['class_pixels'].items():
            percent = (pixels / total_pixels * 100) if total_pixels else 0
            img_percent = (stats['class_images'][label] / stats['images'] * 100
                           if stats['images'] else 0)
            name = CLASS_NAMES.get(label, f'class_{label}')
            print(f'  {name:10s} (class {label}): '
                  f'{pixels:12,d} px ({percent:5.2f}%), '
                  f'in {stats["class_images"][label]} images ({img_percent:5.2f}%)')

    print('\n=== Overall ===')
    print(f'Total images: {total_images}')
    total_pixels = sum(overall_pixels.values())
    for label, pixels in overall_pixels.items():
        percent = (pixels / total_pixels * 100) if total_pixels else 0
        name = CLASS_NAMES.get(label, f'class_{label}')
        print(f'  {name:10s}: {pixels:12,d} px ({percent:5.2f}%)')

    if overall_pixels:
        plt.figure(figsize=(6, 4))
        labels = list(overall_pixels.keys())
        counts = [overall_pixels[l] for l in labels]
        names = [CLASS_NAMES.get(l, f'class_{l}') for l in labels]
        plt.bar(names, counts, color=['#00aa00', '#8000ff'][:len(names)])
        for idx, count in enumerate(counts):
            percent = count / sum(counts) * 100
            plt.text(idx, count, f'{percent:.1f}%', ha='center', va='bottom')
        plt.ylabel('Pixels')
        plt.title('Class distribution')
        plt.tight_layout()
        out_path = os.path.join(args.data_root, 'class_distribution.png')
        plt.savefig(out_path, dpi=150)
        print(f'\nVisualization saved to: {out_path}')


if __name__ == '__main__':
    main()
