#!/usr/bin/env python3
"""Extract cleaned FoodSeg image IDs from visualization folder."""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Collect food image IDs from *_overlay.jpg files.')
    parser.add_argument('--vis-dir', required=True, help='Visualization root')
    parser.add_argument('--output', required=True, help='Path to keep-list txt')
    return parser.parse_args()


def main():
    args = parse_args()
    vis_root = Path(args.vis_dir)
    entries = []

    for split in ['train', 'val']:
        split_dir = vis_root / split
        if not split_dir.is_dir():
            continue
        for fname in split_dir.iterdir():
            name = fname.name
            if name.startswith('food_') and name.endswith('_overlay.jpg'):
                img_id = name.replace('food_', '').replace('_overlay.jpg', '')
                entries.append((split, img_id))

    # remove duplicates while preserving split info
    unique_entries = sorted({(split, img_id)
                             for split, img_id in entries},
                            key=lambda x: (x[0], int(x[1])))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for split, img_id in unique_entries:
            f.write(f'{split} {img_id}\n')

    print(f'Collected {len(unique_entries)} entries -> {output_path}')


if __name__ == '__main__':
    main()
