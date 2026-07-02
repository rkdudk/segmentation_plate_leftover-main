#!/usr/bin/env python3
"""Copy Food images/masks from data/food_dish_final_bg using keep list."""

import argparse
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Copy subset of food images based on keep list')
    parser.add_argument('--source', required=True, help='Source dataset root')
    parser.add_argument('--dest', required=True, help='Destination root')
    parser.add_argument('--keep-list', required=True, help='Text file with split+ID')
    return parser.parse_args()


def load_ids(path: Path):
    keep = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            split, img_id = line.split()
            keep.setdefault(split, set()).add(img_id)
    return keep


def copy_split(split: str, src_root: Path, dst_root: Path, keep_ids):
    img_src = src_root / 'images' / split
    ann_src = src_root / 'annotations' / split
    img_dst = dst_root / 'images' / split
    ann_dst = dst_root / 'annotations' / split
    img_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    for img_path in img_src.glob('food_*.jpg'):
        img_id = img_path.stem.replace('food_', '')
        if img_id not in keep_ids.get(split, set()):
            continue
        shutil.copy2(img_path, img_dst / img_path.name)
        mask_name = img_path.with_suffix('.png').name
        shutil.copy2(ann_src / mask_name, ann_dst / mask_name)


def main():
    args = parse_args()
    src = Path(args.source)
    dst = Path(args.dest)
    keep_ids = load_ids(Path(args.keep_list))
    if dst.exists():
        shutil.rmtree(dst)
    for split in ['train', 'val']:
        copy_split(split, src, dst, keep_ids)


if __name__ == '__main__':
    main()
