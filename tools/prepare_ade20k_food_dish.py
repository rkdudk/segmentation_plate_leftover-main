#!/usr/bin/env python3
"""Filter ADE20K annotations and remap to a 3-class food/dish/background dataset."""

import argparse
import os
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm


RAW_DISH_CLASS_IDS = {138, 143}  # tray, plate
RAW_LEFTOVER_CLASS_IDS = {121}  # food, solid food
CLASS_LABELS = {'dishes': 0, 'leftovers': 1, 'background': 2}


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare ADE20K 3-class dataset')
    parser.add_argument(
        '--src-root',
        default='data/ade20k/ADEChallengeData2016',
        help='Path to ADE20K root directory')
    parser.add_argument(
        '--dst-root',
        default='data/ade20k_3class',
        help='Output directory for the filtered dataset')
    return parser.parse_args()


def process_split(src_root, dst_root, split):
    img_src = os.path.join(src_root, 'images', split)
    ann_src = os.path.join(src_root, 'annotations', split)
    img_dst = os.path.join(dst_root, 'images', split)
    ann_dst = os.path.join(dst_root, 'annotations', split)
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(ann_dst, exist_ok=True)

    ann_files = sorted(f for f in os.listdir(ann_src) if f.endswith('.png'))
    stats = defaultdict(int)

    for ann_file in tqdm(ann_files, desc=f'{split}'):
        ann_path = os.path.join(ann_src, ann_file)
        ann = np.array(Image.open(ann_path))
        unique_classes = set(np.unique(ann).tolist())

        has_food = bool(unique_classes.intersection(RAW_LEFTOVER_CLASS_IDS))
        has_dish = bool(unique_classes.intersection(RAW_DISH_CLASS_IDS))

        if not (has_food or has_dish):
            continue

        new_ann = np.full_like(ann, CLASS_LABELS['background'], dtype=np.uint8)
        for cls_id in RAW_DISH_CLASS_IDS:
            new_ann[ann == cls_id] = CLASS_LABELS['dishes']
        for cls_id in RAW_LEFTOVER_CLASS_IDS:
            new_ann[ann == cls_id] = CLASS_LABELS['leftovers']

        dish_pixels = int((new_ann == CLASS_LABELS['dishes']).sum())
        food_pixels = int((new_ann == CLASS_LABELS['leftovers']).sum())
        if dish_pixels == 0 and food_pixels == 0:
            continue

        img_file = ann_file.replace('.png', '.jpg')
        src_img = os.path.join(img_src, img_file)
        if not os.path.exists(src_img):
            continue

        dst_img = os.path.join(img_dst, img_file)
        dst_ann = os.path.join(ann_dst, ann_file)
        shutil.copy(src_img, dst_img)
        Image.fromarray(new_ann).save(dst_ann)

        stats['kept'] += 1
        stats['dish_pixels'] += dish_pixels
        stats['food_pixels'] += food_pixels
        stats['background_pixels'] += int(
            (new_ann == CLASS_LABELS['background']).sum())
        stats['has_food'] += int(has_food)
        stats['has_dish'] += int(has_dish)
        stats['has_both'] += int(has_food and has_dish)

    if stats['kept']:
        avg_dish = stats['dish_pixels'] // max(stats['has_dish'], 1)
        avg_food = stats['food_pixels'] // max(stats['has_food'], 1)
    else:
        avg_dish = avg_food = 0

    print(f'\n[{split}] kept {stats["kept"]} / {len(ann_files)} annotations')
    print(f'  Images with food     : {stats["has_food"]}')
    print(f'  Images with dishes   : {stats["has_dish"]}')
    print(f'  Images with both     : {stats["has_both"]}')
    print(f'  Avg dish pixels      : {avg_dish}')
    print(f'  Avg leftover pixels  : {avg_food}')
    print(f'  Background pixels    : {stats["background_pixels"]}')


def main():
    args = parse_args()
    for split in ['training', 'validation']:
        process_split(args.src_root, args.dst_root, split)
    print(f'\nFiltered dataset saved to: {args.dst_root}')


if __name__ == '__main__':
    main()
