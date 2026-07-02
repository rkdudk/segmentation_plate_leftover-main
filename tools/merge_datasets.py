#!/usr/bin/env python3
"""Merge ADE20K and FoodSeg103 3-class datasets into a unified train/val set."""

import argparse
import os
import shutil

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='Merge food/dish datasets')
    parser.add_argument(
        '--ade-root',
        default='data/ade20k_3class',
        help='Root directory of filtered ADE20K dataset (3-class)')
    parser.add_argument(
        '--food-root',
        default='data/foodseg103_3class',
        help='Root directory of FoodSeg103 pseudo-labeled dataset (3-class)')
    parser.add_argument(
        '--out-root',
        default='data/food_dish_final_bg',
        help='Output directory for the merged dataset')
    return parser.parse_args()


def copy_dataset(src_img, src_ann, dst_img, dst_ann, prefix, desc):
    files = [f for f in os.listdir(src_img) if f.lower().endswith('.jpg')]
    for img_file in tqdm(files, desc=desc):
        ann_file = img_file.replace('.jpg', '.png')
        shutil.copy(
            os.path.join(src_img, img_file),
            os.path.join(dst_img, f'{prefix}_{img_file}'))
        shutil.copy(
            os.path.join(src_ann, ann_file),
            os.path.join(dst_ann, f'{prefix}_{ann_file}'))
    return len(files)


def merge(args):
    split_mapping = {
        'train': ('training', 'train'),
        'val': ('validation', 'val'),
    }
    os.makedirs(args.out_root, exist_ok=True)

    for final_split, (ade_split, food_split) in split_mapping.items():
        print(f'\n=== Building {final_split} split ===')
        dst_img = os.path.join(args.out_root, 'images', final_split)
        dst_ann = os.path.join(args.out_root, 'annotations', final_split)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_ann, exist_ok=True)

        total = 0

        ade_img = os.path.join(args.ade_root, 'images', ade_split)
        ade_ann = os.path.join(args.ade_root, 'annotations', ade_split)
        if os.path.isdir(ade_img) and os.path.isdir(ade_ann):
            total += copy_dataset(ade_img, ade_ann, dst_img, dst_ann,
                                  'ade', f'ADE20K->{final_split}')
        else:
            print(f'  ⚠️ ADE split missing: {ade_img}')

        food_img = os.path.join(args.food_root, 'images', food_split)
        food_ann = os.path.join(args.food_root, 'annotations', food_split)
        if os.path.isdir(food_img) and os.path.isdir(food_ann):
            total += copy_dataset(food_img, food_ann, dst_img, dst_ann,
                                  'food', f'FoodSeg->{final_split}')
        else:
            print(f'  ⚠️ FoodSeg split missing: {food_img}')

        print(f'  ✅ {final_split} total images: {total}')

    print(f'\nMerged dataset saved to: {args.out_root}')
    for split in ['train', 'val']:
        dir_path = os.path.join(args.out_root, 'images', split)
        if os.path.isdir(dir_path):
            print(f'  {split}: {len(os.listdir(dir_path))} images')


def main():
    args = parse_args()
    merge(args)


if __name__ == '__main__':
    main()
