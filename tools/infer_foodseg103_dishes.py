#!/usr/bin/env python3
"""Generate pseudo 3-class labels (dish/leftover/background) for FoodSeg103."""

import argparse
import os
import shutil

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_AFFINITY', 'disabled')
os.environ.setdefault('KMP_INIT_AT_FORK', 'FALSE')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
from PIL import Image
from tqdm import tqdm

from mmseg.apis import inference_model, init_model


DEFAULT_CONFIG = ('configs/mask2former/'
                  'mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py')
DEFAULT_CHECKPOINT = ('mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_'
                      '20221203_235933-7120c214.pth')
# reduce_zero_label=True -> class index = raw_id - 1
RAW_DISH_IDS = [138, 143]  # tray, plate
DISH_CLASSES = [cls_id - 1 for cls_id in RAW_DISH_IDS]
CLASS_LABELS = {'dishes': 0, 'leftovers': 1, 'background': 2}


def parse_args():
    parser = argparse.ArgumentParser(description='Infer dish labels for FoodSeg103')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Config path')
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT, help='Checkpoint')
    parser.add_argument(
        '--data-root',
        default='data/FoodSeg103',
        help='FoodSeg103 root directory')
    parser.add_argument(
        '--out-root',
        default='data/foodseg103_3class',
        help='Output directory for 3-class dataset')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='Optional limit on the number of images per split (for debugging)')
    parser.add_argument(
        '--keep-list',
        default=None,
        help='Text file containing Food image IDs to keep (one per line)')
    return parser.parse_args()


def load_keep_ids(path):
    if path is None:
        return None
    with open(path) as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f'Loaded {len(ids)} keep IDs from {path}')
    return ids


def generate_split(model, data_root, out_root, split, keep_ids=None, max_images=None):
    img_src = os.path.join(data_root, 'images', split)
    ann_src = os.path.join(data_root, 'annotations', split)
    img_dst = os.path.join(out_root, 'images', split)
    ann_dst = os.path.join(out_root, 'annotations', split)
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(ann_dst, exist_ok=True)

    img_files = sorted(f for f in os.listdir(img_src) if f.lower().endswith('.jpg'))
    if keep_ids:
        img_files = [
            f for f in img_files if f.replace('.jpg', '') in keep_ids]
    if max_images is not None:
        img_files = img_files[:max_images]
    dish_pixels = 0
    food_pixels = 0
    background_pixels = 0

    for img_file in tqdm(img_files, desc=f'{split}'):
        img_path = os.path.join(img_src, img_file)
        ann_file = img_file.replace('.jpg', '.png')
        ann_path = os.path.join(ann_src, ann_file)
        if not os.path.exists(ann_path):
            continue

        result = inference_model(model, img_path)
        pred_mask = result.pred_sem_seg.data.squeeze().cpu().numpy()

        food_mask = np.array(Image.open(ann_path))
        new_ann = np.full_like(food_mask, CLASS_LABELS['background'], dtype=np.uint8)

        for cls_id in DISH_CLASSES:
            new_ann[pred_mask == cls_id] = CLASS_LABELS['dishes']

        new_ann[food_mask > 0] = CLASS_LABELS['leftovers']

        if not np.any((new_ann == CLASS_LABELS['dishes']) |
                      (new_ann == CLASS_LABELS['leftovers'])):
            continue

        shutil.copy(img_path, os.path.join(img_dst, img_file))
        Image.fromarray(new_ann).save(os.path.join(ann_dst, ann_file))

        dish_pixels += int((new_ann == CLASS_LABELS['dishes']).sum())
        food_pixels += int((new_ann == CLASS_LABELS['leftovers']).sum())
        background_pixels += int((new_ann == CLASS_LABELS['background']).sum())

    count = len(img_files)
    if count:
        avg_dish = dish_pixels // count
        avg_food = food_pixels // count
        avg_bg = background_pixels // count
    else:
        avg_dish = avg_food = avg_bg = 0

    print(f'\n[{split}] images processed: {count}')
    print(f'  Avg dish pixels : {avg_dish}')
    print(f'  Avg food pixels : {avg_food}')
    print(f'  Avg background : {avg_bg}')


def main():
    args = parse_args()
    keep_ids = load_keep_ids(args.keep_list)
    model = init_model(args.config, args.checkpoint, device=args.device)
    for split in ['train', 'validation']:
        generate_split(
            model,
            args.data_root,
            args.out_root,
            split,
            keep_ids=keep_ids,
            max_images=args.max_images)
    print(f'\nPseudo labels saved to: {args.out_root}')


if __name__ == '__main__':
    main()
