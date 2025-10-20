#!/usr/bin/env python3
"""
Convert dataset (image + per-image .txt annotations) to COCO format.

Assumed input annotation format (default):
  each line: class,x,y,w,h
  where x,y,w,h are in absolute pixels and x,y are box center coordinates.

Supports other formats via --input-format: xywh_abs_topleft or yolo (normalized cx,cy,w,h).

Generates annotations_{split}.json under output dir.
"""
import argparse
import json
import os
from PIL import Image


def iter_image_files(folder):
    exts = {'.png', '.jpg', '.jpeg', '.bmp'}
    for fn in sorted(os.listdir(folder)):
        name, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            yield fn


def parse_annotation_line(line, fmt, img_w, img_h):
    # split by comma or whitespace
    line = line.strip()
    if not line:
        return None
    if ',' in line:
        parts = [p.strip() for p in line.split(',') if p.strip()!='']
    else:
        parts = [p for p in line.split() if p!='']
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    x = float(parts[1])
    y = float(parts[2])
    w = float(parts[3])
    h = float(parts[4])

    if fmt == 'yolo':
        # x,y,w,h are relative to image size, centered
        cx = x * img_w
        cy = y * img_h
        bw = w * img_w
        bh = h * img_h
        xmin = cx - bw / 2.0
        ymin = cy - bh / 2.0
    elif fmt == 'xywh_abs_topleft':
        xmin = x
        ymin = y
        bw = w
        bh = h
    else:  # xywh_abs_center
        cx = x
        cy = y
        bw = w
        bh = h
        xmin = cx - bw / 2.0
        ymin = cy - bh / 2.0

    # clip
    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    bw = max(0.0, bw)
    bh = max(0.0, bh)
    if xmin + bw > img_w:
        bw = max(0.0, img_w - xmin)
    if ymin + bh > img_h:
        bh = max(0.0, img_h - ymin)

    return {'category': cls, 'bbox': [xmin, ymin, bw, bh], 'area': bw * bh}


def build_coco_from_list(images_list, images_dir, input_format='xywh_abs_center', start_img_id=1, start_ann_id=1):
    images = []
    annotations = []
    categories_set = set()
    ann_id = start_ann_id
    img_id = start_img_id

    for fn in images_list:
        img_path = os.path.join(images_dir, fn)
        try:
            im = Image.open(img_path)
            w, h = im.size
        except Exception:
            print('WARN: cannot open', img_path)
            continue

        images.append({'id': img_id, 'file_name': fn, 'width': w, 'height': h})

        txt_path = os.path.join(images_dir, os.path.splitext(fn)[0] + '.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parsed = parse_annotation_line(line, input_format, w, h)
                    if parsed is None:
                        continue
                    cat = int(parsed['category'])
                    categories_set.add(cat)
                    xmin, ymin, bw, bh = parsed['bbox']
                    if bw <= 0 or bh <= 0:
                        continue
                    ann = {
                        'id': ann_id,
                        'image_id': img_id,
                        'category_id': cat + 1,  # make categories 1-based
                        'bbox': [round(xmin, 2), round(ymin, 2), round(bw, 2), round(bh, 2)],
                        'area': round(parsed['area'], 2),
                        'iscrowd': 0,
                        'segmentation': []
                    }
                    annotations.append(ann)
                    ann_id += 1

        img_id += 1

    return images, annotations, categories_set, img_id, ann_id


def convert_with_val_split(images_dir, annotations_out_dir, input_format='xywh_abs_center', val_split=0.0, seed=42):
    # gather image filenames
    imgs = list(iter_image_files(images_dir))
    import random
    rng = random.Random(seed)
    n = len(imgs)
    n_val = int(round(n * val_split)) if val_split > 0 else 0
    imgs_shuf = imgs.copy()
    rng.shuffle(imgs_shuf)
    val_imgs = set(imgs_shuf[:n_val])
    train_imgs = [x for x in imgs if x not in val_imgs]
    val_imgs_list = [x for x in imgs if x in val_imgs]

    # build train
    train_images, train_annotations, train_categories_set, next_img_id, next_ann_id = build_coco_from_list(train_imgs, images_dir, input_format, start_img_id=1, start_ann_id=1)

    # build val (image ids continue after train)
    val_images, val_annotations, val_categories_set, _, _ = build_coco_from_list(val_imgs_list, images_dir, input_format, start_img_id=next_img_id, start_ann_id=next_ann_id)

    # unify categories
    all_cats = sorted(train_categories_set.union(val_categories_set))
    categories = [{'id': cid + 1, 'name': f'class_{cid}'} for cid in all_cats]

    os.makedirs(annotations_out_dir, exist_ok=True)
    train_out = os.path.join(annotations_out_dir, 'annotations_train.json')
    val_out = os.path.join(annotations_out_dir, 'annotations_val.json')

    with open(train_out, 'w') as f:
        json.dump({'images': train_images, 'annotations': train_annotations, 'categories': categories}, f, indent=2)
    with open(val_out, 'w') as f:
        json.dump({'images': val_images, 'annotations': val_annotations, 'categories': categories}, f, indent=2)

    print('Wrote', train_out, 'images:', len(train_images), 'annotations:', len(train_annotations), 'categories:', len(categories))
    print('Wrote', val_out, 'images:', len(val_images), 'annotations:', len(val_annotations), 'categories:', len(categories))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', default=os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    p.add_argument('--output-dir', default=None)
    p.add_argument('--input-format', choices=['xywh_abs_center', 'xywh_abs_topleft', 'yolo'], default='xywh_abs_center')
    p.add_argument('--val-split', type=float, default=0.0, help='fraction of train images to use as validation (0..1)')
    p.add_argument('--seed', type=int, default=42, help='random seed for val split')
    args = p.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        out_dir = os.path.join(dataset_dir, 'annotations')

    for split in ['train', 'test']:
        images_dir = os.path.join(dataset_dir, split)
        if not os.path.isdir(images_dir):
            print('skip missing split', split)
            continue
        if split == 'train' and args.val_split and args.val_split > 0.0:
            convert_with_val_split(images_dir, out_dir, args.input_format, val_split=args.val_split, seed=args.seed)
        else:
            out_json = os.path.join(out_dir, f'annotations_{split}.json')
            # use existing single-split builder
            images = []
            annotations = []
            categories_set = set()
            img_id = 1
            ann_id = 1
            imgs = list(iter_image_files(images_dir))
            imgs.sort()
            images, annotations, categories_set, _, _ = build_coco_from_list(imgs, images_dir, args.input_format, start_img_id=1, start_ann_id=1)
            categories = [{'id': cid + 1, 'name': f'class_{cid}'} for cid in sorted(categories_set)]
            os.makedirs(out_dir, exist_ok=True)
            with open(out_json, 'w') as f:
                json.dump({'images': images, 'annotations': annotations, 'categories': categories}, f, indent=2)
            print('Wrote', out_json, 'images:', len(images), 'annotations:', len(annotations), 'categories:', len(categories))


if __name__ == '__main__':
    main()
