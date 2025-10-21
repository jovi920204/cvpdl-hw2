#!/usr/bin/env python3
"""Create a YOLO-style dataset directory `dataset_yolo` from the existing `dataset`.

Behaviour:
- Reads images and label files from `dataset/train` (expects images like imgXXXX.png and corresponding imgXXXX.txt)
- Splits them into train/val according to `--val-fraction` (default 0.2) and random `--seed`.
- Copies image files into `dataset_yolo/images/train` and `dataset_yolo/images/val` and label files into `dataset_yolo/labels/train` and `dataset_yolo/labels/val`.
- Writes `dataset_yolo.yaml` at project root with paths and class names extracted from COCO-style annotations if present, otherwise from a provided --names argument.

Usage:
    python scripts/create_dataset_yolo.py --val-fraction 0.2 --seed 42
"""
import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List


def find_classes_from_coco(ann_path: Path) -> List[str]:
    if not ann_path.exists():
        return []
    try:
        data = json.loads(ann_path.read_text(encoding='utf-8'))
        cats = data.get('categories') or []
        names = [c.get('name') for c in sorted(cats, key=lambda x: x.get('id', 0))]
        return [n for n in names if n is not None]
    except Exception:
        return []


def write_yaml(root: Path, train_p: str, val_p: str, test_p: str, names: List[str]):
    yaml_path = root / 'dataset_yolo.yaml'
    content_lines = []
    content_lines.append(f"train: {train_p}")
    content_lines.append(f"val: {val_p}")
    content_lines.append(f"test: {test_p}")
    content_lines.append("")
    content_lines.append(f"nc: {len(names)}")
    # write names as a Python list-like representation
    names_repr = '[' + ', '.join([f"'{n}'" for n in names]) + ']' if names else '[]'
    content_lines.append(f"names: {names_repr}")
    yaml_path.write_text('\n'.join(content_lines), encoding='utf-8')
    return yaml_path


def gather_image_label_pairs(src_dir: Path):
    # expects image files and .txt labels in same folder
    images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    pairs = []
    for img in images:
        label = img.with_suffix('.txt')
        pairs.append((img, label if label.exists() else None))
    return pairs


def copy_pairs(pairs, dest_images: Path, dest_labels: Path):
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    for img, label in pairs:
        shutil.copy2(img, dest_images / img.name)
        if label and label.exists():
            shutil.copy2(label, dest_labels / label.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--val-fraction', type=float, default=0.2, help='Fraction of images to use for validation')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--dataset-dir', type=str, default='../dataset', help='Source dataset directory (root)')
    p.add_argument('--out-dir', type=str, default='../dataset_yolo', help='Output dataset directory')
    p.add_argument('--ann-train', type=str, default='dataset/annotations/annotations_train.json', help='COCO train annotation (optional)')
    p.add_argument('--ann-val', type=str, default='dataset/annotations/annotations_val.json', help='COCO val annotation (optional)')
    p.add_argument('--names', type=str, nargs='*', help='Optional override for class names')
    args = p.parse_args()
    
    root = Path.cwd()
    src_root = root / args.dataset_dir
    out_root = root / args.out_dir

    src_train_dir = src_root / 'train'
    if not src_train_dir.exists():
        print(f"Source train directory not found: {src_train_dir}")
        return

    pairs = gather_image_label_pairs(src_train_dir)
    if not pairs:
        print(f"No images found in {src_train_dir}")
        return

    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    # copy files
    copy_pairs(train_pairs, out_root / 'images' / 'train', out_root / 'labels' / 'train')
    copy_pairs(val_pairs, out_root / 'images' / 'val', out_root / 'labels' / 'val')

    # determine names
    names = []
    if args.names:
        names = args.names
    else:
        # try read from COCO ann (train then val)
        names = find_classes_from_coco(Path(args.ann_train))
        if not names:
            names = find_classes_from_coco(Path(args.ann_val))

    if not names:
        print('Warning: no class names found. dataset_yolo.yaml will contain empty names. Use --names to override.')

    # write yaml
    rel_train = str((out_root / 'images' / 'train').relative_to(root))
    rel_val = str((out_root / 'images' / 'val').relative_to(root))
    rel_test = 'dataset/test' if (root / 'dataset' / 'test').exists() else ''
    yaml_path = write_yaml(root, rel_train, rel_val, rel_test, names)

    print(f'Created {out_root} with {len(train_pairs)} train and {len(val_pairs)} val images')
    print(f'Wrote YAML: {yaml_path}')


if __name__ == '__main__':
    main()
