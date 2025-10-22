#!/usr/bin/env python3
"""Downsample class instances in dataset_yolo

Default behavior is a dry-run that reports how many class=0 instances exist in
`dataset_yolo/labels/train` and how many would be removed to reach the target.

Use --apply to actually create `dataset_yolo_downsampled` and perform removals.
"""
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict


def gather_class_instances(labels_dir: Path, target_class: str = '0'):
    instances = []  # list of (label_path, line_index)
    per_file_counts = {}
    label_files = sorted([p for p in labels_dir.glob('*.txt') if not p.name.endswith('.bak')])
    for p in label_files:
        lines = [l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]
        indices = [i for i,l in enumerate(lines) if l.split()[0] == target_class]
        per_file_counts[p] = len(indices)
        for i in indices:
            instances.append((p, i))
    return instances, per_file_counts, label_files


def apply_downsample(src_root: Path, out_root: Path, instances_to_remove, labels_dir: Path, images_dir: Path):
    if out_root.exists():
        raise RuntimeError(f'Output directory already exists: {out_root}')
    # create folder structure
    (out_root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (out_root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (out_root / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (out_root / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (out_root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # Copy val images/labels entirely
    src_val_images = src_root / 'images' / 'val'
    src_val_labels = src_root / 'labels' / 'val'
    if src_val_images.exists():
        for p in src_val_images.glob('*'):
            shutil.copy2(p, out_root / 'images' / 'val' / p.name)
    if src_val_labels.exists():
        for p in src_val_labels.glob('*.txt'):
            shutil.copy2(p, out_root / 'labels' / 'val' / p.name)

    # Build a mapping of removals per file: label_path -> set(line_indices)
    removals = defaultdict(set)
    for p, idx in instances_to_remove:
        removals[p].add(idx)

    # Process each train label file
    for src_label in sorted(labels_dir.glob('*.txt')):
        if src_label.name.endswith('.bak'):
            continue
        lines = [l for l in src_label.read_text(encoding='utf-8').splitlines()]
        # remove by index (indices relative to original)
        keep_lines = [l for i,l in enumerate(lines) if i not in removals.get(src_label, set()) and l.strip()]
        dest_label = out_root / 'labels' / 'train' / src_label.name
        dest_label.write_text('\n'.join(keep_lines) + ('\n' if keep_lines else ''), encoding='utf-8')
        # copy corresponding image (try common extensions)
        stem = src_label.stem
        found_img = None
        for ext in ('.png', '.jpg', '.jpeg'):
            p = images_dir / (stem + ext)
            if p.exists():
                found_img = p
                break
        if found_img:
            shutil.copy2(found_img, out_root / 'images' / 'train' / found_img.name)
        else:
            # image missing; skip copy
            pass

    # Copy test images/labels if present
    src_test_images = src_root / 'images' / 'test'
    if src_test_images.exists():
        (out_root / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        for p in src_test_images.glob('*'):
            shutil.copy2(p, out_root / 'images' / 'test' / p.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='../dataset_yolo', help='Source dataset folder')
    p.add_argument('--out', default='../dataset_yolo_downsampled', help='Output folder name')
    p.add_argument('--class-id', type=str, default='0', help='Class id to downsample (as string)')
    p.add_argument('--target', type=int, default=4000, help='Approx target number of instances for the class')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--apply', action='store_true', help='Actually create the downsampled dataset (otherwise dry-run)')
    args = p.parse_args()

    src_root = Path(args.src)
    if not src_root.exists():
        print('Source dataset not found:', src_root)
        return
    labels_dir = src_root / 'labels' / 'train'
    images_dir = src_root / 'images' / 'train'
    if not labels_dir.exists():
        print('Labels train dir not found:', labels_dir)
        return

    instances, per_file_counts, label_files = gather_class_instances(labels_dir, args.class_id)
    total = len(instances)
    print('Found', total, 'instances of class', args.class_id)

    if total <= args.target:
        print('Total less than or equal to target; nothing to remove.')
        if args.apply:
            # just copy whole dataset
            out_root = Path(args.out)
            if out_root.exists():
                print('Output exists; aborting to avoid overwrite:', out_root)
                return
            shutil.copytree(src_root, out_root)
            print('Copied dataset to', out_root)
        return

    to_remove = total - args.target
    print('Need to remove', to_remove, 'instances to reach target', args.target)

    # show top files by count
    top = sorted(per_file_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    print('\nTop files by class count:')
    for p,c in top:
        print(p.name, c)

    # Choose random instances to remove
    random.seed(args.seed)
    remove_indices = set(random.sample(range(total), to_remove))
    instances_to_remove = [instances[i] for i in sorted(remove_indices)]

    print('\nDry-run summary: will remove {} instances across {} files'.format(len(instances_to_remove), len(set(p for p,_ in instances_to_remove))))

    if args.apply:
        out_root = Path(args.out)
        print('Applying changes to', out_root)
        apply_downsample(src_root, out_root, instances_to_remove, labels_dir, images_dir)
        # write a YAML (pointing to new dataset)
        yaml_text = f"train: {out_root}/images/train\nval: {out_root}/images/val\ntest: {out_root}/images/test\n\nnc: ?\nnames: []\n"
        (out_root / 'dataset_yolo_downsampled.yaml').write_text(yaml_text)
        print('Done. Wrote dataset_yolo_downsampled and YAML.')


if __name__ == '__main__':
    main()
