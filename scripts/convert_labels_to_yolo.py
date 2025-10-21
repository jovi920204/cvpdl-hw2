#!/usr/bin/env python3
"""Convert label files in dataset_yolo from CSV absolute coordinates to YOLO normalized format.

Assumptions:
- Label lines currently look like: class,x_center,y_center,width,height (comma separated) with absolute pixel values.
- Images are in dataset_yolo/images/{train,val} and share same filename (but with image ext like .png).
- If an image can't be opened, default image size of 1920x1080 is used.

This script will create a .bak for each modified label (only once) and overwrite the original with normalized YOLO format.
"""
import sys
from pathlib import Path
import shutil
try:
    from PIL import Image
except Exception:
    Image = None


def find_image_for_label(label_path: Path, images_dirs):
    stem = label_path.stem
    for d in images_dirs:
        for ext in ('.png', '.jpg', '.jpeg'):
            p = d / (stem + ext)
            if p.exists():
                return p
    return None


def convert_file(label_path: Path, img_path: Path, default_size=(1920,1080)):
    text = label_path.read_text(encoding='utf-8').strip()
    if not text:
        return False, 'empty'

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # detect CSV style by presence of comma
    if not any(',' in l for l in lines):
        return False, 'no-commas'

    if Image is not None and img_path and img_path.exists():
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception:
            w, h = default_size
    else:
        w, h = default_size

    out_lines = []
    for l in lines:
        parts = [p.strip() for p in l.split(',') if p.strip()]
        if len(parts) < 5:
            return False, 'bad-line'
        cls = parts[0]
        try:
            cx = float(parts[1])
            cy = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except Exception:
            return False, 'parse-float'

        x = cx / w
        y = cy / h
        ww = bw / w
        hh = bh / h
        out_lines.append(f"{int(float(cls))} {x:.6f} {y:.6f} {ww:.6f} {hh:.6f}")

    # backup
    bak = label_path.with_suffix(label_path.suffix + '.bak')
    if not bak.exists():
        shutil.copy2(label_path, bak)

    label_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    return True, 'converted'


def main():
    root = Path.cwd()
    ds = 'C:\\Users\\jovi9\\Desktop\\cvpdl-hw2\\dataset_yolo'
    print(ds)
    if not Path(ds).exists():
        print('dataset_yolo not found in current working directory')
        sys.exit(1)

    images_dirs = [Path(ds) / 'images' / 'train', Path(ds) / 'images' / 'val']
    labels_dirs = [Path(ds) / 'labels' / 'train', Path(ds) / 'labels' / 'val']

    stats = {'converted':0, 'skipped':0, 'errors':0}
    for ld in labels_dirs:
        if not ld.exists():
            continue
        for f in sorted(ld.glob('*.txt')):
            img = find_image_for_label(f, images_dirs)
            ok, reason = convert_file(f, img)
            if ok:
                stats['converted'] += 1
            else:
                stats['skipped'] += 1
                # only count as error for certain reasons
                if reason in ('bad-line','parse-float'):
                    stats['errors'] += 1

    print('Done. stats:', stats)


if __name__ == '__main__':
    main()
