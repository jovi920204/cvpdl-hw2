#!/usr/bin/env python3
"""Convert CSV labels (class, x_min, y_min, w, h in pixels) to YOLO normalized (class cx cy w h).

Assumptions:
- Input label line format (comma-separated):
  <class label>,<Top-left X>,<Top-left Y>,<Bounding box width>,<Bounding box height>
- Images are located under images/{train,val} corresponding to labels/{train,val}.
- If an image can't be opened, a default image size of 1920x1080 is used.

Behavior:
- For each label .txt, a one-time .bak backup is created before overwriting.
- Output lines are: "class cx cy w h" with values normalized to [0,1] in YOLO order.
"""
import sys
from pathlib import Path
import shutil
import argparse
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


def convert_file(label_path: Path, img_path: Path, default_size=(1920, 1080)):
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
            x_min = float(parts[1])
            y_min = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except Exception:
            return False, 'parse-float'

        # Guard against invalid sizes
        if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
            # skip invalid box
            continue

        # Convert from top-left (x_min, y_min, w, h) to center-based YOLO format
        cx_abs = x_min + bw / 2.0
        cy_abs = y_min + bh / 2.0

        cx = cx_abs / w
        cy = cy_abs / h
        ww = bw / w
        hh = bh / h

        # Clamp to [0,1] to be safe if boxes slightly exceed image bounds
        def clamp01(v):
            return 0.0 if v < 0 else (1.0 if v > 1 else v)

        cx = clamp01(cx)
        cy = clamp01(cy)
        ww = clamp01(ww)
        hh = clamp01(hh)

        out_lines.append(f"{int(float(cls))} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

    # backup
    bak = label_path.with_suffix(label_path.suffix + '.bak')
    if not bak.exists():
        shutil.copy2(label_path, bak)

    label_path.write_text('\n'.join(out_lines) + '\n', encoding='utf-8')
    return True, 'converted'


def main():
    parser = argparse.ArgumentParser(description='Convert CSV labels (class,x_min,y_min,w,h) to YOLO normalized format.')
    parser.add_argument('--dataset', '-d', type=str, default='../dataset_yolo',
                        help='Path to dataset root containing images/ and labels/ (default: ../dataset_yolo)')
    parser.add_argument('--splits', type=str, default='train,val',
                        help='Comma-separated label/image splits to process (default: train,val)')
    parser.add_argument('--default-size', type=str, default='1920x1080',
                        help='Fallback image size WxH if image not found/openable (default: 1920x1080)')
    args = parser.parse_args()

    ds = Path(args.dataset)
    if not ds.exists():
        print(f'Dataset not found: {ds.resolve()}')
        sys.exit(1)

    try:
        dw, dh = [int(x) for x in args.default_size.lower().split('x')]
    except Exception:
        dw, dh = 1920, 1080

    split_list = [s.strip() for s in args.splits.split(',') if s.strip()]
    images_dirs = [ds / 'images' / s for s in split_list]
    labels_dirs = [ds / 'labels' / s for s in split_list]

    stats = {'converted': 0, 'skipped': 0, 'errors': 0}
    for ld in labels_dirs:
        if not ld.exists():
            continue
        for f in sorted(ld.glob('*.txt')):
            img = find_image_for_label(f, images_dirs)
            ok, reason = convert_file(f, img, default_size=(dw, dh))
            if ok:
                stats['converted'] += 1
            else:
                stats['skipped'] += 1
                if reason in ('bad-line', 'parse-float'):
                    stats['errors'] += 1

    print('Done. stats:', stats)


if __name__ == '__main__':
    main()
