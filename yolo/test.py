import os
import argparse
from datetime import datetime
from tqdm import tqdm
from ultralytics import YOLO

def chunk_list(lst, n):
    """將 list 分成大小為 n 的小批次"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def run_inference(args):
    os.makedirs(args.save_dir, exist_ok=True)

    # === 建立輸出 CSV ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_dir, f"{timestamp}_results.csv")

    # === 載入模型 ===
    print(f"[INFO] Loading model from {args.model} ...")
    model = YOLO(args.model)

    # === 取得所有圖片檔案 ===
    img_files = sorted([
        f for f in os.listdir(args.img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"[INFO] Found {len(img_files)} images in {args.img_dir}")
    print(f"[INFO] Using batch size = {args.batch_size}, imgsize = {args.imgsize}")

    with open(csv_path, "w") as fw:
        fw.write("Image_ID,PredictionString\n")

        for batch_files in tqdm(list(chunk_list(img_files, args.batch_size)), desc="Running inference", unit="batch"):
            batch_paths = [os.path.join(args.img_dir, f) for f in batch_files]

            results = model.predict(
                source=batch_paths,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsize,
                augment=args.tta,
                device=[2],
                save=args.save_img,
                project=args.save_dir if args.save_img else None,
                name="predictions",
                exist_ok=True,
                verbose=False
            )

            # YOLOv11 predict() 對 batch 輸入會回傳多個結果
            for fname, res in zip(batch_files, results):
                # fname 範例: img0123.png -> image_id = 123
                image_id = int(''.join(filter(str.isdigit, os.path.splitext(fname)[0])))
                boxes = res.boxes
                pred_str = ""
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().tolist()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        x1, y1, x2, y2 = xyxy
                        w = x2 - x1
                        h = y2 - y1
                        pred_str += f"{conf:.6f} {x1:.0f} {y1:.0f} {w:.0f} {h:.0f} {cls} "
                    pred_str = pred_str.strip()
                fw.write(f"{str(image_id)},{pred_str}\n")

    print(f"\n✅ Inference complete.")
    print(f"[INFO] CSV saved to: {csv_path}")
    if args.save_img:
        print(f"[INFO] Annotated images saved to: {os.path.join(args.save_dir, 'predictions')}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Batch Inference Script")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLOv11 model (.pt)")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--save_dir", type=str, default="results/test", help="Directory to save results")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument("--imgsize", type=int, default=1280, help="Input image size for inference")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of images per batch")
    parser.add_argument("--save_img", action="store_true", help="Save annotated images")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
