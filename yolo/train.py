from ultralytics import YOLO
import argparse
import sys


def train_model(model, dataset_path, epochs, imgsz, device, resume=False):
    """Unified YOLO training function"""
    train_params = dict(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=16,
        patience=100,
        lr0=1e-4,
        optimizer='AdamW',
        seed=42,
        save_period=10,
        plots=True,
        resume=resume,
        project='runs/train',
    )

    print(f"{'Resuming' if resume else 'Starting'} training on {device} for {epochs} epochs ...")
    return model.train(**train_params)


def main(mode, dataset_path, epochs, imgsz, device, pretrained_model):
    if mode not in {"scratch", "resume"}:
        sys.exit("Error: mode must be 'scratch' or 'resume'")

    if mode == "scratch":
        print("Training YOLOv8 model from scratch...")
        model = YOLO("yolo11m.yaml")
        results = train_model(model, dataset_path, epochs, imgsz, device, resume=False)

    elif mode == "resume":
        if not pretrained_model:
            sys.exit("Error: Please provide a --pretrained_model when resuming training.")
        print(f"Resuming training from checkpoint: {pretrained_model}")
        model = YOLO(pretrained_model)
        results = train_model(model, dataset_path, epochs, imgsz, device, resume=True)

    print("Training finished successfully.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv8 training launcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--mode", type=str, choices=["scratch", "resume"], default="scratch",
                        help="Training mode: 'scratch' (new training) or 'resume' (continue training)")
    parser.add_argument("--dataset", type=str, default="../dataset_yolo/dataset_yolo.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Training device, e.g., 'cuda:0', 'cuda:1', or 'cpu'")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to a trained model checkpoint (required if mode=resume)")

    args = parser.parse_args()

    main(args.mode, args.dataset, args.epochs, args.imgsz, args.device, args.pretrained_model)