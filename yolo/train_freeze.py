from ultralytics import YOLO
import argparse
import sys


def freeze_backbone_only(model):
    """Freeze YOLO backbone and keep detection head trainable.

    Strategy:
    - If model.model.yaml defines 'backbone' and 'head', freeze first len(backbone) modules.
    - Fallback: freeze all params, then unfreeze modules of class 'Detect'.
    """
    core = model.model  # ultralytics.nn.tasks.DetectionModel

    # Prefer YAML-aware split (backbone/head)
    try:
        yaml_dict = getattr(core, 'yaml', None)
        seq = getattr(core, 'model', None)  # nn.Sequential of layers
        if isinstance(yaml_dict, dict) and isinstance(seq, (list, tuple)) or hasattr(seq, '__iter__'):
            backbone_layers = yaml_dict.get('backbone', None)
            if isinstance(backbone_layers, (list, tuple)) and hasattr(seq, '__getitem__'):
                blen = len(backbone_layers)
                # Freeze backbone, unfreeze head
                for i, layer in enumerate(seq):
                    trainable = i >= blen
                    for p in layer.parameters():
                        p.requires_grad = trainable
                print(f"[freeze] Froze backbone (0..{blen-1}), head remains trainable (from {blen}).")
                return
    except Exception as e:
        print(f"[freeze] YAML-based split not available ({e}); falling back to Detect-only unfreeze.")

    # Fallback: freeze all, then unfreeze Detect head parameters
    for p in core.parameters():
        p.requires_grad = False
    unfrozen_params = 0
    for m in core.modules():
        if m.__class__.__name__.lower() == 'detect':
            for p in m.parameters():
                p.requires_grad = True
                unfrozen_params += p.numel() if hasattr(p, 'numel') else 1
    print(f"[freeze] Fallback applied: unfroze Detect head params (~{unfrozen_params}).")


def train_model(model, dataset_path, epochs, imgsz, device, resume=False):
    """Unified YOLO training function"""
    train_params = dict(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=8,
        patience=100,
        lr0=1e-4,
        optimizer='AdamW',
        seed=42,
        save_period=10,
        plots=True,
        resume=resume,
        project='runs/train',
        mosaic=0,
        scale=0
    )

    print(f"{'Resuming' if resume else 'Starting'} training on {device} for {epochs} epochs ...")
    return model.train(**train_params)


def main(mode, dataset_path, epochs, imgsz, device, pretrained_model, weights):
    if mode not in {"scratch", "resume"}:
        sys.exit("Error: mode must be 'scratch' or 'resume'")

    if mode == "scratch":
        # For head fine-tuning, start from pretrained weights rather than YAML.
        chosen_weights = weights or "yolo11n.pt"
        print(f"Training with frozen backbone; initializing from pretrained weights: {chosen_weights}")
        model = YOLO(chosen_weights)
        freeze_backbone_only(model)
        results = train_model(model, dataset_path, epochs, imgsz, device, resume=False)

    elif mode == "resume":
        if not pretrained_model:
            sys.exit("Error: Please provide a --pretrained_model when resuming training.")
        print(f"Resuming training from checkpoint: {pretrained_model}")
        model = YOLO(pretrained_model)
        # Re-apply freezing on resume to ensure optimizer param groups honor the constraint
        freeze_backbone_only(model)
        results = train_model(model, dataset_path, epochs, imgsz, device, resume=True)

    print("Training finished successfully.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv11 training with frozen backbone (head fine-tuning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--mode", type=str, choices=["scratch", "resume"], default="scratch",
                        help="Training mode: 'scratch' (new training) or 'resume' (continue training)")
    parser.add_argument("--dataset", type=str, default="../dataset_yolo_downsampled/dataset_yolo_downsampled.yaml",
                        help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=1280, help="Input image size")
    parser.add_argument("--device", type=str, default="cuda:2",
                        help="Training device, e.g., 'cuda:0', 'cuda:1', or 'cpu'")
    parser.add_argument("--pretrained_model", type=str, default=None,
                        help="Path to a trained model checkpoint (required if mode=resume)")
    parser.add_argument("--weights", type=str, default="yolo11n.pt",
                        help="Pretrained weights to initialize from when mode=scratch (e.g., 'yolo11n.pt' or path to .pt)")

    args = parser.parse_args()

    main(args.mode, args.dataset, args.epochs, args.imgsz, args.device, args.pretrained_model, args.weights)