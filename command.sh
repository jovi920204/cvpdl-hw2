CUDA_VISIBLE_DEVICES=2 python train.py --mode resume --pretrained_model runs/train/train7/weights/last.pt 

CUDA_VISIBLE_DEVICES=2 python train.py \
    --mode scratch \
    --epochs 200 \
    --imgsz 1920

CUDA_VISIBLE_DEVICES=0 python train.py \
    --mode resume \
    --epochs 200 \
    --imgsz 1920 \
    --pretrained_model runs/train/train11/weights/last.pt

CUDA_VISIBLE_DEVICES=2 python test.py \
    --model runs/train/train8/weights/epoch150.pt \
    --img_dir ../dataset/test \
    --save_dir results/test8 \
    --conf 0.15 \
    --iou 0.75 \
    --imgsize 960 \
    --tta \
    --batch_size 1 \
    --save_img

CUDA_VISIBLE_DEVICES=2 python train_freeze.py \
    --mode scratch \
    --epochs 100 \
    --imgsz 1280 \
    --weights runs/train/train7/weights/best.pt
    