CUDA_VISIBLE_DEVICES=2 python train.py --mode resume --pretrained_model runs/train/train7/weights/last.pt 


CUDA_VISIBLE_DEVICES=2 python test.py \
    --model runs/train/train8/weights/best.pt \
    --img_dir ../dataset/test \
    --save_dir results/test8 \
    --conf 0.15 \
    --iou 0.75 \
    --imgsize 1280 \
    --batch_size 32 \
    --save_img

CUDA_VISIBLE_DEVICES=2 python train_freeze.py \
    --mode scratch \
    --epochs 100 \
    --imgsz 1280 \
    --weights runs/train/train7/weights/best.pt
    