## Classes Distribution
Class 0: 23275 instances
Class 1: 1345 instances 
Class 2: 3363 instances 
Class 3: 5348 instances 

## COCO Converter for provided dataset

這個 repository 含有一個簡單的轉換腳本，用來把每張影像對應的 `.txt` 標註轉成 COCO JSON。

預設假設
- 每個影像放在 `dataset/train` 或 `dataset/test`，影像副檔名支援 png/jpg/jpeg/bmp。
- 每張影像若有標註，則有一個同名的 `.txt` 檔案。例如 `img0001.png` 對應 `img0001.txt`。
- 預設輸入標註格式為：class,cx,cy,w,h，其中 cx,cy,w,h 為絕對像素，且 (cx,cy) 為 bounding box 的中心點。
- 類別會以 `class_{id}` 命名，並且 COCO 的 category id 會從 1 開始（對應原始 class 0 -> id 1）。

如果你的標註格式不同，可以使用 `--input-format` 選項：
- xywh_abs_center (預設)：class,cx,cy,w,h（絕對像素，中心）
- xywh_abs_topleft：class,x,y,w,h（絕對像素，左上角）
- yolo：class,xc_norm,yc_norm,w_norm,h_norm（相對座標，範圍 0..1，中心）

用法
1. 在此資料夾下執行：

   python3 scripts/convert_to_coco.py --dataset-dir ./dataset

2. 或指定輸出與格式：

   python3 scripts/convert_to_coco.py --dataset-dir ./dataset --output-dir ./dataset/annotations --input-format yolo

輸出
- 會在 `dataset/annotations`（或 `--output-dir` 指定）產生 `annotations_train.json` 與 `annotations_test.json`。

注意事項
- 程式會自動略過沒有對應 `.txt` 的影像（因此 test 可能沒有 annotation）。
- 如果你需要自訂 category 名稱對照表，可以在產生後編輯 JSON 的 `categories` 欄位。
