# YOLO11 Vehicle Detection with Roboflow

使用 Ultralytics YOLO11s 與 Roboflow Vehicles 資料集進行遷移學習的車輛偵測專案。訓練在 Google Colab 的 NVIDIA Tesla T4 上完成，包含可重現的訓練、驗證、推論程式，以及保留原始輸出的 notebook。

## 實驗結果

以下數值來自最佳 checkpoint 對 **validation split** 的評估（966 images、13,450 instances）。Notebook 對 test split 執行了推論，但未計算 test split 的 ground-truth metrics，因此沒有把 test inference 誤標為 test evaluation。

| Metric | Score | Percentage |
|---|---:|---:|
| Precision | 0.4964 | 49.64% |
| Recall | 0.6102 | 61.02% |
| mAP@50 | 0.4676 | 46.76% |
| mAP@50–95 | 0.3260 | 32.60% |

完整精度數值與每類別 AP 儲存在 [`metrics/validation_metrics.json`](metrics/validation_metrics.json)。

## 訓練設定

| Item | Value |
|---|---|
| Base model | `yolo11s.pt` |
| Dataset | Roboflow Universe `roboflow-100/vehicles-q0x2v`, version 2 |
| Classes | 12 |
| Train / Validation / Test | 2,634 / 966 / 458 images |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | AdamW (Ultralytics auto selection) |
| Hardware | NVIDIA Tesla T4 |
| Ultralytics | 8.4.152 |

偵測類別：`big bus`、`big truck`、`bus-l-`、`bus-s-`、`car`、`mid truck`、`small bus`、`small truck`、`truck-l-`、`truck-m-`、`truck-s-`、`truck-xl-`。

## 專案結構

```text
.
├── notebooks/
│   └── yolo11_vehicle_detection_roboflow.ipynb
├── scripts/
│   ├── download_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── metrics/
│   └── validation_metrics.json
├── .env.example
└── requirements.txt
```

## 快速開始

建議使用 Python 3.10 以上版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

設定 Roboflow API key，並下載資料集：

```bash
export ROBOFLOW_API_KEY="your_api_key"
python scripts/download_dataset.py
```

資料下載完成後，以實際的 `data.yaml` 路徑執行訓練：

```bash
python scripts/train.py --data datasets/vehicles-2/data.yaml
```

評估最佳 checkpoint：

```bash
python scripts/evaluate.py \
  --model runs/detect/train_50e/weights/best.pt \
  --data datasets/vehicles-2/data.yaml \
  --split val
```

對圖片、資料夾或影片執行推論：

```bash
python scripts/predict.py \
  --model runs/detect/train_50e/weights/best.pt \
  --source path/to/images
```

## Notebook

[`notebooks/yolo11_vehicle_detection_roboflow.ipynb`](notebooks/yolo11_vehicle_detection_roboflow.ipynb) 保留了原始訓練紀錄、曲線、confusion matrix 與推論範例。程式已改用環境變數讀取 Roboflow API key，repository 不包含資料集、模型權重或任何密鑰。

## 資料與模型注意事項

- 資料集不納入 Git；請透過 Roboflow API 下載，並遵守來源資料集的授權條款。
- `best.pt`、`last.pt` 等大型模型權重不納入 Git；可自行訓練產生。
- 公開分享模型或資料前，請另外確認 Ultralytics 與資料集的授權是否符合使用情境。

