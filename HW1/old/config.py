# config.py


def get_config(path=None):
    # 預設設定
    cfg = {
        # 資料集路徑
        "data_root": "./hw1-data/data",
        # 模型與輸出
        "output_dir": "./outputs",
        "model_name": "resnext50_32x4d",
        "pretrained": True,
        "freeze_backbone": True,  # 可選擇是否 freeze encoder during classifier training
        "num_classes": 100,
        "tta_times": 5,  # 測試時對每張圖片推論 n 次，平均後預測
        "classifier_type": "mlp",
        # 訓練參數
        "epochs_supcon": 100,
        "epochs_cls": 50,
        "batch_size": 64,
        "learning_rate": 0.03,
        "weight_decay": 1e-4,
        "temperature": 0.07,
        "num_workers": 4,
        "mlp_learning_rate": 1e-3,
        "mlp_weight_decay": 3e-6,
        # 評估
        "knn_k": 5,
        "use_tta": True,
        # 影像處理
        "image_size": 224,
        "resize_size": 256,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "use_multi_crop": True,
        "num_global_crops": 3,
        "num_local_crops": 6,
        "local_crop_size": 96,
    }

    # 如果有額外 config 檔案可載入覆蓋
    if path:
        import json

        with open(path, "r") as f:
            user_cfg = json.load(f)
            cfg.update(user_cfg)

    return cfg
