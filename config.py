import os

class Config:
    # 模型設定
    model_name = "seresnext101d_32x8d"
    num_classes = 100
    pretrained = True

    # 資料路徑
    data_dir = "hw1-data/data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test", "unknown")

    # 圖片大小
    image_size = 384  # 一般是224

    # 訓練設定
    epochs = 50
    batch_size = 16
    num_workers = 8
    pin_memory = True
    num_augmentations = 3

    optimizer = "sgd"
    lr = 0.05 * (batch_size / 256)
    momentum = 0.9
    weight_decay = 1e-4

    # MixUp 設定
    use_mixup = True
    mixup_alpha = 0.4

    # Scheduler
    scheduler = "cosine"
    warmup_epochs = 5

    # 損失函數
    label_smoothing = 0.15

    # Log & Checkpoint
    output_dir = "outputs"
    log_interval = 10
    save_interval = 1
    early_stopping_patience = 10

    # 測試設定
    use_tta = True

    # GPU 設定
    use_amp = False
    multi_gpu = True
    device = "cuda"

config = Config()
