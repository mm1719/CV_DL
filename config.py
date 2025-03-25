import os

class Config:
    # 模型設定
    model_name = "regnety_160"
    num_classes = 100
    pretrained = True

    # 資料路徑
    data_dir = "hw1-data/data"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test", "unknown")

    # 圖片大小
    image_size = 768

    # 訓練設定
    epochs = 50
    batch_size = 16
    num_workers = 8
    pin_memory = True

    # 優化器設定（依照論文）
    optimizer = "sgd"
    # lr = 0.1 * (batch_size / 128)  # scaling rule
    lr = 0.01
    momentum = 0.9
    # weight_decay = 5e-5
    weight_decay = 1e-4

    # Scheduler
    scheduler = "cosine"
    warmup_epochs = 5

    # Log & Checkpoint
    output_dir = "outputs"
    log_interval = 10
    save_interval = 1

    # GPU 設定
    use_amp = False  # 自動混合精度
    multi_gpu = True
    device = "cuda"

config = Config()