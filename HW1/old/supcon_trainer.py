import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_datasets
from losses import SupConLoss
from model import SupConResNet
from utils.logger import setup_logger


def train_supcon(cfg):
    print("[SupCon] 啟動表示學習階段...")

    # Logger
    log_dir = os.path.join(cfg['output_dir'], cfg['timestamp'])
    logger = setup_logger(log_dir)
    logger.info(f"[SupCon] 使用 {cfg['num_global_crops']} 張 global crop 和 {cfg['num_local_crops']} 張 local crop")

    # 資料與 dataloader
    train_set, _, _ = get_datasets(cfg, mode='supcon')
    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True,
                              num_workers=cfg['num_workers'], pin_memory=True, drop_last=True)

    # 模型與損失
    model = SupConResNet(name=cfg['model_name'], pretrained=cfg['pretrained'])
    model.to(cfg['device'])

    # 多 GPU 支援
    if torch.cuda.device_count() > 1:
        print(f"[Info] 使用 {torch.cuda.device_count()} 顆 GPU 進行 SupCon 訓練")
        model = nn.DataParallel(model)

    criterion = SupConLoss(temperature=cfg['temperature']).to(cfg['device'])
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg['learning_rate'],
                                momentum=0.9, weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs_supcon'])

    writer = SummaryWriter(os.path.join(log_dir, 'logs'))

    model.train()
    for epoch in range(cfg['epochs_supcon']):
        epoch_loss = 0
        start_time = time.time()

        for images, labels in tqdm(train_loader, desc=f"SupCon Epoch {epoch+1}"):
            bsz = labels.size(0)
            labels = labels.to(cfg['device'])

            # 將 global crops 合併送進 model
            global_views = images[:cfg['num_global_crops']]  # list of tensors
            global_images = torch.cat(global_views, dim=0).to(cfg['device'])  # (num_global_crops * B, C, H, W)
            features = model(global_images)  # (num_global_crops * B, D)

            # 拆成每個 view
            features_split = torch.split(features, bsz, dim=0)
            f1, f2 = features_split[0], features_split[1]
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # (B, 2, D)

            loss = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        logger.info(f"[Epoch {epoch+1}/{cfg['epochs_supcon']}] Loss: {avg_loss:.4f}  Time: {time.time() - start_time:.1f}s")

    # 儲存 encoder
    model_path = os.path.join(cfg['output_dir'], cfg['timestamp'], f"supcon_encoder_{cfg['timestamp']}.pth")
    torch.save(model.module.encoder.state_dict() if isinstance(model, nn.DataParallel) else model.encoder.state_dict(), model_path)
    logger.info(f"[儲存模型] Encoder 權重儲存於: {model_path}")

    writer.close()
