import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.utils as vutils
import numpy as np
from utils.logger import get_logger


def psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction="mean")
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    epoch_loss = 0
    for x, y in tqdm(loader, desc="Training", leave=False):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.l1_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)
    return epoch_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    total_psnr = 0
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating", leave=False):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_psnr += psnr(pred, y).item() * x.size(0)
            total_loss += F.l1_loss(pred, y, reduction="mean").item() * x.size(
                0
            )  # 平均後再乘 batch size
    avg_psnr = total_psnr / len(loader.dataset)
    avg_loss = total_loss / len(loader.dataset)
    return avg_psnr, avg_loss


def save_sample_images(pred, gt, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    grid = torch.cat([pred, gt], dim=0)  # Stack pred and GT
    vutils.save_image(
        grid,
        os.path.join(output_dir, f"epoch{epoch:03d}.png"),
        nrow=gt.size(0),
        normalize=True,
    )


def train(model, train_set, val_set, config):
    logger = get_logger(os.path.join(config.output_dir, "train.log"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        logger.info(f"🖥️ 使用 {torch.cuda.device_count()} 張 GPU 進行 DataParallel 訓練")
        model = torch.nn.DataParallel(model)

    model.to(device)

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )

    best_psnr = 0
    for epoch in range(1, config.epochs + 1):
        logger.info(f"\n🟢 Epoch {epoch}/{config.epochs}")
        loss = train_one_epoch(model, train_loader, optimizer, device)
        logger.info(f"Train Loss: {loss:.4f}")

        val_psnr, val_loss = evaluate(model, val_loader, device)
        logger.info(f"Val PSNR: {val_psnr:.2f} dB")
        logger.info(f"Val Loss: {val_loss:.4f}")

        scheduler.step()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(config.output_dir, "best_model.pt")
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            logger.info("💾 Saved best model")

        if epoch % 10 == 0:
            x, y = next(iter(val_loader))
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
            save_sample_images(pred, y, config.output_dir, epoch)

    logger.info(f"\n🎉 Best Val PSNR: {best_psnr:.2f} dB")
