import torch
import os
from tqdm import tqdm
from config import cfg


def train_one_epoch(model, optimizer, dataloader, epoch):
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}")
    for images, targets in pbar:
        images = [img.to(cfg.device) for img in images]
        targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader):
    model.train()  # ⚠️ 為了啟用 loss 分支（但不會訓練）
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc="[Eval]")
    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(cfg.device) for img in images]
            targets = [{k: v.to(cfg.device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())

            total_loss += losses.item()
            num_batches += 1
            pbar.set_postfix(loss=losses.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def save_checkpoint(model, optimizer, epoch, path):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )
    print(f"[儲存模型] Checkpoint 儲存於: {path}")
