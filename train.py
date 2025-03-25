import torch
import torch.nn as nn
from tqdm import tqdm
import os
import time

# 相容新舊版本 AMP
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, config, logger):
    model.train()
    running_loss = 0.0
    correct = total = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=config.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = 100. * correct / total
        pbar.set_postfix(loss=running_loss / total, acc=f"{acc:.2f}%")

    epoch_time = time.time() - start_time
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    logger.info(f"[Epoch {epoch}/{config.epochs}] Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.2f}%  Time: {epoch_time:.1f}s")

    return epoch_loss, epoch_acc, epoch_time


def validate(model, dataloader, criterion, device, epoch, config, logger):
    model.eval()
    running_loss = 0.0
    correct = total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[Val]   Epoch {epoch}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = 100. * correct / total
            pbar.set_postfix(loss=running_loss / total, acc=f"{acc:.2f}%")

    val_loss = running_loss / total
    val_acc = 100. * correct / total
    logger.info(f"[Val   {epoch}/{config.epochs}] Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")

    return val_loss, val_acc


def train_model(model, train_loader, val_loader, config, output_dir, writer, logger):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = GradScaler(enabled=config.use_amp)

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, train_time = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, config, logger)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, config, logger)
        scheduler.step()

        # TensorBoard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"[儲存模型] Encoder 權重儲存於: {model_path}")
