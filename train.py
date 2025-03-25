import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time
from config import config

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, logger, writer):
    model.train()
    running_loss = 0.0
    correct = total = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with autocast(device_type="cuda", enabled=config.use_amp):
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
        pbar.set_postfix(loss=running_loss/total, acc=f"{acc:.2f}%")

    elapsed = time.time() - start_time
    avg_loss = running_loss / total
    acc = 100. * correct / total
    logger.info(f"[Epoch {epoch}] Loss: {avg_loss:.4f}  Time: {elapsed:.1f}s")
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)
    return avg_loss, acc

def validate(model, dataloader, criterion, device, epoch, logger, writer):
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
            pbar.set_postfix(loss=running_loss/total, acc=f"{acc:.2f}%")

    avg_loss = running_loss / total
    acc = 100. * correct / total
    logger.info(f"[Val Epoch {epoch}] Loss: {avg_loss:.4f} Acc: {acc:.2f}%")
    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/Accuracy", acc, epoch)
    return avg_loss, acc

def train_model(model, train_loader, val_loader, logger, writer):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs - config.warmup_epochs, eta_min=config.min_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch / config.warmup_epochs if epoch < config.warmup_epochs else 1.0)
    scheduler_after = cosine_scheduler

    scaler = GradScaler(enabled=config.use_amp)

    best_val_loss = float("inf")
    best_epoch = 0
    early_stop_counter = 0

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, logger, writer)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, logger, writer)

        if epoch <= config.warmup_epochs:
            scheduler.step()
        else:
            scheduler_after.step()

        logger.info(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0
            os.makedirs(config.output_dir, exist_ok=True)
            model_path = os.path.join(config.output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"✅ 儲存模型於: {model_path}")
        else:
            early_stop_counter += 1
            logger.info(f"📉 EarlyStop counter: {early_stop_counter}/{config.early_stopping_patience}")

        if early_stop_counter >= config.early_stopping_patience:
            logger.info(f"🛑 早停於 Epoch {epoch}（最佳為 Epoch {best_epoch}）")
            break