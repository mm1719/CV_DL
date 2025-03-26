import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time
from config import config
import numpy as np


def mixup_data(x, y, alpha=1.0):
    """
    Apply mixup to a batch of images and labels.
    Args:
        x: Input batch of images.
        y: Input batch of labels.
        alpha: Mixup hyperparameter.
    Returns:
        mixed_x: Mixed images.
        y_a: Labels of the first image.
        y_b: Labels of the second image.
        lam: Lambda value sampled from the beta distribution.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, logger, writer):
    """
    One epoch of training of the model.
    Args:
        model: Model to train.
        dataloader: DataLoader for training images.
        optimizer: Optimizer for training.
        criterion: Loss function.
        scaler: GradScaler for mixed precision training.
        device: Device to use.
        epoch: Current epoch number.
        logger: Logger object.
        writer: TensorBoard writer.
    Returns:
        avg_loss: Average loss of the epoch.
        acc: Accuracy of the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = total = 0
    start_time = time.time()

    pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}")
    for images, labels in pbar:
        if isinstance(images, list):
            images = torch.cat(images, dim=0)
            labels = labels.repeat(len(images) // len(labels))

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        if config.use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=config.mixup_alpha)
            with autocast(device_type="cuda", enabled=config.use_amp):
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        else:
            with autocast(device_type="cuda", enabled=config.use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        if config.use_mixup:
            _, preds = outputs.max(1)
            correct += lam * (preds == labels_a).sum().item() + (1 - lam) * (preds == labels_b).sum().item()
            total += labels.size(0)
        else:
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)


        acc = 100. * correct / total
        pbar.set_postfix(loss=running_loss/total, acc=f"{acc:.2f}%")

    elapsed = time.time() - start_time
    avg_loss = running_loss / total
    acc = 100. * correct / total
    logger.info(f"[Epoch {epoch}/{config.epochs}] Loss: {avg_loss:.4f}  Acc: {acc:.2f}%  Time: {elapsed:.1f}s")
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", acc, epoch)
    writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
    return avg_loss, acc

def validate(model, dataloader, criterion, device, epoch, logger, writer):
    """ 
    Validate the model on the validation set.
    Args:
        model: Model to validate.
        dataloader: DataLoader for validation images.
        criterion: Loss function.
        device: Device to use.
        epoch: Current epoch number.
        logger: Logger object.
        writer: TensorBoard writer.
    Returns:
        avg_loss: Average loss of the validation.
        acc: Accuracy of the validation.
    """
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
    logger.info(f"[Val   {epoch}/{config.epochs}] Loss: {avg_loss:.4f}  Acc: {acc:.2f}%")
    writer.add_scalar("Val/Loss", avg_loss, epoch)
    writer.add_scalar("Val/Accuracy", acc, epoch)
    return avg_loss, acc

def train_model(model, train_loader, val_loader, output_dir, writer, logger):
    """
    Train the model.
    Args:
        model: Model to train.
        train_loader: DataLoader for training images.
        val_loader: DataLoader for validation images.
        output_dir: Directory to save the best model.
        writer: TensorBoard writer.
        logger: Logger object.
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: epoch / config.warmup_epochs if epoch < config.warmup_epochs else 1.0)
    
    scheduler_after = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,       # 初始週期
        T_mult=2     # 每次重啟後週期乘上的倍數
    )

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
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"[儲存模型] Encoder 權重儲存於: {model_path}")
        else:
            early_stop_counter += 1
            logger.info(f"EarlyStop counter: {early_stop_counter}/{config.early_stopping_patience}")
            if early_stop_counter >= config.early_stopping_patience:
                logger.info(f"早停於 Epoch {epoch}（最佳為 Epoch {best_epoch}）")
                break
