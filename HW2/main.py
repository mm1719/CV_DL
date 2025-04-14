import argparse
import os
import torch
from datetime import datetime

from torch.utils.data import DataLoader
from config import cfg
from utils.logger import setup_logger
from utils.tensorboard_writer import create_tensorboard_writer
from utils.metrics import coco_evaluation
from models.faster_rcnn import get_model
from dataset.coco_dataset import CocoDetectionDataset
from engine.trainer import train_one_epoch, evaluate, save_checkpoint
from engine.tester import run_inference
from utils.image_transform import ResizeWithPaddingForTraining, AugmentWithoutResize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test", "eval"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--pred-json", type=str, default=None)
    return parser.parse_args()


def train():
    logger, log_file, timestamp = setup_logger(cfg.output_dir)
    writer = create_tensorboard_writer(cfg.tensorboard_logdir)

    transform = (
        AugmentWithoutResize()
        if cfg.backbone == "resnet50"
        else ResizeWithPaddingForTraining(target_size=256)
    )
    train_dataset = CocoDetectionDataset(
        cfg.train_dir, cfg.train_json, transforms=transform
    )
    valid_dataset = CocoDetectionDataset(
        cfg.valid_dir, cfg.valid_json, transforms=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: tuple(zip(*x)),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    model = get_model(cfg.num_classes).to(cfg.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = (
        torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if cfg.optimizer_name == "AdamW"
        else torch.optim.SGD(
            params, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
        )
    )

    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)
        if cfg.scheduler_name == "CosineAnnealingLR"
        else torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma
        )
    )

    best_loss = float("inf")
    best_map = 0.0
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, cfg.num_epochs + 1):
        loss = train_one_epoch(model, optimizer, train_loader, epoch)
        writer.add_scalar("Loss/train", loss, epoch)

        val_loss = evaluate(model, valid_loader)
        writer.add_scalar("Loss/valid", val_loss, epoch)
        lr_scheduler.step()

        # mAP 評估
        run_inference(
            model,
            cfg.valid_dir,
            save_json_path=os.path.join(cfg.output_dir, "val_pred.json"),
        )
        scores = coco_evaluation(
            pred_json_path=os.path.join(cfg.output_dir, "val_pred.json"),
            gt_json_path=cfg.valid_json,
        )

        val_ap = scores["AP"]
        val_ap50 = scores["AP50"]
        val_acc = scores["ACC"]

        logger.info(
            f"[Epoch {epoch}] Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | AP: {val_ap:.4f} | "
            f"AP50: {val_ap50:.4f} | ACC: {val_acc:.4f}"
        )

        # TensorBoard
        writer.add_scalar("mAP/val", val_ap, epoch)
        writer.add_scalar("AP50/val", val_ap50, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        # 儲存最佳模型（根據 val_loss）
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(cfg.output_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_path)
            logger.info(f"[更新最佳模型] 🔥 Epoch {epoch} 儲存為 best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"[早停計數] No improvement, patience={patience_counter}/{cfg.early_stopping_patience}"
            )
            if patience_counter >= cfg.early_stopping_patience:
                logger.info("⛔️ 觸發 Early Stopping，提前結束訓練！")
                break

    writer.close()


def test(model_path):
    model = get_model(cfg.num_classes)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model"])
    run_inference(
        model=model,
        test_dir=cfg.test_dir,
        save_json_path=os.path.join(cfg.output_dir, "pred.json"),
        save_csv_path=os.path.join(cfg.output_dir, "pred.csv"),
    )


def evaluate_pred(model_path):
    model = get_model(cfg.num_classes)
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model"])
    model.to(cfg.device)
    run_inference(
        model=model,
        test_dir=cfg.valid_dir,
        save_json_path=os.path.join(cfg.output_dir, "val_pred.json"),
        save_csv_path=os.path.join(cfg.output_dir, "val_pred.csv"),
    )
    scores = coco_evaluation(
        pred_json_path=os.path.join(cfg.output_dir, "val_pred.json"),
        gt_json_path=cfg.valid_json,
    )

    print("\n[COCO 評估結果]")
    with open(os.path.join(cfg.output_dir, "val_metrics.txt"), "w") as f:
        f.write("[COCO 評估結果]\n")
        for k, v in scores.items():
            line = f"{k}: {v:.4f}"
            print(line)
            f.write(f"{line}\n")
    print("📄 評估結果已儲存")


if __name__ == "__main__":
    args = parse_args()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        cfg.output_dir = os.path.join("outputs", timestamp)
    os.makedirs(cfg.output_dir, exist_ok=True)

    cfg.tensorboard_logdir = os.path.join(cfg.output_dir, "tensorboard")
    cfg.checkpoint_path = os.path.join(cfg.output_dir, "model.pth")
    cfg.log_file = os.path.join(cfg.output_dir, "log.txt")

    if args.mode == "train":
        train()
    elif args.mode == "test":
        if not args.model_path:
            raise ValueError("請提供 --model-path")
        test(args.model_path)
    elif args.mode == "eval":
        if not args.model_path:
            raise ValueError("請提供 --model-path")
        evaluate_pred(args.model_path)
