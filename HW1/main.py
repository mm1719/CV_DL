import argparse
import os
import torch
from datetime import datetime

from config import config
from model import build_model
from dataset import get_dataloaders, get_test_loader
from train import train_model
from prediction import predict

from utils.logger import setup_logger
from utils.tensorboard import init_tensorboard
from utils.feature_extract import extract_features_for_tsne
from utils.metrics import plot_tsne, plot_confusion_matrix
from utils.remap import remap_predictions


def create_output_dir(timestamp):
    """
    Create output directory for saving logs, models, predictions, and analysis images
    Args:
        timestamp (str): custom run folder name
    Returns:
        str: output directory path
    """
    output_dir = os.path.join(config.output_dir, timestamp)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    return output_dir


def main():
    """
    Run the training, testing, t-SNE visualization, or confusion matrix generation
    based on the mode argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, required=True, choices=["train", "test", "tsne", "cm"]
    )
    parser.add_argument(
        "--timestamp", type=str, default=None, help="custom run folder name"
    )
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = create_output_dir(timestamp)

    logger, _, _ = setup_logger(output_dir)
    writer = init_tensorboard(log_dir=os.path.join(output_dir, "logs"))

    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output Directory: {output_dir}")

    if args.mode == "train":
        logger.info("開始訓練")
        model = build_model(config)
        train_loader, val_loader = get_dataloaders(config)
        train_model(model, train_loader, val_loader, output_dir, writer, logger)

    elif args.mode == "test":
        logger.info("開始預測")
        model = build_model(config)
        model_path = os.path.join(output_dir, "best_model.pth")
        if not os.path.exists(model_path):
            logger.error(f"找不到模型權重: {model_path}")
            return
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        test_loader = get_test_loader(config)
        predict(model, test_loader, config, output_dir, writer, logger)
        remap_predictions(os.path.join(output_dir, "prediction.csv"))

    elif args.mode == "tsne":
        logger.info("生成 t-SNE 圖")
        model = build_model(config)
        model_path = os.path.join(output_dir, "best_model.pth")
        if not os.path.exists(model_path):
            logger.error(f"找不到模型權重: {model_path}")
            return

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        model.eval()
        _, val_loader = get_dataloaders(config)
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model.to(device)
        features, labels = extract_features_for_tsne(model, val_loader, device)
        tsne_path = os.path.join(output_dir, "tsne.png")
        plot_tsne(features, labels, save_path=tsne_path)

    elif args.mode == "cm":
        logger.info("生成 Confusion Matrix")
        model = build_model(config)
        model_path = os.path.join(output_dir, "best_model.pth")
        if not os.path.exists(model_path):
            logger.error(f"找不到模型權重: {model_path}")
            return
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        _, val_loader = get_dataloaders(config)
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        model.to(device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plot_confusion_matrix(all_labels, all_preds, save_path=cm_path)

    logger.info("任務完成!")


if __name__ == "__main__":
    main()
