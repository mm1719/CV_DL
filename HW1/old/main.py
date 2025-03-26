import argparse
import os
import torch
from datetime import datetime

from supcon_trainer import train_supcon
from classifier_trainer import train_classifier
from tester import test_model
from config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description="SupCon Training Pipeline")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['supcon', 'classifier', 'test', 'knn'],
                        help='選擇要執行的階段')
    parser.add_argument('--classifier_type', type=str, default='mlp', choices=['linear', 'mlp'], help='選擇分類器類型')
    parser.add_argument('--config', type=str, default=None, help="可選的 config.py 檔案路徑")
    parser.add_argument('--timestamp', type=str, default=None, help="時間戳記（可選，預設為當前時間）")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config(args.config)
    cfg['classifier_type'] = args.classifier_type

    # 設定時間戳記
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg['timestamp'] = timestamp

    # 建立輸出資料夾
    os.makedirs(os.path.join(cfg['output_dir'], timestamp), exist_ok=True)

    # GPU 檢查
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device

    # 執行指定階段
    if args.mode == 'supcon':
        train_supcon(cfg)

    elif args.mode == 'classifier':
        train_classifier(cfg)

    elif args.mode == 'test':
        test_model(cfg)

    elif args.mode == 'knn':
        from knn_eval import evaluate_knn
        evaluate_knn(cfg)

    else:
        raise ValueError("無效的 mode 選項")


if __name__ == '__main__':
    main()
