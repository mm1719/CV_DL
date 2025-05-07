# main.py

import argparse
from config import Config
from engine.trainer import do_train
from engine.tester import do_test
from utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="HW3 - Instance Segmentation")
    parser.add_argument(
        "--mode", choices=["train", "test"], required=True, help="train or test"
    )
    parser.add_argument(
        "--timestamp", type=str, help="Reuse timestamp folder (e.g., 20250506-210000)"
    )
    parser.add_argument("--weights", type=str, help="Path to model weights for testing")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate without training (train mode only)",
    )
    parser.add_argument(
        "--variant", type=str, default="default", help="default or weakly"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(mode=args.mode, timestamp=args.timestamp, variant=args.variant)

    # Setup logger
    logger = setup_logger(str(cfg.log_path))
    logger.info(f"Mode = {args.mode}, Timestamp = {cfg.timestamp}")

    if args.mode == "train":
        do_train(cfg, resume=args.resume, eval_only=args.eval_only)
    elif args.mode == "test":
        if not args.weights:
            raise ValueError("Must provide --weights for test mode")
        cfg.pretrained_weights = args.weights
        do_test(cfg)


if __name__ == "__main__":
    main()
