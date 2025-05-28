import logging
import os


def get_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清空之前的 handler，避免重複寫入

    # 檔案記錄
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console 輸出
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger
