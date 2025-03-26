import os
import logging
from datetime import datetime

def setup_logger(output_dir, name='log'):
    """
    Setup logging configuration
    Args:
        output_dir: Directory to save log file.
        name: Name of the logger.
    Returns:
        logger: Logger object.
        log_file: Path to the log file.
        timestamp: Timestamp string.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(output_dir, f"{name}_{timestamp}.txt")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 防止重複 log 多次

    # 移除舊 handler（避免 Jupyter 或多次初始化重複寫入）
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S")

    # Console log
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File log
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"📋 Logging to file: {log_file}")
    return logger, log_file, timestamp
