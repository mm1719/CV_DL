import os
import logging
from datetime import datetime

def setup_logger(output_dir, name='log'):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_file = os.path.join(output_dir, f"{name}_{timestamp}.txt")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
