import os
from torch.utils.tensorboard import SummaryWriter


def create_tensorboard_writer(log_dir):
    """
    建立 TensorBoard writer 並確保資料夾存在。
    Args:
        log_dir: tensorboard log 輸出路徑
    Returns:
        writer: SummaryWriter 實例
    """
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"🖋️ TensorBoard log 寫入：{log_dir}")
    return writer
