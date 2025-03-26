from torch.utils.tensorboard import SummaryWriter
import os

def init_tensorboard(log_dir="runs"):
    """Initialize TensorBoard writer.
    Args:
        log_dir: Directory to save TensorBoard logs.
    Returns:
        writer: TensorBoard writer.
    """
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)
