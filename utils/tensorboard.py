from torch.utils.tensorboard import SummaryWriter
import os

def init_tensorboard(log_dir="runs"):
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)
