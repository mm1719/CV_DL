import re
import matplotlib.pyplot as plt

def parse_losses_from_log(log_path):
    train_losses = []
    val_losses = []
    epochs = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(r"\[Epoch (\d+)\] Train Loss: ([\d.]+) \| Valid Loss: ([\d.]+)", line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))

                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)

    return epochs, train_losses, val_losses


def plot_losses(log_path, save_path="loss_curve.png"):
    epochs, train_losses, val_losses = parse_losses_from_log(log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label="Train Loss")
    plt.plot(epochs, val_losses, marker='s', label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 圖表已儲存至 {save_path}")


# 使用方式
if __name__ == "__main__":
    log_path = "log_20250412-004329.txt"
    plot_losses(log_path)
