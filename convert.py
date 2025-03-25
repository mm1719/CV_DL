import torch
import argparse
import os

def convert_ckpt_to_pth(ckpt_path, save_path=None):
    print(f"🔍 Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" not in ckpt:
        raise ValueError("❌ This checkpoint does not contain 'state_dict'. It may not be a PyTorch Lightning checkpoint.")

    state_dict = ckpt["state_dict"]

    # 去除 'model.' 或 'net.' 等前綴（視情況調整）
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "").replace("net.", "")
        new_state_dict[new_key] = v

    if save_path is None:
        save_path = os.path.splitext(ckpt_path)[0] + ".pth"

    torch.save(new_state_dict, save_path)
    print(f"✅ Converted successfully: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="Path to the .ckpt file")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save .pth file")
    args = parser.parse_args()

    convert_ckpt_to_pth(args.ckpt_path, args.save_path)
