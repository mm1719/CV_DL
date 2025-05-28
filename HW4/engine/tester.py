import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from dataloader.hw4_dataset import HW4ImageDataset
from utils.img2npz import convert_folder_to_npz
from PIL import Image
import zipfile
from utils.logger import get_logger


def predict(model, test_set, config):
    logger = get_logger(os.path.join(config.output_dir, "test.log"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    state_dict = torch.load(config.model_path, map_location=device)

    # 如果是從多卡儲存的 state_dict，鍵可能會包含 'module.'
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)

    model.eval()
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    results = {}

    with torch.no_grad():
        for filename, x in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            pred = model(x).clamp(0, 1)
            pred_np = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)
            results[filename[0]] = pred_np

    np.savez(config.output_npz, **results)
    logger.info(f"✅ Saved prediction to {config.output_npz}")

    zip_path = config.output_npz.replace(".npz", ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(config.output_npz, arcname="pred.npz")
    logger.info(f"📦 Compressed to {zip_path}")


if __name__ == "__main__":
    import argparse
    from model.promptir import PromptIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_npz", type=str, default="pred.npz")
    args = parser.parse_args()

    class Config:
        data_root = args.data_root
        model_path = args.model_path
        output_npz = args.output_npz

    test_set = HW4ImageDataset(root_dir=Config.data_root, mode="test")
    model = PromptIR(decoder=True)
    predict(model, test_set, Config)
