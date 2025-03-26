import torch
import pandas as pd
from tqdm import tqdm
import os
import time
import torch.nn.functional as F


def predict(model, test_loader, config, output_dir, writer, logger):
    """
    Predict test images and save the results to a CSV file.
    Args:
        model: Model to predict test images.
        test_loader: DataLoader for test images.
        config: Configuration object.
        output_dir: Directory to save the prediction CSV file.
        writer: TensorBoard writer.
        logger: Logger object
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    predictions = []
    image_names = []

    img_paths = test_loader.dataset.image_paths

    logger.info("開始測試集預測")
    start_time = time.time()

    with torch.no_grad():
        for i, (images, _) in enumerate(tqdm(test_loader, desc="Predicting")):
            # TTA（[B, 10, C, H, W] → [B, num_classes])
            if images.ndim == 5:
                B, N, C, H, W = images.shape
                images = images.view(-1, C, H, W).to(device)  # [B×10, C, H, W]
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                probs = probs.view(B, N, -1).mean(dim=1)  # [B, num_classes]
                preds = probs.argmax(dim=1)
            else:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

            predictions.extend(preds.cpu().tolist())

            batch_paths = img_paths[i * config.batch_size : (i + 1) * config.batch_size]
            names = [os.path.splitext(os.path.basename(p))[0] for p in batch_paths]
            image_names.extend(names)

    total_time = time.time() - start_time
    logger.info(f"預測完成，總耗時 {total_time:.1f} 秒，共 {len(predictions)} 張圖片")

    # 儲存 prediction.csv
    df = pd.DataFrame({"image_name": image_names, "pred_label": predictions})
    pred_path = os.path.join(output_dir, "prediction.csv")
    df.to_csv(pred_path, index=False)
    logger.info(f"已儲存預測結果: {pred_path}")

    # TensorBoard log
    writer.add_scalar("Test/Total Images", len(predictions))
    writer.add_scalar("Test/Time (s)", total_time)
