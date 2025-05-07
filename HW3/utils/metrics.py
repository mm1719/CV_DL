# utils/metrics.py

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


def evaluate_model(cfg, model, dataset_name="cells_val", output_folder=None):
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)
    val_loader = build_detection_test_loader(cfg, dataset_name)
    return inference_on_dataset(model, val_loader, evaluator)
