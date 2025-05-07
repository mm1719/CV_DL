# engine/trainer.py

import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.evaluation import COCOEvaluator
from detectron2 import model_zoo

from config import Config
from dataloader.dataloader import register_cell_dataset


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)


def setup_config(cfg_obj: Config):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_obj.yaml))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_obj.yaml)
    cfg.MODEL.DEVICE = "cuda"

    cfg.MODEL.META_ARCHITECTURE = cfg_obj.meta_arch

    if cfg_obj.variant == "weakly":
        from dataloader.coco_mil_dataloader import register_mil_dataset

        register_mil_dataset()
        cfg.DATASETS.TRAIN = ("cells_train_mil",)
        cfg.DATASETS.TEST = ("cells_val_mil",)
    else:
        from dataloader.dataloader import register_cell_dataset

        register_cell_dataset()
        cfg.DATASETS.TRAIN = ("cells_train",)
        cfg.DATASETS.TEST = ("cells_val",)

    cfg.DATALOADER.NUM_WORKERS = 4

    cfg.INPUT.MASK_FORMAT = "bitmask"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg_obj.num_classes

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = (7000, 9000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000

    cfg.INPUT.MIN_SIZE_TRAIN = (512,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 0
    cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.TEST.EVAL_PERIOD = 1000

    cfg.OUTPUT_DIR = str(cfg_obj.output_dir.resolve())
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.freeze()
    default_setup(cfg, {})  # 若你未使用 detectron2 的 command line argument，可空傳 {}

    return cfg


def do_train(cfg_obj: Config, resume=False, eval_only=False):
    register_cell_dataset()
    cfg = setup_config(cfg_obj)
    trainer = Trainer(cfg)

    if eval_only:
        trainer.resume_or_load(resume=resume)
        trainer.test(cfg, trainer.model)
        return

    trainer.resume_or_load(resume=resume)
    trainer.train()
