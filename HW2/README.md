# Selected Topics in Visual Recognition using Deep Learning HW2 Report

- **Name:** 袁孟華
- **Student ID:** 110550004
- **Github Repo:** <https://github.com/mm1719/CV_DL>

## How to run

`>./run.sh` includes all training, testing, and evaluation phases. After all the procedures of one experiment are done, the result would store in `outputs/` with this sort of structure:

```txt
.
|
`-- 20250412-004325                 # starting timestamp
    |-- best_model.pth              # model weights
    |-- log_20250412-004329.txt     # training log
    |-- pred.csv                    # prediction in csv format
    |-- pred.json                   # prediction in coco json format
    |-- tensorboard                 # tensorboard record
    |-- val_metrics.txt             # evaluation results
    |-- val_pred.csv                # validation in csv format
    `-- val_pred.json               # valudatoin in coco json format
```

Or you can run each phases seperately by following the format provided in `run.sh`.

## Introduction

This report detailed introduce most the tactics I have attempted to improving the performance of a Faster RCNN model on digits detection problem. Overall, the venture was full of twins and turns. Due to the limited time and hardware resources, my experiments mainly focused on preprocessing of data and alteration of the RCNN backbone. For better description, I would explain the backbone part first.

## Method

Aside from backbone selection and image preprocessing, most of the variables remains controled, include:

- Model Architectures
  - RPN (neck of Faster RCNN)
  - classification heads: 1 (background) + 10 (0~9) = 11 heads
- Training Hyerparamerter
  - lr: 5e-3 (resnet50) / 1e-4 (swin)
  - batch size: 4 (resnet50) / 2 (swin)
  - optimizer: SGD
  - schedular: stepLR
    - step size: 10
    - gamma: 0.1
    - weight decay: 5e-4
  - patience: 6

### Backbone

Backbone in Faster R-CNN is for initial feature extraction. **ResNet50** and **Swin-Transformer** was seleted for comparison. For ResNet50, I used `fasterrcnn_resnet50_fpn` and its pretrained weights from PyTorch. As for Swin, I downloaded `swinv2_base_window8_256` from timm with its pretrained weights as well.

### Data Preprocessing

To cater the necessity of different model, especially the input size, data preprocess is necessary. The input image size for these 2 models are pretty different.

- `fasterrcnn_resnet50_fpn`: has built-in object called `GeneralizedRCNNTransform` that would help programmers to proportionally scaling the image.
