# Selected Topics in Visual Recognition using Deep Learning HW3 Report

- **Name:** 袁孟華
- **Student ID:** 110550004
- **Github Repo:** <https://github.com/mm1719/CV_DL>

## How to run

### Train

`>./run.sh --mode train` includes all training and evaluation phases. After the training procedures of one experiment are done, the result would store in `outputs/<timestamp>`

`>./run.sh train default` or `>./run.sh train weakly` includes all training and evaluation phases. After the training procedures of one experiment are done, the result would store in `outputs/` with this sort of structure:

### Test

Any trained model can be picked for testing by `>./run.sh --mode test -- timestamp <timestamp>`

## Introduction

Image restoration tasks such as deraining and desnowing aim to recover clean visual information from degraded inputs, often affected by various weather conditions. Recently, transformer-based architectures have shown promising results in addressing such challenges due to their ability to capture long-range dependencies. In this experiment, we investigate the performance of the PromptIR model—a prompt-driven image restoration transformer—trained on a real-world dataset composed of synthetic rain and snow degradation. Our goal is to evaluate the model's ability to generalize across both degradation types and measure performance through objective metrics such as PSNR and L1 loss.

## Method

### Data Preprocess

The dataset comprises 3,200 degraded-clean image pairs (1,600 rainy and 1,600 snowy images), each with a resolution of 256×256 pixels. The dataset is split into training and validation subsets at an 80:20 ratio using a fixed random seed to ensure reproducibility. To enhance data diversity and prevent overfitting, we apply random cropping to 64×64 patches during training. All images are transformed into tensors using PyTorch's ToTensor() without applying normalization, since pixel intensities are already bounded between 0 and 1. For validation and testing, we skip cropping to maintain the original spatial content.

### Model Architecture

We adopt the PromptIR model with a decoding head enabled. PromptIR integrates Transformer blocks with spatial prompt routing to effectively capture hierarchical representations of visual structures. The encoder processes multi-scale features, while prompt tokens guide the decoder to adaptively refine the output based on content-aware attention. The final output is a reconstructed RGB image.

The model is trained using a supervised L1 loss (mean absolute error), which is robust to outliers and better preserves fine-grained visual details compared to MSE loss.

### Hyperparameters

Detailed setting can be found in `outputs/timestamp/config.yaml`:

| Parameter        | Value              |
| ---------------- | ------------------ |
| Batch size       | 32                 |
| Crop size        | 64 × 64            |
| Optimizer        | AdamW              |
| Learning rate    | $4 \times 10^{-4}$ |
| Scheduler        | Cosine Annealing   |
| Epochs           | 400                |
| Validation ratio | 0.2                |
| Loss function    | L1 loss            |
| PSNR evaluation  | Enabled            |

## Result

Train Loss:

![alt text](train_loss.png)

Val Loss:

![alt text](val_loss.png)

Val PSNR:

![alt text](PSNR.png)

Note: 400 (epoch) x 2 (iterations/epoch) = 800 steps
**Test PSNR: 30.11**

## Additional Expirements and Results

To explore the potential benefits of increasing model capacity, we doubled the number of prompt vectors in each prompt generation block from 5 to 10. While this modification aimed to enhance the model’s ability to capture diverse restoration patterns, the results indicated a decline in generalization performance. Specifically, the test PSNR dropped from 30.11 dB to 29.42 dB. This suggests that the additional prompt vectors may have led to overfitting, especially given the limited size of the training dataset. The increase in model complexity might have enabled it to memorize training samples rather than learn more generalizable restoration patterns. These findings highlight the importance of balancing model capacity and dataset scale when tuning prompt-related hyperparameters.

## Reference

* Potlapalli, V., Zamir, S. W., Khan, S., & Khan, F. S. (2023, June 22). ProMpTIR: Prompting for All-in-One Blind Image Restoration. arXiv.org. https://arxiv.org/abs/2306.13090
