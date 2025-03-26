# Selected Topics in Visual Recognition using Deep Learning HW1 Report
- **Name:** 袁孟華
- **Student ID:** 110550004
- - **Github Repo:** <https://github.com/mm1719/CV_DL>
## How to run
Simply run the `run.sh`. It includes training, testing and visualizing.  
Results of one expiriment would store in `outputs/<timestamp>`
## Introduction
This report detailed describes the tactics I have approached to training the resnet-based classification model on this iNaturalist-like dataset. Initially I attempted to use Supervised Contrastive Learning (SupCon) with a downstream classfier, but found it time consuming and the results were not satisfiable. However, it gave me an insight on how the high-dimensional encoded results clustered on 2D t-SNE graph, which aided me in the later 2 experiments. As a result, I decided to simplify the classification process by directly training the whole resnet-based model with classifier. In this report, I would mainly describe the methods I have implemented for the latter approach. One was naive approach resulting in 0.91 accuracy. The other was enhanced measure using various tricks (label-smoothing, data-augmentation, etc) leading to 0.94 accuracy. 

## Method
### Data Preprocessing
#### Method 1 (naive)
* Resize all the images (train, val, test) to 224 x 224.
* Normalize with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].
#### Method 2 (enhanced)
* Resize all the images (train, val, test) to 384 x 384.
* Normalize with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].
* Augmentation include RandomHorizontalFlip and RandomRotation, each image have 3 augmentation, so the dataset was 3 times larger than the original

### Model Architecture and Hyperparameters
#### Method 1 (naive)
* seresnext101d_32x8d from timm
* batch_size = 32, lr = 0.05 * (batch_size / 256), momentum = 0.9, weight_decay = 1e-4, epochs = 50, warmup_epochs = 5
#### Method 2 (enhanced)
* seresnext101d_32x8d from timm
* batch_size = 16, lr = 0.05 * (batch_size / 256), momentum = 0.9, weight_decay = 1e-4, epochs = 50, warmup_epochs = 5, label_smoothing = 0.15
### Model Training and Saving
code snippet:
```python
for epoch in range(1, config.epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, logger, writer)
    val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, logger, writer)

    if epoch <= config.warmup_epochs:
        scheduler.step()
    else:
        scheduler_after.step()

    logger.info(f"Epoch {epoch} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        early_stop_counter = 0
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "best_model.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"[儲存模型] Encoder 權重儲存於: {model_path}")
    else:
        early_stop_counter += 1
        logger.info(f"EarlyStop counter: {early_stop_counter}/{config.early_stopping_patience}")
        if early_stop_counter >= config.early_stopping_patience:
            logger.info(f"早停於 Epoch {epoch}（最佳為 Epoch {best_epoch}）")
            break
```
### Inference and Prediction Output
All the results of a whole process would store in a single folder named by timestamp under a folder called outputs. Looks like this:
```
└─outputs
    ├─20250325-174438
    │  └─logs
    │  best_model.pth
    │  confusion_matrix.png
    │  log_20250325-174442.txt
    │  log_20250325-183514.txt
    │  log_20250325-183532.txt
    │  log_20250325-183645.txt
    │  prediction.csv
    │  tsne.png
```
## Results
#### Method 1 (naive)
* Val Acc: 0.89, Test Acc: 0.91
#### Method 2 (enhanced)
* Val Acc: 0.92, Test Acc: 0.94
### Performance Summary
Since the dataset is actually pretty imbalance and the variance per each class is pretty wide (e.g. a sort of butterfly includes caterpillar and its adulthood), using label-smoothing and data augmentation seemed to better the performance.
## Additional Experiment
As mentioned in introduction, I tried to use supcon but resulted in merely 0.88 accuracy. At first I thought this was a fancy and reasonable method, also some papers told the performance is better. I though the main reason is for the lack of data (or data augmentation), all the related code is in `old`.
## References
- **timm Library:**  
  [rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models)
- **RegNet Architecture:**  
  [RegNet: Design Space for High-Performance Image Classification](https://arxiv.org/abs/2003.13678)
- **SEResNeXt Architecture:**  
  [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- **iNaturalist Dataset:**  
  [iNaturalist](https://www.inaturalist.org/)
- **Contrastive Learning:**  
  [Contrastive Learning](https://youtu.be/1pvxufGRuW4?si=wqHPiqqMvoAH65Ut)
