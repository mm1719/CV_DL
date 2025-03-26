#!/bin/bash

# 設定時間戳記（確保三階段共用同一批資料與資料夾）
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

# classifier 選擇（linear / mlp）
CLASSIFIER_TYPE="--classifier_type mlp"  # ✅ 你現在在測試 MLP

# 第一階段：特徵學習（SupCon）
echo "===== SupCon 表徵學習階段 ====="
python main.py --mode supcon --timestamp $TIMESTAMP $CONFIG_PATH

# 第二階段：訓練分類器（使用指定的 classifier）
echo "===== 分類器訓練階段 ====="
python main.py --mode classifier --timestamp $TIMESTAMP $CONFIG_PATH $CLASSIFIER_TYPE

# 第三階段：測試並輸出 prediction.csv
echo "===== 測試與結果輸出階段 ====="
python main.py --mode test --timestamp $TIMESTAMP $CONFIG_PATH

echo "✅ 三階段訓練完成，輸出位於 outputs/$TIMESTAMP"
