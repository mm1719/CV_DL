#!/bin/bash

# Step 0: 產生 timestamp
timestamp=$(date +"%Y%m%d-%H%M%S")

echo "🚀 開始訓練與分析流程！timestamp = $timestamp"
echo "=============================="

# Step 1: 訓練
echo "[1/4] 訓練模型"
python main.py --mode train --timestamp $timestamp

# Step 2: 預測測試集
echo "[2/4] 預測測試集"
python main.py --mode test --timestamp $timestamp

# Step 3: t-SNE 可視化
echo "[3/4] t-SNE 視覺化"
python main.py --mode tsne --timestamp $timestamp

# Step 4: 混淆矩陣
echo "[4/4] 混淆矩陣"
python main.py --mode cm --timestamp $timestamp

echo "✅ 全部完成！輸出儲存在 outputs/$timestamp"
