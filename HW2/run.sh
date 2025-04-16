#!/bin/bash

# 產生 timestamp（與 config.py 中一致格式）
timestamp=$(date +%Y%m%d-%H%M%S)
output_dir="outputs/$timestamp"
mkdir -p "$output_dir"

# 訓練模型
echo "🚀 開始訓練模型..."
python main.py train --output-dir "$output_dir"

# 推論測試集
echo "🔍 開始進行推論 (test)..."
python main.py test --output-dir "$output_dir" --model-path "$output_dir/best_model.pth"

# 評估 pred.json 對 valid.json 的 mAP
echo "📊 評估預測結果 (eval)..."
python main.py eval --output-dir "$output_dir" --model-path "$output_dir/best_model.pth"

echo "✅ 完成所有流程，輸出位於: $output_dir"
