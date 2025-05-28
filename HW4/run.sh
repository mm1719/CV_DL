#!/bin/bash

MODE=$1
TS=$2

if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
  echo "❌ 請使用: bash run.sh [train|test] [timestamp?]"
  exit 1
fi

if [[ "$MODE" == "train" ]]; then
  TS=$(date +"%Y%m%d-%H%M%S")
  echo "🕒 新訓練 session timestamp: $TS"
else
  if [[ -z "$TS" ]]; then
    echo "❌ 測試模式必須指定 timestamp: bash run.sh test <timestamp>"
    exit 1
  fi
  echo "🔍 使用既有 timestamp: $TS"
fi

LOGDIR="outputs/$TS"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${MODE}.log"

echo "📁 Logging to: $LOGFILE"

python main.py --mode "$MODE" --timestamp "$TS" 2>&1
