#!/bin/bash

# 获取当前脚本所在目录
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$DIR")"

# 设置 Python 路径，确保可以导入主程序的模块
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 检查数据集是否存在
DATASET_PATH="$DIR/sft_dataset.jsonl"
if [ ! -f "$DATASET_PATH" ]; then
    echo "未发现数据集 $DATASET_PATH，正在生成中..."
    python3 "$DIR/generate_sft_data.py"
fi

# 开始训练
echo "开始 special tokens 微调训练..."
python3 "$DIR/train_sft.py"
