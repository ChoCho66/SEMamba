#!/bin/bash

# 檢查 exp 資料夾是否存在
if [ ! -d "exp" ]; then
  echo "錯誤：exp 資料夾不存在！"
  exit 1
fi

# 遍歷 exp 資料夾中的所有子資料夾
for dir in exp/*/*/ ; do
  # 檢查資料夾是否存在 g_00001000.pth 檔案
  file="$dir/g_00001000.pth"
  if [ -f "$file" ]; then
    # 使用 ls -lh 獲取檔案大小（人類可讀格式）
    size=$(ls -lh "$file" | awk '{print $5}')
    echo "$(basename "$dir")"
    echo "model 大小: $size"
    echo "---------------------"
  else
    echo "$(basename "$dir")"
    echo "model 不存在"
    echo "---------------------"
  fi
done