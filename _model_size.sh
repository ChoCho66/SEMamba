#!/bin/bash

# 檢查 exp 資料夾是否存在
if [ ! -d "exp" ]; then
  echo "錯誤：exp 資料夾不存在！"
  exit 1
fi

# 遍歷 exp 下的所有 dataset 資料夾
for dataset_dir in exp/*/ ; do
  # 確認是資料夾
  if [ -d "$dataset_dir" ]; then
    # 遍歷 dataset 資料夾下的子資料夾
    for sub_dir in "${dataset_dir}"*/ ; do
      # 檢查資料夾是否存在 g_00001000.pth 檔案
      file="${sub_dir}g_00001000.pth"
      # 獲取 dataset-type 格式（僅 dataset-type，去掉 exp/）
      dir_name=$(basename "$dataset_dir")
      if [ -f "$file" ]; then
        # 使用 ls -lh 獲取檔案大小（人類可讀格式）
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "$dir_name/$(basename "$sub_dir")"
        echo "model 大小: $size"
        echo "---------------------"
      else
        echo "$dir_name/$(basename "$sub_dir")"
        echo "model 不存在"
        echo "---------------------"
      fi
    done
  fi
done