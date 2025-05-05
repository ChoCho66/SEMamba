#!/bin/bash
source /disk4/chocho/miniconda3/etc/profile.d/conda.sh
conda activate /disk4/chocho/speechbrain/.speechbrain
cd /disk4/chocho/SEMamba

# YAML 檔案路徑
CONFIG_FILE="recipes/SEMamba_advanced_PCS/SEMamba_advanced_PCS.yaml"

# 使用 Go 版本的 yq 提取 model_cfg 中的參數
HID_FEATURE=$(yq eval '.model_cfg.hid_feature' $CONFIG_FILE)
NUM_TFMAMBA=$(yq eval '.model_cfg.num_tfmamba' $CONFIG_FILE)
D_STATE=$(yq eval '.model_cfg.d_state' $CONFIG_FILE)
D_CONV=$(yq eval '.model_cfg.d_conv' $CONFIG_FILE)
EXPAND=$(yq eval '.model_cfg.expand' $CONFIG_FILE)
DEPTH=$(yq eval '.model_cfg.depth' $CONFIG_FILE)

# 檢查是否成功提取參數
if [[ -z "$HID_FEATURE" ]]; then
    echo "Error: Failed to extract hid_feature from $CONFIG_FILE"
    exit 1
fi

# 動態生成 exp_name 的後半部分
EXP_NAME_SUFFIX="dep${DEPTH}_h${HID_FEATURE}_tf${NUM_TFMAMBA}_ds${D_STATE}_dc${D_CONV}_ex${EXPAND}"

# 組合完整的 exp_name
CURRENT_DATE=$(date +%Y%m%d)
# EXP_NAME="${CURRENT_DATE}-PCS-${EXP_NAME_SUFFIX}"
EXP_NAME="${EXP_NAME_SUFFIX}"

# 執行訓練
taskset -c 44-47 nice -n 46 python train.py \
  --config $CONFIG_FILE \
  --exp_folder 'exp/VCTK+THCHS/' \
  --exp_name "$EXP_NAME"