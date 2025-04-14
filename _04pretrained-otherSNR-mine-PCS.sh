#!/bin/bash

# 定義 SNR 值的陣列
# SNR_VALUES=(2 1 0 -1 -5 -10)
# SNR_VALUES=(17.5 12.5 7.5 2.5)
# SNR_VALUES=(-15 -12 -10 -8 -5 -3 -1 0 1 2 2.5 7.5 12.5 17.5)
# SNR_VALUES=(-18 -15 -12 -9 -6 -3 0 3 6 9 12 15 18)
SNR_VALUES=(-18)

# 使用 for 迴圈遍歷每個 SNR 值
for SNR in "${SNR_VALUES[@]}"
do
    taskset -c 46-47 nice -n 46 python inference.py \
        --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k-SNR/noisy/${SNR} \
        --output_folder /disk4/chocho/_enhanced-wav/VCTK_DEMAND16k/SEMamba-PCS/${SNR} \
        --checkpoint_file /disk4/chocho/SEMamba/exp20250330-PCS/SEMamba_v1_PCS/g_00093000.pth \
        --config /disk4/chocho/SEMamba/exp20250330-PCS/SEMamba_v1_PCS/config.yaml \
        --post_processing_PCS True
done


   # --input_folder /mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/ \
   # --input_folder /home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016/noisy_testset_wav \
# CUDA_VISIBLE_DEVICES='1'
        # --checkpoint_file ckpts/SEMamba_advanced.pth \



        