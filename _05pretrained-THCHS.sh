#!/bin/bash

# 定義 SNR 值的陣列
# SNR_VALUES=(2 1 0 -1 -5 -10)
# SNR_VALUES=(17.5 12.5 7.5 2.5)
# SNR_VALUES=(-15 -12 -10 -8 -5 -3 -1 0 1 2 2.5 7.5 12.5 17.5)
# SNR_VALUES=(-18 -15 -12 -9 -6 -3 0 3 6 9 12 15 18)
noises=(cafe car white)

for noise in "${noises[@]}"
do
    taskset -c 46-47 nice -n 46 python inference.py \
        --input_folder /disk4/chocho/_datas/THCHS-30/test-noise1/${noise} \
        --output_folder /disk4/chocho/_enhanced-wav/THCHS-30/SEMamba/${noise} \
        --checkpoint_file ckpts/SEMamba_advanced.pth \
        --config recipes/SEMamba_advanced/SEMamba_advanced.yaml \
        --post_processing_PCS False
done

# --input_folder /disk4/chocho/_datas/THCHS-30/test-noise/0db/${noise} \
   # --input_folder /mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/ \
   # --input_folder /home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016/noisy_testset_wav \
# CUDA_VISIBLE_DEVICES='1'