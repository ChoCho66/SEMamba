source /disk4/chocho/miniconda3/etc/profile.d/conda.sh
conda activate /disk4/chocho/speechbrain/.speechbrain
cd /disk4/chocho/SEMamba

taskset -c 40-47 nice -n 46 python inference-ablation.py \
   --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy \
   --output_folder _results/ablation/VCTK-400/dep4_h64_tf4_ds16_dc4_ex4/tf-skip3 \
   --checkpoint_file exp/VCTK-400/dep4_h64_tf4_ds16_dc4_ex4/g_00093000.pth  \
   --config exp/VCTK-400/dep4_h64_tf4_ds16_dc4_ex4/config.yaml \
   --post_processing_PCS True \
   
   
   # --output_folder _results/ablation/tf/dep4_h32_tf4_ds32_dc3_ex4/skip0 \
   # --checkpoint_file exp/ablation-tf/dep4_h32_tf4_ds32_dc3_ex4/g_00024000.pth  \


   # --input_folder /mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/ \
   # --input_folder /home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016/noisy_testset_wav \
# CUDA_VISIBLE_DEVICES='1'