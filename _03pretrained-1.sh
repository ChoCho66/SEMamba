taskset -c 44-47 nice -n 46 python inference.py \
   --input_folder _test_noisy \
   --output_folder _results/20250402-SEMamba_v1_PCS-tf2/ \
   --checkpoint_file exp/20250402-SEMamba_v1_PCS-tf2/g_00084000.pth  \
   --config exp/20250402-SEMamba_v1_PCS-tf2/config.yaml \
   --post_processing_PCS True \
   
   # --input_folder /mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/ \
   # --input_folder /home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016/noisy_testset_wav \
# CUDA_VISIBLE_DEVICES='1'






   # --output_folder _results/SEMamba-PCS-h32-tf2/ \
   # --checkpoint_file exp/20250403-SEMamba_v1_PCS-h32-tf2/g_00021000.pth  \
   # --config exp/20250403-SEMamba_v1_PCS-h32-tf2/config.yaml \
