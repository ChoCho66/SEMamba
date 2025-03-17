CUDA_VISIBLE_DEVICES='0' python inference.py \
   --input_folder /home/kycho/Speech-Enhancement-Code/datas/my-test-noisy \
   --output_folder results/my-test-noisy \
   --checkpoint_file ckpts/SEMamba_advanced.pth  \
   --config recipes/SEMamba_advanced/SEMamba_advanced.yaml \
   --post_processing_PCS False \
   
   # --input_folder /mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/ \
   # --input_folder /home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016/noisy_testset_wav \