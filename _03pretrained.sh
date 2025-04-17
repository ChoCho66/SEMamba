conda activate /disk4/chocho/speechbrain/.speechbrain
cd /disk4/chocho/SEMamba

taskset -c 46-47 nice -n 46 python inference.py \
   --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy \
   --output_folder _results-SEMamba \
   --checkpoint_file ckpts/SEMamba_advanced.pth  \
   --config recipes/SEMamba_advanced/SEMamba_advanced.yaml \
   --post_processing_PCS False \
   
   # --input_folder /mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/ \
   # --input_folder /home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016/noisy_testset_wav \
# CUDA_VISIBLE_DEVICES='1'