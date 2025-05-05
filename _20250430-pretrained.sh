source /disk4/chocho/miniconda3/etc/profile.d/conda.sh
conda activate /disk4/chocho/speechbrain/.speechbrain
cd /disk4/chocho/SEMamba

taskset -c 46-47 nice -n 46 python inference.py \
   --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy \
   --output_folder _results/20250502-PCS-h32_tf1_ds16_dc4_ex2 \
   --checkpoint_file exp/20250502-PCS-h32_tf1_ds16_dc4_ex2/g_00075000.pth  \
   --config exp/20250502-PCS-h32_tf1_ds16_dc4_ex2/config.yaml \
   --post_processing_PCS True \
   
   # --mode inference \
   
   # --input_folder /mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/ \
   # --input_folder /home/kycho/Speech-Enhancement-Code/datas/VCTK-Valentini-Botinhao-2016/noisy_testset_wav \
# CUDA_VISIBLE_DEVICES='1'