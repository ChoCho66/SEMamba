#!/bin/bash
source /disk4/chocho/miniconda3/etc/profile.d/conda.sh
conda activate /disk4/chocho/speechbrain/.speechbrain
cd /disk4/chocho/SEMamba

output_folder="/disk4/chocho/SEMamba/_test_enhence_chunk_combine/ALL"
taskset -c 44-47 nice -n 46 python inference.py \
  --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy \
  --output_folder "${output_folder}" \
  --checkpoint_file /disk4/chocho/SEMamba/exp/20250330-SEMamba_v1_PCS/g_00093000.pth \
  --config /disk4/chocho/SEMamba/exp/20250330-SEMamba_v1_PCS/config.yaml \
  --post_processing_PCS True \

# chunk_sec 0.2 會吱吱聲

# --input_folder /disk4/chocho/SEMamba/_test_noisy_chunk \
    # --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy \
