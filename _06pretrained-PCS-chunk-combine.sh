#!/bin/bash
source /disk4/chocho/miniconda3/etc/profile.d/conda.sh
conda activate /disk4/chocho/speechbrain/.speechbrain
cd /disk4/chocho/SEMamba

for chunk_sec in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
  for chunk_hop_ratio in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1; do
    output_folder="/disk4/chocho/SEMamba/_test_enhence_chunk_combine/${chunk_sec}-${chunk_hop_ratio}"
    taskset -c 44-47 nice -n 46 python inference-streaming.py \
      --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy \
      --output_folder "${output_folder}" \
      --checkpoint_file /disk4/chocho/SEMamba/exp/20250330-SEMamba_v1_PCS/g_00093000.pth \
      --config /disk4/chocho/SEMamba/exp/20250330-SEMamba_v1_PCS/config.yaml \
      --post_processing_PCS True \
      --chunk_sec "${chunk_sec}" \
      --chunk_hop_ratio "${chunk_hop_ratio}" \
      --num_sampled_ratio 1
  done
done

# chunk_sec 0.2 會吱吱聲

# --input_folder /disk4/chocho/SEMamba/_test_noisy_chunk \
    # --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy \
