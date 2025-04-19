#!/bin/bash
source /disk4/chocho/miniconda3/etc/profile.d/conda.sh
conda activate /disk4/chocho/speechbrain/.speechbrain
cd /disk4/chocho/SEMamba

taskset -c 44-47 nice -n 46 python inference.py \
  --input_folder /disk4/chocho/SEMamba/_test_feature_map \
  --checkpoint_file /disk4/chocho/SEMamba/exp/20250330-SEMamba_v1_PCS/g_00093000.pth \
  --config /disk4/chocho/SEMamba/exp/20250330-SEMamba_v1_PCS/config.yaml \
  --post_processing_PCS True \
