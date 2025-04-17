#!/bin/bash

taskset -c 46-47 nice -n 46 python inference.py \
    --input_folder /disk4/chocho/_datas/VCTK_DEMAND16k/test/noisy \
    --output_folder /disk4/chocho/SEMamba/_test_enhence_chunk \
    --checkpoint_file /disk4/chocho/SEMamba/exp/20250330-SEMamba_v1_PCS/g_00093000.pth \
    --config /disk4/chocho/SEMamba/exp/20250330-SEMamba_v1_PCS/config.yaml \
    --post_processing_PCS True



# --input_folder /disk4/chocho/SEMamba/_test_noisy_chunk \
