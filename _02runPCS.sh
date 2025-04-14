taskset -c 46-47 nice -n 46 python train.py \
  --config recipes/SEMamba_advanced_PCS/SEMamba_advanced_PCS.yaml \
  --exp_folder 'exp' \
  --exp_name '20250403-SEMamba_v1_PCS-h32-tf2' \
