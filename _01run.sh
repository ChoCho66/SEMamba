taskset -c 46-47 nice -n 46 python train.py \
  --config recipes/SEMamba_advanced/SEMamba_advanced.yaml \
  --exp_folder exp20250319 \
  --exp_name SEMamba_v1


# CUDA_VISIBLE_DEVICES='1' 