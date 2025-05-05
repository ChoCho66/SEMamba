import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import argparse
import json
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dataloaders.dataloader_vctk import VCTKDemandDataset
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.loss import pesq_score, phase_losses
from models.discriminator import MetricDiscriminator, batch_pesq
from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

from c66 import pp, pps

torch.backends.cudnn.benchmark = True

def setup_optimizers(models, cfg):
    """Set up optimizers for the models."""
    generator, discriminator = models
    learning_rate = cfg['training_cfg']['learning_rate']
    betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])

    optim_g = optim.AdamW(generator.parameters(), lr=learning_rate, betas=betas)
    optim_d = optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=betas)

    return optim_g, optim_d

def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=last_epoch)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=last_epoch)

    return scheduler_g, scheduler_d

from pathlib import Path
def load_json_files(json_paths):
    """Load and merge multiple JSON files."""
    merged_data = []
    for json_path in json_paths:
        json_path = Path(json_path)  # 確保路徑正確處理
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)  # 假設 JSON 內容是列表，直接合併
                else:
                    merged_data.append(data)  # 如果是其他格式，根據需要處理
        else:
            print(f"Warning: JSON file {json_path} not found.")
    return merged_data


def create_dataset1(cfg, train=True, split=True, device='cuda:0'):
    """Create dataset based on configuration."""
    # 根據 train 選擇對應的 JSON 檔案列表
    clean_json_paths = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    noisy_json_paths = cfg['data_cfg']['train_noisy_json'] if train else cfg['data_cfg']['valid_noisy_json']
    
    # 確保輸入是列表，如果是單個檔案，轉為單元素列表
    if isinstance(clean_json_paths, str):
        clean_json_paths = [clean_json_paths]
    if isinstance(noisy_json_paths, str):
        noisy_json_paths = [noisy_json_paths]
    
    # 讀取並合併 JSON 檔案內容
    clean_data = load_json_files(clean_json_paths)
    noisy_data = load_json_files(noisy_json_paths)
    
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False
    
    print(noisy_data[:6])
    print(clean_data[:6])
    print()
    print(noisy_data[-6:])
    print(clean_data[-6:])
    exit()
    # print((clean_data[:6]))
    # print((clean_data[-6:]))
    return VCTKDemandDataset(
        clean_json=clean_data,
        noisy_json=noisy_data,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs
    )
    
def create_dataset(cfg, train=True, split=True, device='cuda:0'):
    """Create dataset based on cfguration."""
    clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    noisy_json = cfg['data_cfg']['train_noisy_json'] if train else cfg['data_cfg']['valid_noisy_json']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False
    
    return VCTKDemandDataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs
    )

def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_cfg']['training_epochs'])
        batch_size = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers'] if train else 1

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True if train else False
    )


def train(cfg):
    trainset = create_dataset1(cfg, train=True, split=True, device='cpu')
    exit()

# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument('--exp_name', default='SEMamba_advanced')
    parser.add_argument('--config', default='recipes/SEMamba_advanced/SEMamba_advanced.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)

if __name__ == '__main__':
    main()
