import glob
import os
import argparse
import json
import torch
import librosa
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import SEMamba
from models.pcs400 import cal_pcs
import soundfile as sf

from utils.util import (
    load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

from torchinfo import summary
import torch.nn as nn
from c66 import pp, pps

h = None
device = None

# chunk_frame=3500  _start = 65000  崩潰
# 聽感
# chunk_frame 
# 36000 OK
# 26000 OK
# 16000 OK
# 9000 OK
# 6000 OK
# 5500 OK
# 5000 ?
# chunk_sec
# 0.3 OK

import numpy as np

# chunk_hop_size = chunk_frame // 2  
# chunk_sec 
# 0.4 OK 
# 0.3 OK
# 有點崩潰
def inference_chunk_combine(args, device, chunk_sec=0.2):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    def process_audio_chunks(noisy_wav, chunk_frame, chunk_hop_size, sampling_rate, device, model, n_fft, hop_size, win_size, compress_factor, args):
        # 將 noisy_wav 轉換為 Tensor
        noisy_wav_tensor = torch.FloatTensor(noisy_wav).to(device)
        
        # 計算總長度和 chunk 數量
        total_length = len(noisy_wav_tensor)
        
        # 用於儲存最終輸出的緩衝區
        output_buffer = torch.zeros(total_length, device=device)
        weight_buffer = torch.zeros(total_length, device=device)  # 用於正規化重疊區域
        
        # 創建 Hann 窗口
        window = torch.hann_window(chunk_frame, device=device)
        
        # 正規化因子（對整個音訊計算一次）
        norm_factor = torch.sqrt(total_length / torch.sum(noisy_wav_tensor ** 2.0)).to(device)
        
        # 計算 chunk 的起始位置
        start_positions = list(range(0, total_length - chunk_frame + 1, chunk_hop_size))
        if not start_positions or start_positions[-1] + chunk_frame < total_length:
            start_positions.append(max(0, total_length - chunk_frame))
        
        for start in start_positions:
            # 提取 chunk
            end = min(start + chunk_frame, total_length)
            chunk = noisy_wav_tensor[start:end]
            
            # 如果 chunk 長度不足 chunk_frame，填充到 chunk_frame
            if len(chunk) < chunk_frame:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_frame - len(chunk)), mode='constant', value=0)
            
            # 應用正規化
            chunk_tensor = (chunk * norm_factor).unsqueeze(0)
            
            # 應用窗口函數
            chunk_tensor = chunk_tensor * window
            
            # STFT 處理
            try:
                noisy_amp, noisy_pha, noisy_com = mag_phase_stft(chunk_tensor, n_fft, hop_size, win_size, compress_factor)
            except RuntimeError as e:
                print(f"STFT failed for chunk at {start} with length {len(chunk)}: {e}")
                continue
            
            # 模型推理
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            
            # 逆 STFT
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            
            # 移除正規化
            audio_g = audio_g / norm_factor
            
            # 應用窗口函數
            audio_g = audio_g * window
            
            # 裁剪到原始長度並添加到輸出緩衝區
            audio_g = audio_g.squeeze()[:end - start]
            output_buffer[start:end] += audio_g
            weight_buffer[start:end] += window[:end - start]
        
        # 正規化重疊區域
        weight_buffer = torch.clamp(weight_buffer, min=1e-10)  # 避免除以零
        final_audio = (output_buffer / weight_buffer).cpu().numpy()
        
        # 確保輸出長度與輸入一致
        final_audio = final_audio[:total_length]
        
        # 後處理
        if args.post_processing_PCS:
            final_audio = cal_pcs(final_audio)
        
        return final_audio

    # 主處理迴圈
    chunk_frame = int(chunk_sec * 16000)
    chunk_hop_size = chunk_frame // 2  # 預設 hop size 為 chunk_frame 的一半，確保 50% 重疊
    with torch.no_grad():
        for i, fname in enumerate(os.listdir(args.input_folder)[:100]):
            print(fname, args.input_folder)
            
            # 載入音訊
            noisy_wav, _ = librosa.load(os.path.join(args.input_folder, fname), sr=sampling_rate)
            
            # 檢查 chunk_frame 是否合理
            if chunk_frame < win_size:
                print(f"Warning: chunk_frame ({chunk_frame}) is smaller than win_size ({win_size}). Adjusting chunk_frame.")
                chunk_frame = win_size
                chunk_hop_size = chunk_frame // 2
            
            # 處理音訊（分塊）
            try:
                processed_audio = process_audio_chunks(
                    noisy_wav, chunk_frame, chunk_hop_size, sampling_rate, device, model,
                    n_fft, hop_size, win_size, compress_factor, args
                )
            except RuntimeError as e:
                print(f"Failed to process {fname}: {e}")
                continue
            
            # 準備輸出檔案
            output_file = os.path.join(args.output_folder, fname)
            fname1 = fname.split("wav")[0] + "1.wav"
            output_file1 = os.path.join(args.output_folder, fname1)
            
            # 儲存處理後的音訊和原始音訊
            sf.write(output_file, processed_audio, sampling_rate, subtype='FLOAT')
            sf.write(output_file1, noisy_wav, sampling_rate, subtype='FLOAT')
            
def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='/mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/')
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='results')
    # parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--checkpoint_file', default='/disk4/chocho/SEMamba/ckpts/SEMamba_advanced.pth')
    parser.add_argument('--post_processing_PCS', default=False)
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        #device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")

    inference_chunk_combine(args, device)

if __name__ == '__main__':
    main()

