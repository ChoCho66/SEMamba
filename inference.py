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

def inference(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    with torch.no_grad():
        # You can use data.json instead of input_folder with:
        # ---------------------------------------------------- #
        # with open("data/test_noisy.json", 'r') as json_file:
        #     test_files = json.load(json_file)
        # for i, fname in enumerate( test_files ): 
        #     folder_path = os.path.dirname(fname)
        #     fname = os.path.basename(fname)
        #     noisy_wav, _ = librosa.load(os.path.join( folder_path, fname ), sr=sampling_rate)
        #     noisy_wav = torch.FloatTensor(noisy_wav).to(device)
        # ---------------------------------------------------- #
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(args.output_folder, fname)

            if args.post_processing_PCS == True:
                audio_g = cal_pcs(audio_g.squeeze().cpu().numpy())
                # sf.write(output_file, audio_g, sampling_rate, 'PCM_16')
                sf.write(output_file, audio_g, sampling_rate, subtype='FLOAT')
            else:
                # sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')
                sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, subtype='FLOAT')

def inference_feature_map(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            model.get_feature_map(fname, noisy_amp, noisy_pha)
            

def inference_FlopCountAnalysis(args, device):
    from torch.profiler import profile, record_function, ProfilerActivity
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])
    model.eval()

    import pandas as pd
    with torch.no_grad():
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)

            # 使用 profiler 追蹤 FLOPs
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_flops=True  # 啟用 FLOPs 計算
            ) as prof:
                with record_function("model_inference"):
                    output = model(noisy_amp, noisy_pha)
            # 輸出 FLOPs 結果
            # print(prof.key_averages().table(sort_by="flops", row_limit=10))
            events = prof.key_averages()
            # 把想要的欄位整理出來
            rows = []
            for evt in events:
                rows.append({
                    "Name": evt.key,
                    "CPU Mem": evt.cpu_memory_usage,              # 單位: bytes
                    "Self CPU Mem": evt.self_cpu_memory_usage,
                    "CUDA Mem": evt.cuda_memory_usage,
                    "Self CUDA Mem": evt.self_cuda_memory_usage,
                    "Total KFLOPs": int(evt.flops / 1e3) if evt.flops else 0  # 轉為整數避免科學記號
                })
            # 轉為 DataFrame，依 flops 排序
            df = pd.DataFrame(rows)
            df_sorted = df.sort_values(by="Total KFLOPs", ascending=False).head(10)
            # 顯示結果
            print(df_sorted.to_string(index=False))            

            # # 計算 FLOPs 和參數量
            # flops, params = get_model_complexity_info(
            #     model,
            #     input_res,
            #     as_strings=True,
            #     print_per_layer_stat=True,
            #     verbose=True
            # )
            # # 輸出結果
            # print(f"計算複雜度: {flops}")
            # print(f"參數量: {params}")
            break

import random
def inference_chunk(args, device, chunk_sec=0.25):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    chunk_frame = int(chunk_sec * 16000)
    # 1:16000 = x:chunk_frame
    with torch.no_grad():
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            _start = random.randint(1000, len(noisy_wav) - chunk_frame)
            noisy_wav = noisy_wav[_start:]
            noisy_wav = noisy_wav[:chunk_frame]
            pps(noisy_wav)
            
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(args.output_folder, fname)
            fname1 = fname.split("wav")[0] + "1.wav"
            output_file1 = os.path.join(args.output_folder, fname1)

            pps(audio_g, noisy_wav)

            if args.post_processing_PCS == True:
                audio_g = cal_pcs(audio_g.squeeze().cpu().numpy())
                # sf.write(output_file, audio_g, sampling_rate, 'PCM_16')
                sf.write(output_file, audio_g, sampling_rate, subtype='FLOAT')
                sf.write(output_file1, noisy_wav, sampling_rate, subtype='FLOAT')
            else:
                # sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')
                sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, subtype='FLOAT')
                sf.write(output_file1, noisy_wav.squeeze().cpu().numpy(), sampling_rate, subtype='FLOAT')


def inference_chunk_combine(args, device, chunk_sec=0.3):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()

    # 1:16000 = x:chunk_frame
    chunk_frame = int(chunk_sec * 16000)
    with torch.no_grad():
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(args.output_folder, fname)
            fname1 = fname.split("wav")[0] + "1.wav"
            output_file1 = os.path.join(args.output_folder, fname1)

            if args.post_processing_PCS == True:
                audio_g = cal_pcs(audio_g.squeeze().cpu().numpy())
                # sf.write(output_file, audio_g, sampling_rate, 'PCM_16')
                sf.write(output_file, audio_g, sampling_rate, subtype='FLOAT')
                sf.write(output_file1, noisy_wav, sampling_rate, subtype='FLOAT')
            else:
                # sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')
                sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, subtype='FLOAT')
                sf.write(output_file1, noisy_wav.squeeze().cpu().numpy(), sampling_rate, subtype='FLOAT')


def show_model(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    model.eval()
    
    with torch.no_grad():
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            print(noisy_wav.shape)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            print()
            print()
            print()
            print()
            pp(noisy_amp.shape, noisy_pha.shape)
            # 獲取完整 summary
            summary_str = summary(model, input_size=[noisy_amp.shape, noisy_pha.shape], depth=5, col_names=("input_size", "output_size", "num_params"), verbose=0)
            # 將輸出寫入 txt
            with open("model_summary.txt", "w") as f:
                f.write(str(summary_str))
            print()
            print()
            print()
            print()
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            pp(amp_g.shape, pha_g.shape)
            audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_g = audio_g / norm_factor
            pp(audio_g.shape)
            break
    
    # # 找到並顯示第一層參數
    # print("First layer parameters:")
    # for name, param in model.named_parameters():
    #     print(f"Parameter: {name}, Shape: {param.shape}, Type: {param.dtype}")
    #     # print(f"Values:\n{param.data}\n")
    #     # break  # 只顯示第一層，然後跳出迴圈
    
def show_model2(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    print(111)
    print(model.TSMamba[0].time_mamba.forward_blocks[0].mixer.A_log)
    print(222)
    print(nnn)


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

    # inference(args, device)
    # inference_feature_map(args, device)
    inference_FlopCountAnalysis(args, device)
    # inference_chunk(args, device)
    # show_model(args, device)
    # show_model2(args, device)


if __name__ == '__main__':
    main()

