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

def inference_ablation(args, device):
    cfg = load_config(args.config)
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    model = SEMamba(cfg).to(device)
    state_dict = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(state_dict['generator'])

    os.makedirs(args.output_folder, exist_ok=True)

    model.eval()
    
    feature_encoder = model.dense_encoder
    mask_decoder = model.mask_decoder
    phase_decoder = model.phase_decoder
    TSMamba = model.TSMamba

    from einops import rearrange
    with torch.no_grad():
        for i, fname in enumerate(os.listdir( args.input_folder )):
            print(fname, args.input_folder)
            noisy_wav, _ = librosa.load(os.path.join( args.input_folder, fname ), sr=sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)

            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_mag, noisy_pha, noisy_com = mag_phase_stft(noisy_wav, n_fft, hop_size, win_size, compress_factor)
            # noisy_mag, noisy_pha: [1, F, T] = [1, n_fft//2, len_wav//hop_size]
            
            # feature_encoder
            # Reshape inputs
            noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B F T] -> [B, 1, T, F]
            noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B F T] -> [B, 1, T, F]
            
            # Concatenate magnitude and phase inputs
            x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]
            
            # Feature Encoder
            x = feature_encoder(x)
            # [B, 2, T, F] -> [B, h, T, F//2]
            
            # TF-Mamba
            # TSMamba is a instance of TFMambaBlock
            for idx, block in enumerate(TSMamba):
            # for block in TSMamba:
                if idx == 3:  # 跳過第二層
                    continue
            
                b, c, t, f = x.size()
                # b, c, t, f = [1, h, T, F]
                
                x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
                
                x = block.tlinear( block.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
                # [F, T, h] -> [F, T, h]
                
                x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
                
                x = block.flinear( block.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
                # [T, F, h] -> [T, F, h]
                
                x = x.view(b, t, f, c).permute(0, 3, 1, 2)

            # Mag, Pha Decoder
            denoised_mag = rearrange(mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
            # [1, 1, T, F] * [1, 1, T, F] = [1, 1, T, F] -> [1, F, T, 1] -> [1, F, T]
            
            denoised_pha = rearrange(phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)
            # [1, 1, T, F] -> [1, F, T, 1] -> [1, F, T]
            
            audio_g = mag_phase_istft(denoised_mag, denoised_pha, n_fft, hop_size, win_size, compress_factor)
            # [1, ~L]
            
            audio_g = audio_g / norm_factor

            output_file = os.path.join(args.output_folder, fname)

            if args.post_processing_PCS == True:
                audio_g = cal_pcs(audio_g.squeeze().cpu().numpy())
                # sf.write(output_file, audio_g, sampling_rate, 'PCM_16')
                sf.write(output_file, audio_g, sampling_rate, subtype='FLOAT')
            else:
                # sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, 'PCM_16')
                sf.write(output_file, audio_g.squeeze().cpu().numpy(), sampling_rate, subtype='FLOAT')

def main():
    print('Initializing Inference Process..')
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default='/mnt/e/Corpora/noisy_vctk/noisy_testset_wav_16k/')
    parser.add_argument('--output_folder', default='results')
    parser.add_argument('--config', default='results')
    # parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--checkpoint_file', default='/disk4/chocho/SEMamba/ckpts/SEMamba_advanced.pth')
    parser.add_argument('--post_processing_PCS', default=False)
    # parser.add_argument('--mode', default="check")
    args = parser.parse_args()

    global device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        #device = torch.device('cpu')
        raise RuntimeError("Currently, CPU mode is not supported.")

    # if args.mode == "inference":
    inference_ablation(args, device)
    # inference_feature_map(args, device)
    # if args.mode == "FlopCountAnalysis":
        # inference_FlopCountAnalysis(args, device)
    # inference_chunk(args, device)
    # show_model(args, device)
    # show_model2(args, device)

if __name__ == '__main__':
    main()

