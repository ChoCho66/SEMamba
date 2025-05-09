import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import TFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder
import matplotlib.pyplot as plt

class SEMamba(nn.Module):
    """
    SEMamba model for speech enhancement using Mamba blocks.
    
    This model uses a dense encoder, multiple Mamba blocks, and separate magnitude
    and phase decoders to process noisy magnitude and phase inputs.
    """
    def __init__(self, cfg):
        """
        Initialize the SEMamba model.
        
        Args:
        - cfg: Configuration object containing model parameters.
        """
        super(SEMamba, self).__init__()
        self.cfg = cfg
        self.num_tscblocks = cfg['model_cfg']['num_tfmamba'] if cfg['model_cfg']['num_tfmamba'] is not None else 4  # default tfmamba: 4

        # Initialize dense encoder
        self.dense_encoder = DenseEncoder(cfg)

        # Initialize Mamba blocks
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(self.num_tscblocks)])

        # Initialize decoders
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        """
        Forward pass for the SEMamba model.
        
        Args:
        - noisy_mag (torch.Tensor): Noisy magnitude input tensor [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase input tensor [B, F, T].
        
        Returns:
        - denoised_mag (torch.Tensor): Denoised magnitude tensor [B, F, T].
        - denoised_pha (torch.Tensor): Denoised phase tensor [B, F, T].
        - denoised_com (torch.Tensor): Denoised complex tensor [B, F, T, 2].
        """
        # Reshape inputs
        noisy_mag = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B F T] -> [B, 1, T, F]
        noisy_pha = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B F T] -> [B, 1, T, F]

        # Concatenate magnitude and phase inputs
        x = torch.cat((noisy_mag, noisy_pha), dim=1)  # [B, 2, T, F]

        # Encode input
        x = self.dense_encoder(x)

        # Apply Mamba blocks
        for block in self.TSMamba:
            x = block(x)

        # Decode magnitude and phase
        denoised_mag = rearrange(self.mask_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha = rearrange(self.phase_decoder(x), 'b c t f -> b f t c').squeeze(-1)

        # Combine denoised magnitude and phase into a complex representation
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)),
            dim=-1
        )

        return denoised_mag, denoised_pha, denoised_com
    
    def get_feature_map(self, output_folder, file_name, noisy_mag, noisy_pha, clean_or_noisy="noisy"):
        """
        Visualize specified feature maps from the SEMamba model in a single figure.
        X-axis represents time, Y-axis represents frequency.
        Layout:
        - Row 1: Noisy Magnitude, Noisy Phase (after reshape)
        - Row 2: Dense Encoder Output (Channels 1, 2)
        - Row 3: TSMamba Block 1 Output (Channels 1, 2, 3)
        - Row 4: TSMamba Block 2 Output (Channels 1, 2, 3)
        - Row 5: Mag Decoder Output, Phase Decoder Output
        
        Args:
            file_name (str): Name of the file to include in the plot title.
            noisy_mag (torch.Tensor): Input magnitude spectrogram [B, F, T].
            noisy_pha (torch.Tensor): Input phase spectrogram [B, F, T].
        
        Returns:
            Saves a PNG file with the visualized feature maps.
        """
        # List to store feature maps and their labels
        feature_maps = []
        labels = []
        
        # Step 1: Reshape noisy_mag and noisy_pha
        noisy_mag_reshaped = rearrange(noisy_mag, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        noisy_pha_reshaped = rearrange(noisy_pha, 'b f t -> b t f').unsqueeze(1)  # [B, 1, T, F]
        
        # Append noisy inputs
        feature_maps.append(noisy_mag_reshaped[0, 0].detach().cpu().numpy())  # [T, F]
        labels.append("Noisy Magnitude (after reshape)")
        feature_maps.append(noisy_pha_reshaped[0, 0].detach().cpu().numpy())  # [T, F]
        labels.append("Noisy Phase (after reshape)")
        
        # Step 2: Concatenate and encode
        x = torch.cat((noisy_mag_reshaped, noisy_pha_reshaped), dim=1)  # [B, 2, T, F]
        x = self.dense_encoder(x)  # Assume output shape [B, C, T, F]
        
        # Append dense encoder outputs (channels 1, 2)
        num_channels = x.shape[1]
        for i in range(min(2, num_channels)):
            feature_maps.append(x[0, i].detach().cpu().numpy())  # [T, F]
            labels.append(f"Dense Encoder Output (Channel {i+1})")
        
        # Step 3: Apply TSMamba blocks
        for idx, block in enumerate(self.TSMamba):
            x = block(x)  # Assume x remains [B, C, T, F]
            if idx < 2:  # Only first two blocks
                num_channels = x.shape[1]
                for i in range(min(3, num_channels)):  # Channels 1, 2, 3
                    feature_maps.append(x[0, i].detach().cpu().numpy())  # [T, F]
                    labels.append(f"TSMamba Block {idx+1} Output (Channel {i+1})")
        
        # Step 4: Decoder outputs
        mask_decoder_output = self.mask_decoder(x) * noisy_mag_reshaped  # [B, C, T, F] * [B, 1, T, F] -> [B, C, T, F]
        phase_decoder_output = self.phase_decoder(x)  # Assume [B, C, T, F]
        
        # Append decoder outputs
        feature_maps.append(mask_decoder_output[0, 0].detach().cpu().numpy())  # [T, F]
        labels.append("Mag Decoder Output")
        feature_maps.append(phase_decoder_output[0, 0].detach().cpu().numpy())  # [T, F]
        labels.append("Phase Decoder Output")
        
        # Step 5: Plot feature maps in specified layout
        plt.figure(figsize=(15, 20))  # 3 columns, 5 rows
        
        # Row 1: Noisy Magnitude, Noisy Phase
        plt.subplot(5, 3, 1)
        plt.imshow(feature_maps[0].T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title(labels[0], fontsize=10)
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency Bin')
        
        plt.subplot(5, 3, 2)
        plt.imshow(feature_maps[1].T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title(labels[1], fontsize=10)
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency Bin')
        
        # Row 2: Dense Encoder Outputs
        for i in range(2):
            plt.subplot(5, 3, 4 + i)
            plt.imshow(feature_maps[2 + i].T, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Amplitude')
            plt.title(labels[2 + i], fontsize=10)
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency Bin')
        
        # Row 3: TSMamba Block 1 Outputs
        for i in range(3):
            plt.subplot(5, 3, 7 + i)
            plt.imshow(feature_maps[4 + i].T, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Amplitude')
            plt.title(labels[4 + i], fontsize=10)
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency Bin')
        
        # Row 4: TSMamba Block 2 Outputs
        for i in range(3):
            plt.subplot(5, 3, 10 + i)
            plt.imshow(feature_maps[7 + i].T, aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Amplitude')
            plt.title(labels[7 + i], fontsize=10)
            plt.xlabel('Time Frame')
            plt.ylabel('Frequency Bin')
        
        # Row 5: Decoder Outputs
        plt.subplot(5, 3, 13)
        plt.imshow(feature_maps[10].T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title(labels[10], fontsize=10)
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency Bin')
        
        plt.subplot(5, 3, 14)
        plt.imshow(feature_maps[11].T, aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title(labels[11], fontsize=10)
        plt.xlabel('Time Frame')
        plt.ylabel('Frequency Bin')
        
        plt.suptitle(f"Feature Maps for {file_name}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
        
        import os
        # Save the plot
        plt.savefig(os.path.join(output_folder,f'feature_maps_{file_name.split(".")[0]}_{clean_or_noisy}.png'), bbox_inches='tight')
        plt.close()