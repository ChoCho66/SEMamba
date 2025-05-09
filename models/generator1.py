import torch
import torch.nn as nn
from einops import rearrange
from .mamba_block import TFMambaBlock
from .codec_module import DenseEncoder, MagDecoder, PhaseDecoder
import matplotlib.pyplot as plt

class SEMamba(nn.Module):
    """
    SEMamba model for speech enhancement using Transformer blocks.
    
    This model uses a dense encoder, multiple Transformer blocks, and separate magnitude
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

        # Initialize Transformer blocks
        self.TSTransformer = nn.ModuleList([TFTransformerBlock(cfg) for _ in range(self.num_tscblocks)])

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

        # Apply Transformer blocks
        for block in self.TSTransformer:
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

class TFTransformerBlock(nn.Module):
    """
    Temporal-Frequency Transformer block for sequence modeling.
    
    Attributes:
    cfg (Config): Configuration for the block.
    time_transformer (TransformerEncoderLayer): Transformer layer for temporal dimension.
    freq_transformer (TransformerEncoderLayer): Transformer layer for frequency dimension.
    tlinear (ConvTranspose1d): ConvTranspose1d layer for temporal dimension.
    flinear (ConvTranspose1d): ConvTranspose1d layer for frequency dimension.
    """
    def __init__(self, cfg):
        super(TFTransformerBlock, self).__init__()
        self.cfg = cfg
        self.hid_feature = cfg['model_cfg']['hid_feature']
        
        # Transformer parameters
        dim_feedforward = self.hid_feature * 16  # Adjusted to keep parameter count similar
        nhead = cfg['model_cfg'].get('nhead', 4)  # Number of attention heads, default 4
        
        # Initialize Transformer layers
        self.time_transformer = nn.TransformerEncoderLayer(
            d_model=self.hid_feature,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=cfg['model_cfg'].get('dropout', 0.1),
            batch_first=True
        )
        self.freq_transformer = nn.TransformerEncoderLayer(
            d_model=self.hid_feature,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=cfg['model_cfg'].get('dropout', 0.1),
            batch_first=True
        )
        
        # Initialize ConvTranspose1d layers
        self.tlinear = nn.ConvTranspose1d(self.hid_feature, self.hid_feature, 1, stride=1)
        self.flinear = nn.ConvTranspose1d(self.hid_feature, self.hid_feature, 1, stride=1)
    
    def forward(self, x):
        """
        Forward pass of the TFTransformer block.
        
        Parameters:
        x (Tensor): Input tensor with shape (batch, channels, time, freq).
        
        Returns:
        Tensor: Output tensor after applying temporal and frequency Transformer blocks.
        """
        # x is [B, hid, T, F]
        b, c, t, f = x.size()

        # Temporal Transformer
        x_t = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)  # [B*F, T, C]
        x_t = self.time_transformer(x_t)  # [B*F, T, C]
        x_t = self.tlinear(x_t.permute(0, 2, 1)).permute(0, 2, 1) + x_t  # Residual connection
        x = x_t.view(b, f, t, c).permute(0, 2, 1, 3)  # [B, T, F, C]

        # Frequency Transformer
        x_f = x.contiguous().view(b * t, f, c)  # [B*T, F, C]
        x_f = self.freq_transformer(x_f)  # [B*T, F, C]
        x_f = self.flinear(x_f.permute(0, 2, 1)).permute(0, 2, 1) + x_f  # Residual connection
        x = x_f.view(b, t, f, c).permute(0, 3, 1, 2)  # [B, C, T, F]

        return x