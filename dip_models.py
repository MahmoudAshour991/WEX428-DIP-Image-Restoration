"""
WEX 428 - Computer Vision Project
Module 3: Deep Image Prior Reconstruction Module (Neural Network Architectures)

Implements two architectures from scratch in PyTorch for comparison:
1. UNetDIP: Hourglass network WITH skip connections (preserves fine spatial details)
2. EncoderDecoderDIP: Hourglass network WITHOUT skip connections (stronger smoothing prior)

Includes automatic hardware detection for AMD DirectML (RX 5700 XT), CUDA, or multi-core CPU (Ryzen 5 5600).
"""

import os
import torch
import torch.nn as nn


def get_compute_device():
    """
    Detects the best available device:
    1. AMD Radeon GPU via torch-directml (e.g., RX 5700 XT on Windows)
    2. NVIDIA GPU via CUDA
    3. Multi-threaded CPU (optimized for Ryzen 5 5600 12 threads)
    """
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device(), "AMD DirectML GPU"
    except ImportError:
        pass

    if torch.cuda.is_available():
        return torch.device("cuda"), f"CUDA ({torch.cuda.get_device_name(0)})"

    cpu_threads = min(12, os.cpu_count() or 8)
    torch.set_num_threads(cpu_threads)
    return torch.device("cpu"), f"CPU ({cpu_threads} threads)"


class ConvBlock(nn.Module):
    """Standard double convolution block: Conv2d (reflection pad) -> BatchNorm -> LeakyReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(pad),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDIP(nn.Module):
    """
    Deep Image Prior U-Net architecture with 1x1 skip connections.
    Follows the architectural principles of Ulyanov et al. (2018).
    """

    def __init__(self, in_channels=16, out_channels=3, base_channels=32, skip_channels=4):
        super().__init__()
        c = base_channels

        # Encoder pathway
        self.enc1 = ConvBlock(in_channels, c)
        self.down1 = nn.AvgPool2d(2)

        self.enc2 = ConvBlock(c, c * 2)
        self.down2 = nn.AvgPool2d(2)

        self.enc3 = ConvBlock(c * 2, c * 4)
        self.down3 = nn.AvgPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(c * 4, c * 4)

        # 1x1 Skip connection projections (standard in DIP to control high-frequency leakage)
        self.skip1 = nn.Sequential(
            nn.Conv2d(c, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(c * 2, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(c * 4, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder pathway with bilinear upsampling
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(c * 4 + skip_channels, c * 2)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(c * 2 + skip_channels, c)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(c + skip_channels, c)

        self.out_head = nn.Sequential(
            nn.Conv2d(c, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        e1 = self.enc1(z)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, self.skip3(e3)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, self.skip2(e2)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, self.skip1(e1)], dim=1))

        return self.out_head(d1)


class EncoderDecoderDIP(nn.Module):
    """
    Standard Hourglass Encoder-Decoder architecture WITHOUT skip connections.
    Used for experimental comparison against UNetDIP.
    """

    def __init__(self, in_channels=16, out_channels=3, base_channels=32):
        super().__init__()
        c = base_channels

        self.encoder = nn.Sequential(
            ConvBlock(in_channels, c),
            nn.AvgPool2d(2),
            ConvBlock(c, c * 2),
            nn.AvgPool2d(2),
            ConvBlock(c * 2, c * 4),
            nn.AvgPool2d(2),
        )

        self.bottleneck = ConvBlock(c * 4, c * 4)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(c * 4, c * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(c * 2, c),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBlock(c, c),
            nn.Conv2d(c, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        latent = self.bottleneck(self.encoder(z))
        return self.decoder(latent)


def build_dip_model(arch_name="unet", in_channels=16, base_channels=32):
    """Factory helper to create the chosen DIP architecture."""
    arch_name = arch_name.lower().strip()
    if arch_name in ("unet", "u-net", "unetdip"):
        return UNetDIP(in_channels=in_channels, out_channels=3, base_channels=base_channels)
    elif arch_name in ("encoder_decoder", "encoder-decoder", "encdec"):
        return EncoderDecoderDIP(in_channels=in_channels, out_channels=3, base_channels=base_channels)
    else:
        raise ValueError(f"Unknown architecture '{arch_name}'. Choose 'unet' or 'encoder_decoder'.")
