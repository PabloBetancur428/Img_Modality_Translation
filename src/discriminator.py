# src/discriminator.py
import torch
import torch.nn as nn
from config import settings

class Discriminator(nn.Module):
    """
    2D PatchGAN discriminator that takes [T1, T2] pairs and
    predicts real/fake for each patch.
    Input shape: [B, in_channels+out_channels, H, W]
    Output shape: [B, 1, H/2^4, W/2^4]  (patch map)
    """
    def __init__(self):
        super().__init__()
        inp = settings.in_channels + settings.out_channels  # 2 channels
        nf = 64  # base number of filters

        def conv_block(in_f, out_f, stride, use_bn=True):
            layers = [nn.Conv2d(in_f, out_f, kernel_size=4, stride=stride, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # no batchnorm on first layer
            *conv_block(inp,    nf,    stride=2, use_bn=False),  # [B,64,H/2,W/2]
            *conv_block(nf,     nf*2,  stride=2),                # [B,128,H/4,W/4]
            *conv_block(nf*2,   nf*4,  stride=2),                # [B,256,H/8,W/8]
            *conv_block(nf*4,   nf*8,  stride=1),                # [B,512,H/8-1,W/8-1]
            # final conv → 1 channel output (no activation here, we'll use BCEWithLogits)
            nn.Conv2d(nf*8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
