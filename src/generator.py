# src/model.py
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from config import settings

class Generator(nn.Module):
    """
    2D U-Net generator mapping T1→T2 slices (or vice-versa).
    Input:  x [B, in_channels, H, W] ∈ [-1,1]
    Output: y [B, out_channels, H, W] ∈ [-1,1]
    """
    def __init__(self):
        super().__init__()
        # core U-Net
        self.unet = UNet(
            spatial_dims   = settings.spatial_dims,   # =2 for 2D
            in_channels    = settings.in_channels,    # =1
            out_channels   = settings.out_channels,   # =1
            channels       = (64, 128, 256, 512),      # encoder/decoder widths
            strides        = (2, 2, 2),               # three 2× downsamples
            num_res_units  = 2,
            norm           = "batch",
            act            = "leakyrelu",
        )
        # final tanh to squash into [-1,1]
        self.final_act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W], values ∈[-1,1]
        returns: [B,1,H,W], values ∈[-1,1]
        """
        x = self.unet(x)
        return self.final_act(x)


if __name__ == "__main__":
    # quick shape/dtype sanity check
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Generator().to(device)
    dummy = torch.randn(2, settings.in_channels, *settings.img_size, device=device)
    out   = model(dummy)
    assert out.shape == dummy.shape, \
        f"Shape mismatch: in {dummy.shape} vs out {out.shape}"
    # tanh(0)=0, so zero input→zero output
    zero_out = model(torch.zeros_like(dummy))
    assert torch.allclose(zero_out, torch.zeros_like(zero_out), atol=1e-4), \
        "Zero input should produce near-zero output"
    print("✅ Generator module sanity checks passed.")
