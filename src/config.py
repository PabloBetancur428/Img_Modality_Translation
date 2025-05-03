# src/config.py
from pydantic_settings import BaseSettings
from typing import Tuple

class Settings(BaseSettings):
    # data + model
    data_root: str = "data"
    in_channels: int = 1
    out_channels: int = 1
    spatial_dims: int = 2
    img_size: Tuple[int, int] = (256, 256)

    # optim
    lr_G: float = 1e-4         # generator LR
    lr_D: float = 1e-5         # discriminator LR
    batch_size: int = 4
    num_epochs: int = 30

    # device + checkpoints
    device: str = "cuda"
    ckpt_dir: str = "./checkpoints"

    # GAN losses
    adv_weight: float = 1.0    # weight for adversarial loss in G
    l1_weight: float = 10.0   # weight for L1 reconstruction loss in G

    class Config:
        validate_assignment = True

settings = Settings()
