# src/config.py
from pydantic_settings import BaseSettings
from typing import Tuple

class Settings(BaseSettings):
    # ─── Data paths ────────────────────────────────────────────────
    # Base folder under which your train/val subdirectories live
    data_root: str = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization"

    # Training split: two time‐points per patient
    train_baseline_dir: str = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/trial_baseline"
    train_followup_dir: str = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/trial_follow_up"
    train_excel: str = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/patients_with_qsm.xlsx"

    # Validation split: just one folder of held‐out patients
    val_baseline_dir: str = "/home/jbetancur/Desktop/codes/python_qsm/exploratory_pipeline/data_automatization/validation_synthesis"

    # ─── Image & model specs ───────────────────────────────────────
    in_channels: int = 1
    out_channels: int = 1
    spatial_dims: int = 2
    img_size: Tuple[int, int] = (256, 256)

    # ─── Optimizer hyperparameters ─────────────────────────────────
    lr_G: float = 2e-4     # faster generator learning
    lr_D: float = 1e-4     # slower discriminator learning

    # ─── GAN loss balancing ────────────────────────────────────────
    adv_weight: float = 1.0  # (less adversarial pressure)
    l1_weight: float  = 30.0  # (stronger reconstruction)

    # ─── Training loop settings ───────────────────────────────────
    batch_size: int = 12 # better gradient estimates
    num_epochs: int = 50 # Needs more training, increase later

    # Warm up
    warmup_epochs: int = 5
    use_gradient_penalty: bool = True
    d_every:int = 1 # update D once every 2 batches
    # ─── Compute & I/O ─────────────────────────────────────────────
    device: str = "cuda"       # or "cpu"
    ckpt_dir: str = "./checkpoints_T2_QSM"

    class Config:
        validate_assignment = True  # catch typos immediately

settings = Settings()
