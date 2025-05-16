# src/inference.py
import sys
import os
import torch
import numpy as np
import nibabel as nib
from config import settings
from generator import Generator
from monai.transforms import (
    Compose,
    Lambdad,
    ResizeWithPadOrCropd,
    ToTensord,
)

def infer(input_path: str, output_path: str, ckpt_path: str):
    """
    Translate T1 MRI to synthetic T2 MRI using a trained generator.
    Full pipeline matches training normalization:
    1. Clip to 1st-99th percentiles
    2. Normalize to [0,1]
    3. Scale to [-1,1] for model input
    4. Output converted back to [0,1] for NIfTI storage
    """
    # Initialize device and verify inputs
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    print(f"⚡ Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Verify files exist
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 1) Load model with explicit float32 precision
    print("⏳ Loading generator model...")
    gen = Generator().to(device).float()  # Force float32
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()

    # 2) Load and preprocess volume
    print("🧠 Processing input volume...")
    vol = nib.load(input_path).get_fdata().astype(np.float32)  # Force float32
    p1, p99 = np.percentile(vol, [1, 99])
    print(f"   Intensity range: [{vol.min():.2f}, {vol.max():.2f}]")
    print(f"   Percentiles (1%, 99%): {p1:.2f}, {p99:.2f}")

    # 3) Normalization pipeline
    vol_clipped = np.clip(vol, p1, p99)
    vol_norm = (vol_clipped - p1) / (p99 - p1 + 1e-6)  # [0,1] range

    # 4) Define transforms with explicit type control
    transforms = Compose([
        Lambdad(keys=["image"], 
               func=lambda x: (x * 2.0 - 1.0)[None, ...].astype(np.float32)),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=settings.img_size),
        ToTensord(keys=["image"], dtype=torch.float32)  # Explicit dtype
    ])

    # 5) Process slices with progress feedback
    preds = []
    depth = vol_norm.shape[-1]
    print(f"🔍 Translating {depth} slices...")
    for z in range(depth):
        slice2d = vol_norm[..., z]
        sample = {"image": slice2d}
        
        # Verify input type before model
        img_t = transforms(sample)["image"].unsqueeze(0).to(device)
        if img_t.dtype != torch.float32:
            raise TypeError(f"Expected float32 input, got {img_t.dtype}")

        with torch.no_grad():
            out_t = gen(img_t)  # Model outputs [-1,1]
            out_np = out_t.cpu().numpy()[0, 0]
            preds.append((out_np + 1.0) / 2.0)  # Convert to [0,1]

        # Progress feedback
        if (z + 1) % 50 == 0 or (z + 1) == depth:
            print(f"   Processed slice {z + 1}/{depth}")

    # 6) Save output volume
    pred_vol = np.stack(preds, axis=-1).astype(np.float32)
    nib.save(nib.Nifti1Image(pred_vol, np.eye(4)), output_path)
    print(f"\n✅ Success! Saved synthetic T2 volume to:\n   {os.path.abspath(output_path)}")
    print(f"   Output range: [{pred_vol.min():.2f}, {pred_vol.max():.2f}]")

if __name__ == "__main__":
    # Command-line interface
    if len(sys.argv) != 4:
        print("\nUsage: python inference.py <input_T1.nii.gz> <output_T2.nii.gz> <generator_checkpoint.pth>")
        print("Example:")
        print("  python inference.py patient1_T1.nii.gz patient1_synthetic_T2.nii.gz checkpoints/G_epoch100.pth\n")
        sys.exit(1)
    
    try:
        infer(sys.argv[1], sys.argv[2], sys.argv[3])
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        sys.exit(1)