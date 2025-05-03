# src/inference.py
import sys
import torch
import numpy as np
import nibabel as nib
from config import settings
from model import Generator

def infer(input_path: str, output_path: str, ckpt_path: str):
    """
    Load a trained checkpoint and translate a full T1 volume → T2 volume.
    For 2D pipeline, we process slice-by-slice and stack back.
    """
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(ckpt_path, map_location=device))
    gen.eval()

    # load & normalize to [-1,1]
    vol = nib.load(input_path).get_fdata().astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min())  # [0,1]
    vol = vol * 2 - 1                                  # [-1,1]

    # process each slice
    preds = []
    for z in range(vol.shape[-1]):
        slice2d = vol[..., z]
        tensor  = torch.from_numpy(slice2d)[None, None].to(device)
        with torch.no_grad():
            out2d = gen(tensor).cpu().numpy()[0, 0]
        preds.append((out2d + 1) / 2)  # back to [0,1]

    pred_vol = np.stack(preds, axis=-1)
    nib.save(nib.Nifti1Image(pred_vol, np.eye(4)), output_path)
    print(f"✅ Saved translated volume to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python inference.py <input.nii.gz> <output.nii.gz> <checkpoint.pth>")
        sys.exit(1)
    infer(*sys.argv[1:])
