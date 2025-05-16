# src/dataset.py

import os
import glob
import nibabel as nib
import numpy as np
from torch.utils.data import DataLoader, Dataset
from monai.transforms import (
    Lambdad, ScaleIntensityd, ScaleIntensityRanged, ResizeWithPadOrCropd,
    ToTensord, RandFlipd, RandRotate90d, RandGaussianNoised,
    Compose
)
from config import settings
from data_utils import load_all_patient_files, load_patient_files


class Slice2DDataset(Dataset):
    """
    Enhanced version with volume caching and percentile-based normalization
    """
    def __init__(self, records):
        self.slice_info = []
        self.volume_cache = {}  # Stores loaded volumes: {path: numpy_array}
        
        # Load volumes once and compute statistics
        for rec in records:
            # Cache T1 volume if not already loaded
            if rec["t1"] not in self.volume_cache:
                self.volume_cache[rec["t1"]] = nib.load(rec["t1"]).get_fdata().astype(np.float32)
            
            # Cache T2 volume if not already loaded
            if rec["t2"] not in self.volume_cache:
                self.volume_cache[rec["t2"]] = nib.load(rec["t2"]).get_fdata().astype(np.float32)
            
            vol1 = self.volume_cache[rec["t1"]]
            vol2 = self.volume_cache[rec["t2"]]
            
            # Compute volume-specific percentiles (robust to outliers)
            p1_1, p99_1 = np.percentile(vol1, (1, 99))
            p1_2, p99_2 = np.percentile(vol2, (1, 99))
            
            depth = min(vol1.shape[-1], vol2.shape[-1])
            if vol1.shape[-1] != vol2.shape[-1]:
                print(f"⚠️  Slice count mismatch: {os.path.basename(rec['t1'])} "
                      f"has {vol1.shape[-1]} vs {vol2.shape[-1]} → using {depth}")
            
            # Store slice metadata
            for z in range(depth):
                self.slice_info.append({
                    "t1": rec["t1"], "t2": rec["t2"], "z": z,
                    "p1_1": p1_1, "p99_1": p99_1,
                    "p1_2": p1_2, "p99_2": p99_2
                })

        # Spatial transforms only (normalization happens in __getitem__)
        self.transforms = Compose([
            Lambdad(keys=["image","label"],
                   func=lambda x: np.expand_dims(x, axis=0)),
            ResizeWithPadOrCropd(keys=["image","label"],
                                 spatial_size=settings.img_size),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            ToTensord(keys=["image","label"]),
        ])

    def __len__(self):
        return len(self.slice_info)

    def __getitem__(self, idx):
        info = self.slice_info[idx]
        
        # Get slices from cache (no disk I/O)
        vol1_slice = self.volume_cache[info["t1"]][..., info["z"]]
        vol2_slice = self.volume_cache[info["t2"]][..., info["z"]]
        
        # 1) Clip to precomputed percentiles
        vol1_slice = np.clip(vol1_slice, info["p1_1"], info["p99_1"])
        vol2_slice = np.clip(vol2_slice, info["p1_2"], info["p99_2"])
        
        # 2) Normalize to [0,1] using volume stats
        eps = 1e-6  # Prevent division by zero
        vol1_slice = (vol1_slice - info["p1_1"]) / (info["p99_1"] - info["p1_1"] + eps)
        vol2_slice = (vol2_slice - info["p1_2"]) / (info["p99_2"] - info["p1_2"] + eps)
        
        # 3) Scale to [-1, 1] range
        vol1_slice = vol1_slice * 2.0 - 1.0
        vol2_slice = vol2_slice * 2.0 - 1.0

        return self.transforms({
            "image": vol1_slice.astype(np.float32),
            "label": vol2_slice.astype(np.float32)
        })

    def __del__(self):
        """Clean up cached volumes when dataset is deleted"""
        self.volume_cache.clear()


def get_dataloader(
    baseline_dir: str,
    followup_dir: str = None,
    excel_path: str = None,
    num_workers: int = 6,
):
    """
    If excel_path is provided => train mode: use load_all_patient_files()
    If excel_path is None   => val mode: use load_patient_files() on baseline_dir
    """
    if excel_path:
        # TRAINING: filter by Excel
        records = load_all_patient_files(baseline_dir, followup_dir, excel_path)
    else:
        # VALIDATION: include all patients under baseline_dir
        patient_set = set(os.listdir(baseline_dir))
        records = load_patient_files(baseline_dir, patient_set)

    print(f"Found {len(records)} volumes, extracting 2D slices...")
    ds = Slice2DDataset(records)
    print(f"Total 2D slice-pairs: {len(ds)}    workers={num_workers}")
    print(f"Metadata: {ds.slice_info[200]}")

    return DataLoader(
        ds,
        batch_size=settings.batch_size,
        shuffle=bool(excel_path),   # shuffle only in train mode
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # Train loader example
    train_loader = get_dataloader(
        baseline_dir=os.path.join(settings.data_root, "trial_baseline"),
        followup_dir=os.path.join(settings.data_root, "trial_follow_up"),
        excel_path=os.path.join(settings.data_root, "patients_with_qsm.xlsx"),
        num_workers=6
    )
    print("Train batches:", len(train_loader))

    # Validation loader example
    val_loader = get_dataloader(
        baseline_dir=os.path.join(settings.data_root, "validation_synthesis"),
        excel_path=None,
        num_workers=6
    )
    print("Val batches:", len(val_loader))