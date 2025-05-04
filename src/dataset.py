# src/dataset.py
import os
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Lambdad, ScaleIntensityRanged,
    ResizeWithPadOrCropd, ToTensord, Compose,
    RandFlipd, RandRotate90d, RandGaussianNoised
)
from config import settings

class Slice2DDataset(Dataset):
    def __init__(self, split="train"):
        data_dir = os.path.join(settings.data_root, split)
        # discover each subject ID from your *_t1.nii.gz files
        subj_ids = [
            os.path.basename(p).split("_t1.nii.gz")[0]
            for p in sorted(
                [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith("_t1.nii.gz")]
            )
        ]
        self.slices = []
        # build list of (path1, path2, slice_index)
        for sid in subj_ids:
            img_path = os.path.join(data_dir, f"{sid}_t1.nii.gz")
            lbl_path = os.path.join(data_dir, f"{sid}_t2.nii.gz")
            vol = nib.load(img_path).get_fdata().astype(np.float32)
            depth = vol.shape[-1]
            for z in range(depth):
                self.slices.append((img_path, lbl_path, z))

        # define your 2D transforms
        self.transforms = Compose([
            # load happens in __getitem__ below
            Lambdad(
            keys=["image", "label"],
            func=lambda arr: np.expand_dims(arr, axis=0),
            ),
            ScaleIntensityRanged(
                keys=["image", "label"],
                a_min=0.0, a_max=1.0,
                b_min=-1.0, b_max=1.0,
                clip=True
            ),
            ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=settings.img_size
            ),
            RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
            RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
            ToTensord(keys=["image", "label"]),
        ])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, lbl_path, z = self.slices[idx]
        # load just the z-th slice
        img_vol = nib.load(img_path).get_fdata().astype(np.float32)
        lbl_vol = nib.load(lbl_path).get_fdata().astype(np.float32)
        img_slice = img_vol[..., z]
        lbl_slice = lbl_vol[..., z]
        sample = {"image": img_slice, "label": lbl_slice}
        return self.transforms(sample)


def get_dataloader(split="train"):
    ds = Slice2DDataset(split)
    return DataLoader(ds, batch_size=settings.batch_size, shuffle=True)

if __name__ == "__main__":

    # 1) build a loader and fetch one batch
    loader: DataLoader = get_dataloader("train")
    batch = next(iter(loader))
    imgs = batch["image"]   # shape: [B, 1, H, W]
    lbls = batch["label"]   # shape: [B, 1, H, W]

    # 2) print shapes
    print(f"  image batch shape: {imgs.shape}")
    print(f"  label batch shape: {lbls.shape}")