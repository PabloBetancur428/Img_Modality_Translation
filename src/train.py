# src/train.py
import os
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import L1Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from tqdm import tqdm
from pytorch_msssim import ssim as ssim_torch

from config import settings
from dataset import get_dataloader
from generator import Generator
from discriminator import Discriminator

# ─── hinge‐loss definitions ───────────────────────────────────────────
def discriminator_hinge_loss(D_real, D_fake):
    
    loss_real = F.relu(1.0 - D_real).mean()
    loss_fake = F.relu(1.0 + D_fake).mean()
    return 0.5 * (loss_real + loss_fake)

def generator_hinge_loss(D_fake):
    return -D_fake.mean()

# ─── warm‐up & update‐ratio config ────────────────────────────────────
warmup_epochs = settings.warmup_epochs
d_every = settings.d_every   # update D once every 2 batches

def train():
    """Train a conditional GAN with: hinge‐loss, 1:D/2:G updates, warm‐up, cosine LR decay, and validation."""
    set_determinism(42)
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, {torch.cuda.get_device_name(0)}")


    # ─── models ─────────────────────────────────────────────────────────
    G = Generator().to(device)
    D = Discriminator().to(device)

    # ─── optimizers & scheduler ─────────────────────────────────────────
    opt_G   = optim.Adam(G.parameters(), lr=settings.lr_G, betas=(0.5, 0.999))
    sched_G = CosineAnnealingLR(opt_G, T_max=settings.num_epochs, eta_min=1e-6)
    opt_D   = optim.Adam(D.parameters(), lr=settings.lr_D, betas=(0.5, 0.999))

    # ─── reconstruction loss ─────────────────────────────────────────────
    recon_loss_fn = L1Loss()

    # ─── data loaders ───────────────────────────────────────────────────
    train_loader = get_dataloader(
        settings.train_baseline_dir,
        settings.train_followup_dir,
        settings.train_excel,
        num_workers=6
    )
    val_loader = get_dataloader(
        settings.val_baseline_dir,
        followup_dir=None,
        excel_path=None,
        num_workers=6
    )
    print(f"Training on {len(train_loader.dataset)} slices; validating on {len(val_loader.dataset)} slices")

    os.makedirs(settings.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir="runs/cGAN_hinge_T2_QSM_variationsconfig")

    for epoch in range(1, settings.num_epochs + 1):
        G.train(); D.train()
        epoch_D_loss, epoch_G_loss = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{settings.num_epochs}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            real_t1 = batch["image"].to(device)
            real_t2 = batch["label"].to(device)

            # ─── (1) Warm‐up: L1 only ────────────────────────────────
            if epoch <= warmup_epochs:
                fake_t2 = G(real_t1)
                loss_G_l1 = recon_loss_fn(fake_t2, real_t2)
                loss_G    = settings.l1_weight * loss_G_l1

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

                epoch_G_loss += loss_G.item()
                pbar.set_postfix(G=f"{loss_G.item():.4f}", mode="warmup")
                continue

            # always generate
            fake_t2 = G(real_t1)

            # ─── (2) Discriminator (hinge) every d_every batches ──────
            if batch_idx % d_every == 0:
                real_pair = torch.cat([real_t1, real_t2], dim=1)
                fake_pair = torch.cat([real_t1, fake_t2.detach()], dim=1)
                D_real    = D(real_pair)
                D_fake    = D(fake_pair)
                loss_D    = discriminator_hinge_loss(D_real, D_fake)

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
            else:
                loss_D = torch.tensor(0.0, device=device)

            # ─── (3) Generator (every batch) ─────────────────────────
            fake_pair_G = torch.cat([real_t1, fake_t2], dim=1)
            D_fake_G    = D(fake_pair_G)
            loss_G_adv  = generator_hinge_loss(D_fake_G)
            loss_G_l1   = recon_loss_fn(fake_t2, real_t2)
            loss_G      = settings.adv_weight * loss_G_adv + settings.l1_weight * loss_G_l1

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # accumulate
            epoch_D_loss += loss_D.item()
            epoch_G_loss += loss_G.item()
            pbar.set_postfix(D=f"{loss_D.item():.4f}", G=f"{loss_G.item():.4f}")
        pbar.close()

        # ─── validation ──────────────────────────────────────────────
        G.eval()
        val_l1, val_ssim, val_psnr = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                real_t1 = batch["image"].to(device)
                real_t2 = batch["label"].to(device)
                fake_t2 = G(real_t1)

                # L1
                val_l1 += recon_loss_fn(fake_t2, real_t2).item()

                # SSIM ([-1,1]→[0,1])
                f01  = (fake_t2 + 1.0) / 2.0
                r01  = (real_t2 + 1.0) / 2.0
                val_ssim += ssim_torch(f01, r01, data_range=1.0).item()

                # PSNR
                mse = F.mse_loss(f01, r01).item()
                val_psnr += 10.0 * math.log10(1.0 / mse)

        n_val    = len(val_loader)
        avg_l1   = val_l1   / n_val
        avg_ssim = val_ssim / n_val
        avg_psnr = val_psnr / n_val

        writer.add_scalar("val/L1",   avg_l1,   epoch)
        writer.add_scalar("val/SSIM", avg_ssim, epoch)
        writer.add_scalar("val/PSNR", avg_psnr, epoch)
        print(f"↪ Val L1={avg_l1:.4f}  SSIM={avg_ssim:.4f}  PSNR={avg_psnr:.2f} dB")

        G.train()

        # ─── logging & scheduler ──────────────────────────────────
        avg_D = epoch_D_loss / len(train_loader)
        avg_G = epoch_G_loss / len(train_loader)
        print(f"Epoch {epoch}: D_loss={avg_D:.4f}, G_loss={avg_G:.4f}")
        writer.add_scalars("Losses", {"D": avg_D, "G": avg_G}, epoch)

        sched_G.step()
        lr_now = sched_G.get_last_lr()[0]
        writer.add_scalar("lr/G", lr_now, epoch)
        print(f"  → LR_G: {lr_now:.2e}")

        # ─── checkpoint ──────────────────────────────────────────
        if epoch % 10 == 0:
            torch.save(G.state_dict(), os.path.join(settings.ckpt_dir, f"G_ep{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(settings.ckpt_dir, f"D_ep{epoch}.pth"))

    writer.close()

if __name__ == "__main__":
    train()
