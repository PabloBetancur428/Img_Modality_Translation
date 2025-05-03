# src/train.py
import os
import torch
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss, L1Loss
from torch.utils.tensorboard import SummaryWriter
from monai.utils import set_determinism
from tqdm import tqdm

from config import settings
from dataset import get_dataloader
from model import Generator
from discriminator import Discriminator

def train():
    """Train a conditional GAN: Update D then G each batch."""
    set_determinism(42)
    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")

    # instantiate networks
    G = Generator().to(device)
    D = Discriminator().to(device)

    # optimizers
    opt_G = optim.Adam(G.parameters(), lr=settings.lr_G, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=settings.lr_D, betas=(0.5, 0.999))

    # losses
    adv_loss_fn = BCEWithLogitsLoss()
    recon_loss_fn = L1Loss()

    loader = get_dataloader("train")
    os.makedirs(settings.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir="runs/cGAN")

    for epoch in range(1, settings.num_epochs+1):
        G.train(); D.train()
        epoch_D_loss, epoch_G_loss = 0.0, 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{settings.num_epochs}", unit="batch")
        for batch in pbar:
            real_t1 = batch["image"].to(device)   # [B,1,H,W]
            real_t2 = batch["label"].to(device)   # [B,1,H,W]

            # ─── Train Discriminator ───────────────────────────────
            # real pair
            real_pair = torch.cat([real_t1, real_t2], dim=1)
            # fake pair (detach so G's grads aren't computed here)
            fake_t2 = G(real_t1)
            fake_pair = torch.cat([real_t1, fake_t2.detach()], dim=1)

            D_real = D(real_pair)
            D_fake = D(fake_pair)

            loss_D_real = adv_loss_fn(D_real, torch.ones_like(D_real))
            loss_D_fake = adv_loss_fn(D_fake, torch.zeros_like(D_fake))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ─── Train Generator ───────────────────────────────────
            # adversarial: want D(fake_pair) → 1
            fake_pair_for_G = torch.cat([real_t1, fake_t2], dim=1)
            D_fake_for_G = D(fake_pair_for_G)
            loss_G_adv = adv_loss_fn(D_fake_for_G, torch.ones_like(D_fake_for_G))

            # reconstruction (L1) loss
            loss_G_l1 = recon_loss_fn(fake_t2, real_t2)

            # total G loss
            loss_G = settings.adv_weight * loss_G_adv + settings.l1_weight * loss_G_l1

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            # accumulate
            epoch_D_loss += loss_D.item()
            epoch_G_loss += loss_G.item()

            # update progress bar
            pbar.set_postfix(D=f"{loss_D.item():.4f}", G=f"{loss_G.item():.4f}")
        pbar.close()

        # epoch averages & logging
        avg_D = epoch_D_loss / len(loader)
        avg_G = epoch_G_loss / len(loader)
        print(f"Epoch {epoch}: D_loss={avg_D:.4f}, G_loss={avg_G:.4f}")
        writer.add_scalars("Losses", {"D": avg_D, "G": avg_G}, epoch)

        # checkpoint both nets
        if epoch % 10 == 0:
            torch.save(G.state_dict(), os.path.join(settings.ckpt_dir, f"G_epoch{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(settings.ckpt_dir, f"D_epoch{epoch}.pth"))

    writer.close()

if __name__ == "__main__":
    train()
