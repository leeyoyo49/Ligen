from typing import Optional, Dict
import os
import time

import torch
import torch.nn as nn
try:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
except Exception:
    from torch.amp import autocast, GradScaler  # type: ignore
from types import SimpleNamespace

# ...existing code...
from models.tabgan import TabGenerator, TabDiscriminator  # keep current class names
from trainers.common import ETAMeter, format_hms
from utils.logger import get_logger
from tqdm.auto import tqdm


def dict_to_ns(d: dict):
    return SimpleNamespace(**d)


class TabGANTrainer:
    """
    Generic Tabular GAN trainer (unconditional).
    DataLoader should yield a tensor batch X with shape [B, x_dim].
    """

    def __init__(self, x_dim: int, cfg_tabgan: dict, device: str = "cuda", wb=None):
        self.x_dim = x_dim
        self.cfg = dict_to_ns(cfg_tabgan)
        self.device = torch.device(device)
        self.wb = wb
        self.log = get_logger("trainer.tabgan")

        # hyperparams
        self.z_dim = getattr(self.cfg, "z_dim", x_dim)
        self.hidden_size = getattr(self.cfg, "hidden_size", 256)
        # TTUR
        self.lr_g = getattr(self.cfg, "lr_g", getattr(self.cfg, "lr", 2e-4))
        self.lr_d = getattr(self.cfg, "lr_d", getattr(self.cfg, "lr", 2e-4))
        self.betas = tuple(getattr(self.cfg, "betas", (0.5, 0.999)))
        self.smooth = getattr(self.cfg, "label_smoothing", 0.9)
        self.epochs = getattr(self.cfg, "epochs", 5000)
        self.save_every = getattr(self.cfg, "save_every", None)
        # multi D steps like CGAN
        self.d_steps = int(getattr(self.cfg, "d_steps", 1))
        # synth batching for large generations (align with CGAN)
        self.synth_batch_size = int(getattr(self.cfg, "synth_batch_size", 65536))
        # stability extras
        self.inst_noise_std = float(getattr(self.cfg, "instance_noise_std", 0.0) or 0.0)
        self.r1_gamma = float(getattr(self.cfg, "r1_gamma", 0.0) or 0.0)
        self.grad_clip = float(getattr(self.cfg, "grad_clip", 0.0) or 0.0)
        self.use_amp = bool(getattr(self.cfg, "amp", True))
        self.ema_decay = float(getattr(self.cfg, "ema_decay", 0.0) or 0.0)

        # models/optim
        self.G = TabGenerator(self.z_dim, self.hidden_size, x_dim, cfg=self.cfg).to(self.device)
        self.D = TabDiscriminator(x_dim, self.hidden_size, cfg=self.cfg).to(self.device)
        self.crit = nn.BCEWithLogitsLoss()
        self.optG = torch.optim.Adam(self.G.parameters(), lr=self.lr_g, betas=self.betas)
        self.optD = torch.optim.Adam(self.D.parameters(), lr=self.lr_d, betas=self.betas)
        self.scalerG = GradScaler(enabled=self.use_amp)
        self.scalerD = GradScaler(enabled=self.use_amp)
        # EMA
        self.G_ema = None
        if self.ema_decay > 0:
            import copy
            self.G_ema = copy.deepcopy(self.G).eval()
            for p in self.G_ema.parameters():
                p.requires_grad_(False)

        # one-time W&B metric definition guard (align with CGAN)
        self._wb_defined = False

        self.log.info(
            f"initialized TabGAN: x_dim={x_dim} z_dim={self.z_dim} hidden={self.hidden_size} "
            f"lr_g={self.lr_g:.2e} lr_d={self.lr_d:.2e} betas={self.betas} epochs={self.epochs} d_steps={self.d_steps}"
        )

    # checkpoints aligned with CGAN
    def _save_ckpt(self, path: str, epoch: int, g_loss: float, d_loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
                "optG": self.optG.state_dict(),
                "optD": self.optD.state_dict(),
                "g_loss": g_loss,
                "d_loss": d_loss,
            },
            path,
        )
        tqdm.write(f"[TabGAN] checkpoint saved @ epoch {epoch} → {path}")
        self.log.info(f"checkpoint saved @ epoch {epoch} → {path}")

    def _maybe_resume(self, path: Optional[str]) -> int:
        if path and os.path.exists(path):
            ckpt = torch.load(path, map_location=self.device)
            self.G.load_state_dict(ckpt["G"])
            self.D.load_state_dict(ckpt["D"])
            self.optG.load_state_dict(ckpt["optG"])
            self.optD.load_state_dict(ckpt["optD"])
            self.log.info(f"resuming from epoch {ckpt['epoch']+1}")
            return int(ckpt["epoch"]) + 1
        return 0

    def train(self, loader: torch.utils.data.DataLoader, ckpt_path: Optional[str] = None, resume: bool = True) -> Dict[str, float]:
        start_epoch = self._maybe_resume(ckpt_path) if resume else 0
        self.log.info(f"start training: epochs={self.epochs} lr_g={self.lr_g} lr_d={self.lr_d} device={self.device} amp={self.use_amp}")
        pbar = tqdm(total=self.epochs, initial=start_epoch, desc="[TabGAN] Training", dynamic_ncols=True)
        # define W&B metric here too (defensive)
        if self.wb and not self._wb_defined:
            try:
                self.wb.define_metric("tabgan/*", step_metric="tabgan_step")
            except Exception:
                pass
            self._wb_defined = True
        t0 = time.time()
        last_d = 0.0
        last_g = 0.0

        def add_instance_noise(x):
            if self.inst_noise_std > 0:
                return x + torch.randn_like(x) * self.inst_noise_std
            return x

        def ema_update():
            if self.G_ema is None:
                return
            with torch.no_grad():
                for p_ema, p in zip(self.G_ema.parameters(), self.G.parameters()):
                    p_ema.data.lerp_(p.data, 1.0 - self.ema_decay)

        for epoch in range(start_epoch, self.epochs):
            d_total = g_total = 0.0
            nb = 0
            for batch in loader:
                real = batch[0] if isinstance(batch, (list, tuple)) else batch
                real = real.to(self.device).float()
                nb += 1
                bsz = real.size(0)

                # Discriminator (optionally multiple steps like CGAN)
                d_losses = []
                for _ in range(self.d_steps):
                    z = torch.randn(bsz, self.z_dim, device=self.device)
                    with autocast(enabled=self.use_amp):
                        fake = self.G(z).detach()
                        self.optD.zero_grad(set_to_none=True)
                        real_in = add_instance_noise(real)
                        if self.r1_gamma > 0:
                            real_in.requires_grad_(True)
                        fake_in = add_instance_noise(fake)
                        d_real = self.crit(self.D(real_in), torch.full((bsz,1), self.smooth, device=self.device))
                        d_fake = self.crit(self.D(fake_in), torch.zeros(bsz,1, device=self.device))
                        d_loss = d_real + d_fake
                        if self.r1_gamma > 0:
                            rl = self.D(real_in).sum()
                            grads = torch.autograd.grad(rl, real_in, create_graph=True, retain_graph=True, only_inputs=True)[0]
                            r1_pen = (grads.view(bsz, -1).pow(2).sum(1)).mean() * self.r1_gamma
                            d_loss = d_loss + r1_pen
                    self.scalerD.scale(d_loss).backward()
                    if self.grad_clip > 0:
                        self.scalerD.unscale_(self.optD)
                        torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)
                    self.scalerD.step(self.optD)
                    self.scalerD.update()
                    d_losses.append(d_loss.item())
                
                # Average discriminator loss across multiple steps
                last_d = sum(d_losses) / len(d_losses)

                # Generator (optimizer.zero_grad for consistency with CGAN)
                z = torch.randn(bsz, self.z_dim, device=self.device)
                with autocast(enabled=self.use_amp):
                    fake2 = self.G(z)
                    out_fake = self.D(fake2)
                    g_loss = self.crit(out_fake, torch.ones_like(out_fake))
                self.optG.zero_grad(set_to_none=True)
                self.scalerG.scale(g_loss).backward()
                if self.grad_clip > 0:
                    self.scalerG.unscale_(self.optG)
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip)
                self.scalerG.step(self.optG)
                self.scalerG.update()
                last_g = g_loss.item()
                ema_update()

                d_total += float(last_d)
                g_total += float(last_g)

            pbar.update(1)
            pbar.set_postfix_str(f"D={last_d:.4f} G={last_g:.4f}")

            if self.wb:
                self.wb.log({
                    "tabgan/loss/D": d_total / max(nb, 1),
                    "tabgan/loss/G": g_total / max(nb, 1),
                    "tabgan_step": epoch,
                })

            if ckpt_path and self.save_every and ((epoch + 1) % self.save_every == 0 or epoch == self.epochs - 1):
                tqdm.write(f"[TabGAN] saving checkpoint at epoch {epoch+1}")
                self.log.info(f"saving checkpoint at epoch {epoch+1}")
                self._save_ckpt(ckpt_path, epoch, float(last_g), float(last_d))

        pbar.close()
        total_sec = time.time() - t0
        if self.wb:
            self.wb.log({"tabgan/time/train_total_sec": float(total_sec)})

        self.log.info(f"training finished. total_sec={total_sec:.2f}")
        return {"train_total_sec": total_sec}

    @torch.no_grad()
    def synthesize(self, n: int, columns=None):
        import pandas as pd
        import numpy as np
        n = int(n)
        bs = max(1, int(self.synth_batch_size))
        feats = []
        for i in range(0, n, bs):
            j = min(i + bs, n)
            z = torch.randn(j - i, self.z_dim, device=self.device)
            G = self.G_ema if self.G_ema is not None else self.G
            fake = G(z).detach().cpu().numpy()
            feats.append(fake)
        synth = np.concatenate(feats, axis=0) if feats else np.zeros((0, self.x_dim), dtype=float)
        if columns is None:
            columns = [f"feat_{i}" for i in range(self.x_dim)]
        df = pd.DataFrame(synth, columns=columns)
        self.log.info(f"synthesized features: n={len(df)}")
        return df