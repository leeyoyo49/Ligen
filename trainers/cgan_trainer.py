from typing import Optional
from types import SimpleNamespace
import os
import time

import torch
import torch.nn as nn
# Robust AMP import across environments
try:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
except Exception:  # fallback for CPU-only or PyTorch>=2 environments
    from torch.amp import autocast, GradScaler  # type: ignore

from models.cgan import CGANGenerator, CGANDiscriminator
from trainers.common import ETAMeter, format_hms  # kept in case used elsewhere

from tqdm.auto import tqdm
from utils.logger import get_logger


def dict_to_ns(d: dict):
    return SimpleNamespace(**d)


class CGANTrainer:
    """
    Conditional GAN: DataLoader must yield (coords [B,coord_dim], feats [B,feat_dim]).
    """

    def __init__(
        self,
        coord_dim: int,
        feat_dim: int,
        cfg_cgan: dict,          # 🔑 接 dict
        device: str = "cuda",
        wb=None,
    ):
        self.coord_dim = coord_dim
        self.feat_dim = feat_dim
        self.cfg = dict_to_ns(cfg_cgan)      # 🔑 dict → Namespace
        self.device = torch.device(device)
        self.wb = wb
        self.log = get_logger("trainer.cgan")

        # hyperparameters
        self.noise_dim = getattr(self.cfg, "noise_dim", 10)
        # TTUR learning rates
        self.lr_g = getattr(self.cfg, "lr_g", getattr(self.cfg, "lr", 1e-4))
        self.lr_d = getattr(self.cfg, "lr_d", getattr(self.cfg, "lr", 1e-4))
        self.betas = tuple(getattr(self.cfg, "betas", (0.5, 0.999)))
        self.d_steps = getattr(self.cfg, "d_steps", 2)
        self.fid_w = getattr(self.cfg, "fid_mse_weight", 0.1)
        self.epochs = getattr(self.cfg, "epochs", 5000)
        self.save_every = getattr(self.cfg, "save_every", None)
        self.label_smooth = float(getattr(self.cfg, "label_smooth", 0.0) or 0.0)
        self.inst_noise_std = float(getattr(self.cfg, "instance_noise_std", 0.0) or 0.0)
        self.r1_gamma = float(getattr(self.cfg, "r1_gamma", 0.0) or 0.0)  # 0 disables
        self.grad_clip = float(getattr(self.cfg, "grad_clip", 0.0) or 0.0)
        self.use_amp = bool(getattr(self.cfg, "amp", True))
        self.ema_decay = float(getattr(self.cfg, "ema_decay", 0.0) or 0.0)
        # adaptive balancing: adjust D steps based on D loss
        self.adaptive_balance = bool(getattr(self.cfg, "adaptive_balance", False))
        self.d_loss_low = float(getattr(self.cfg, "d_loss_low", 0.35))
        self.d_loss_high = float(getattr(self.cfg, "d_loss_high", 0.85))

    # models and optimizers
        self.G = CGANGenerator(self.noise_dim, coord_dim, feat_dim, cfg=self.cfg).to(self.device)
        self.D = CGANDiscriminator(coord_dim, feat_dim, cfg=self.cfg).to(self.device)
        # BCE with logits for numerical stability (no Sigmoid in D)
        self.bce_logits = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.optG = torch.optim.Adam(self.G.parameters(), lr=self.lr_g, betas=self.betas)
        self.optD = torch.optim.Adam(self.D.parameters(), lr=self.lr_d, betas=self.betas)

        # AMP scaler
        self.scalerG = GradScaler(enabled=self.use_amp)
        self.scalerD = GradScaler(enabled=self.use_amp)

        # optional EMA for generator
        self.G_ema = None
        if self.ema_decay > 0:
            import copy
            self.G_ema = copy.deepcopy(self.G).eval()
            for p in self.G_ema.parameters():
                p.requires_grad_(False)

        # cache training dataset and control synth batch size
        self.train_dataset = None
        self.synth_batch_size = getattr(self.cfg, "synth_batch_size", 65536)

        # one-time W&B metric definition guard
        self._wb_defined = False

        self.log.info(
            "initialized CGAN: coord_dim=%d feat_dim=%d noise_dim=%d lr_g=%.2e lr_d=%.2e d_steps=%d epochs=%d"
            % (coord_dim, feat_dim, self.noise_dim, self.lr_g, self.lr_d, self.d_steps, self.epochs)
        )

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
        tqdm.write(f"[CGAN] checkpoint saved @ epoch {epoch} → {path}")
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

    def train(self, loader: torch.utils.data.DataLoader, ckpt_path: Optional[str] = None, resume: bool = True):
        start_epoch = self._maybe_resume(ckpt_path) if resume else 0
        self.log.info(
            f"start training: epochs={self.epochs} d_steps={self.d_steps} lr_g={self.lr_g} lr_d={self.lr_d} "
            f"noise_dim={self.noise_dim} device={self.device} amp={self.use_amp}"
        )

        # remember dataset for later synthesis
        self.train_dataset = getattr(loader, "dataset", None)

        # tqdm 進度條（按 epoch 走）
        pbar = tqdm(total=self.epochs, initial=start_epoch, desc="[CGAN] Training", dynamic_ncols=True)

        # define W&B step metric here as well (defensive, in case not defined by caller)
        if self.wb and not self._wb_defined:
            try:
                self.wb.define_metric("cgan/*", step_metric="cgan_step")
            except Exception as e:
                self.log.debug(f"wandb.define_metric failed: {e}")
            self._wb_defined = True

        t0 = time.time()
        last_d = 0.0
        last_g = 0.0

        def ones(b):
            val = 1.0 - self.label_smooth if self.label_smooth > 0 else 1.0
            return torch.full((b, 1), val, device=self.device)
        def zeros(b):
            return torch.zeros(b, 1, device=self.device)

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
            for coords, feats in loader:
                coords = coords.to(self.device).float()
                feats = feats.to(self.device).float()
                bsz = coords.size(0)

                # 1) Train D (可能多步)
                d_reps = self.d_steps
                if self.adaptive_balance:
                    # if D is too strong (low loss), reduce its steps; if too weak (high loss), increase steps
                    if last_d < self.d_loss_low:
                        d_reps = max(1, d_reps - 1)
                    elif last_d > self.d_loss_high:
                        d_reps = d_reps + 1
                d_losses = []
                for _ in range(d_reps):
                    z = torch.randn(bsz, self.noise_dim, device=self.device)
                    with autocast(enabled=self.use_amp):
                        fake = self.G(z, coords).detach()

                        self.optD.zero_grad(set_to_none=True)
                        real_in = add_instance_noise(feats)
                        if self.r1_gamma > 0:
                            real_in.requires_grad_(True)
                        fake_in = add_instance_noise(fake)
                        real_logits = self.D(coords, real_in)
                        fake_logits = self.D(coords, fake_in)

                        d_real = self.bce_logits(real_logits, ones(bsz))
                        d_fake = self.bce_logits(fake_logits, zeros(bsz))
                        d_loss = (d_real + d_fake) * 0.5

                        # optional R1 gradient penalty on real samples
                        if self.r1_gamma > 0:
                            grads = torch.autograd.grad(
                                outputs=real_logits.sum(), inputs=real_in,
                                create_graph=True, retain_graph=True, only_inputs=True
                            )[0]
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

                # 2) Train G
                z = torch.randn(bsz, self.noise_dim, device=self.device)
                with autocast(enabled=self.use_amp):
                    fake = self.G(z, coords)
                    g_adv = self.bce_logits(self.D(coords, fake), ones(bsz))
                    # feature/consistency term（與真實特徵接近）
                    g_fid = self.mse(fake, feats)
                    g_loss = g_adv + self.fid_w * g_fid

                self.optG.zero_grad(set_to_none=True)
                self.scalerG.scale(g_loss).backward()
                if self.grad_clip > 0:
                    self.scalerG.unscale_(self.optG)
                    torch.nn.utils.clip_grad_norm_(self.G.parameters(), self.grad_clip)
                self.scalerG.step(self.optG)
                self.scalerG.update()
                last_g = g_loss.item()
                ema_update()

            # 更新進度條與即時顯示
            pbar.update(1)
            pbar.set_postfix_str(f"D={last_d:.4f} G={last_g:.4f}")

            # W&B 紀錄
            if self.wb:
                self.wb.log({
                    "cgan/loss/D": float(last_d),
                    "cgan/loss/G": float(last_g),
                    "cgan_step": epoch,
                })

            # 存檢查點時用 tqdm.write() 避免進度條被打亂
            if ckpt_path and self.save_every and (
                (epoch + 1) % self.save_every == 0 or epoch == self.epochs - 1
            ):
                tqdm.write(f"[CGAN] saving checkpoint at epoch {epoch+1}")
                self.log.info(f"saving checkpoint at epoch {epoch+1}")
                self._save_ckpt(ckpt_path, epoch, float(last_g), float(last_d))
                # optionally upload checkpoint to W&B as artifact
                if self.wb and getattr(self.cfg, "log_ckpt_to_wandb", False):
                    try:
                        art = self.wb.Artifact("cgan-checkpoint", type="model")
                        art.add_file(ckpt_path)
                        self.wb.log_artifact(art)
                    except Exception as e:
                        self.log.debug(f"W&B artifact upload failed: {e}")

        pbar.close()

        total_sec = time.time() - t0
        if self.wb:
            self.wb.log({"cgan/time/train_total_sec": float(total_sec)})
            # NEW: table for single-value train-time summary
            try:
                time_tbl = self.wb.Table(columns=["train_total_sec"])
                time_tbl.add_data(float(total_sec))
                self.wb.log({"cgan/time/train_summary_table": time_tbl})
            except Exception as e:
                self.log.debug(f"W&B time table log failed: {e}")

        self.log.info(f"training finished. total_sec={total_sec:.2f}")
        return {"train_total_sec": total_sec}

    @torch.no_grad()
    def synthesize_grid(self, synth_multi: int, columns=None):
        """
        Synthesize using all unique (X, Y) pairs found in the original training dataset.
        synth_multi: multiplier applied to the maximum occurrence count of any unique (X, Y) to
                     determine how many samples to generate per unique coordinate.
        Returns (feats_df, coords_np).
        """
        import pandas as pd
        import numpy as np

        if self.train_dataset is None:
            raise ValueError("No training dataset cached. Call train(...) first so we can read all (X,Y) pairs.")

        # collect coords from the original dataset
        coords_list = []
        for i in range(len(self.train_dataset)):
            sample = self.train_dataset[i]
            # dataset expected to return (coords, feats) as used in train()
            c = sample[0]
            c = torch.as_tensor(c).flatten()[: self.coord_dim]
            coords_list.append(c)

        if not coords_list:
            raise ValueError("Training dataset has no coordinates to synthesize from.")

        # on CPU for unique + counts
        coords = torch.stack(coords_list, dim=0).to(dtype=torch.float32, device="cpu")
        coords_np_all = coords.numpy()

        # unique (X, Y) pairs with counts; robust via numpy
        coords_unique_np, counts = np.unique(coords_np_all, axis=0, return_counts=True)
        max_count = int(counts.max()) if counts.size > 0 else 1

        # compute effective repeats = synth_multi * max_count
        synth_multi = max(1, int(synth_multi))
        repeats = max(1, synth_multi * max_count)

        coords_unique = torch.from_numpy(coords_unique_np).to(dtype=torch.float32, device="cpu")
        coords_rep = coords_unique.repeat_interleave(repeats, dim=0)  # [num_unique*repeats, coord_dim]
        n = coords_rep.size(0)

        self.G.eval()
        feats_chunks = []
        bs = int(self.synth_batch_size)
        coords_rep_dev = coords_rep.to(self.device)
        for i in range(0, n, bs):
            j = min(i + bs, n)
            z = torch.randn(j - i, self.noise_dim, device=self.device)
            G = self.G_ema if getattr(self, "G_ema", None) is not None else self.G
            fake = G(z, coords_rep_dev[i:j]).detach().cpu()
            feats_chunks.append(fake)

        feats = torch.cat(feats_chunks, dim=0).numpy()
        coords_np = coords_rep.detach().cpu().numpy()

        if columns is None:
            columns = [f"feat_{i}" for i in range(self.feat_dim)]
        feats_df = pd.DataFrame(feats, columns=columns)

        # log synthesis stats to W&B
        if self.wb:
            try:
                self.wb.log({
                    "cgan/synth/unique_pairs": int(coords_unique.size(0)),
                    "cgan/synth/synth_multi": int(synth_multi),
                    "cgan/synth/max_count": int(max_count),
                    "cgan/synth/repeats": int(repeats),
                    "cgan/synth/samples": int(n),
                })
                # NEW: table for single-value synth stats
                synth_tbl = self.wb.Table(columns=[
                    "unique_pairs","synth_multi","max_count","repeats","samples"
                ])
                synth_tbl.add_data(
                    int(coords_unique.size(0)), int(synth_multi),
                    int(max_count), int(repeats), int(n)
                )
                self.wb.log({"cgan/synth/stats_table": synth_tbl})
            except Exception as e:
                self.log.debug(f"W&B log synth stats failed: {e}")

        self.log.info(f"synthesized from dataset coords: unique_pairs={coords_unique.size(0)} "
                      f"synth_multi={synth_multi} max_count={max_count} repeats={repeats} samples={n}")
        return feats_df, coords_np