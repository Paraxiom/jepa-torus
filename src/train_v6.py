"""V6: Dual-path — 512D prediction + Fourier torus with Karmonic filter.

V1-V5 showed a hard tradeoff:
  - 512D-only prediction: 72% accuracy, 0 topology (V1/V3)
  - Torus-only bottleneck: 20% accuracy, rich topology (V4/V5)

V6 resolves this by running BOTH paths simultaneously:
  1. Standard 512D predictor (encoder -> predictor -> target): preserves accuracy
  2. Fourier torus branch (encoder -> torus_head -> torus_predictor -> target_angles):
     creates class structure on torus
  3. Karmonic-filtered uniformity on Fourier coords: enforces torus topology

The encoder must satisfy both — rich 512D representations AND torus-compatible structure.
This is NOT curriculum (V3) where phases alternate. Both losses are active every step.

Architecture:
    Online:  encoder (512D) ---> predictor_512 ---------> predicted_512
                            \\--> FourierTorusHead (24D) -> FourierPredictor -> predicted_angles
    Target:  EMA encoder ----> target_512
                           \\--> EMA FourierTorusHead --> target_angles, target_fourier

Loss:
    L = L_pred_512 + lambda_torus * L_pred_angles + lambda_std * L_std + lambda_karmonic * L_karmonic

Usage:
    python -m src.train_v6 --config configs/toroidal_v6_dual.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import yaml
except ImportError:
    yaml = None

from .train import ResNetEncoder, Predictor, JEPAAugmentation
from .train_v5 import FourierTorusHead, FourierPredictor, KarmonicFilterLoss
from .toroidal_loss import StandardDeviationLoss


# ---------------------------------------------------------------------------
# V6 Model
# ---------------------------------------------------------------------------


class EBJEPA_V6(nn.Module):
    """V6: Dual-path with shared encoder.

    Path 1 (accuracy): encoder -> predictor_512 -> predicted_512
    Path 2 (topology): encoder -> FourierTorusHead -> FourierPredictor -> predicted_angles

    Both paths share the same encoder. The encoder gets gradients from:
    - 512D prediction loss (class discrimination in 512D)
    - Torus prediction loss (class discrimination on torus angles)
    - Karmonic uniformity (torus topology via Fourier modes)
    - Std loss (collapse prevention)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        ema_decay: float = 0.996,
        torus_dim: int = 2,
        n_modes: int = 6,
        torus_hidden: int = 128,
        predictor_hidden: int = 256,
    ):
        super().__init__()
        self.online_encoder = ResNetEncoder(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.ema_decay = ema_decay

        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.fourier_dim = 2 * torus_dim * n_modes

        # Path 1: 512D prediction (standard JEPA predictor)
        self.predictor = Predictor(embed_dim, hidden_dim)

        # Path 2: Fourier torus branch
        self.torus_projection = FourierTorusHead(
            input_dim=embed_dim,
            hidden_dim=torus_hidden,
            torus_dim=torus_dim,
            n_modes=n_modes,
        )
        self.target_torus_projection = copy.deepcopy(self.torus_projection)

        # Torus predictor: predicts target angles from online Fourier coords
        self.torus_predictor = FourierPredictor(
            fourier_dim=self.fourier_dim,
            torus_dim=torus_dim,
            n_modes=n_modes,
            hidden_dim=predictor_hidden,
        )

        # Freeze targets
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_torus_projection.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """EMA update of target encoder and target torus head."""
        for online, target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)
        for online, target in zip(
            self.torus_projection.parameters(),
            self.target_torus_projection.parameters(),
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """Forward pass through both paths.

        Returns:
            predicted_512: (B, 512) predicted target encoder output
            target_512: (B, 512) actual target encoder output
            predicted_angles: (B, k) predicted target torus angles
            target_angles: (B, k) actual target torus angles
            encoder_output: (B, 512) online encoder output (for std loss)
            online_angles: (B, k) online torus angles
            online_fourier: (B, 2km) online Fourier embedding
        """
        # Online path
        encoder_output = self.online_encoder(x1)

        # Path 1: 512D prediction
        predicted_512 = self.predictor(encoder_output)

        # Path 2: Fourier torus
        online_angles, online_fourier = self.torus_projection(encoder_output)
        predicted_angles, _ = self.torus_predictor(online_fourier)

        # Target path
        with torch.no_grad():
            target_512 = self.target_encoder(x2)
            target_angles, target_fourier = self.target_torus_projection(target_512)

        return (
            predicted_512,
            target_512,
            predicted_angles,
            target_angles,
            encoder_output,
            online_angles,
            online_fourier,
        )


# ---------------------------------------------------------------------------
# V6 Loss
# ---------------------------------------------------------------------------


class DualPathLoss(nn.Module):
    """Loss for V6: dual-path prediction + Karmonic-filtered torus.

    L = L_pred_512
      + lambda_torus_pred * L_pred_angles
      + lambda_std * L_std
      + lambda_karmonic * L_karmonic

    where:
    - L_pred_512: MSE between predicted and target 512D encoder outputs
    - L_pred_angles: circular distance between predicted and target angles
    - L_std: VICReg std loss on 512D encoder
    - L_karmonic: mode-weighted Fourier uniformity + angle spread
    """

    def __init__(
        self,
        torus_dim: int = 2,
        n_modes: int = 6,
        grid_size: int = 12,
        lambda_std: float = 25.0,
        lambda_torus_pred: float = 0.5,
        lambda_karmonic: float = 5.0,
        t_uniformity: float = 2.0,
        spread_weight: float = 1.0,
    ):
        super().__init__()
        self.lambda_std = lambda_std
        self.lambda_torus_pred = lambda_torus_pred
        self.lambda_karmonic = lambda_karmonic

        self.std_loss = StandardDeviationLoss()
        self.karmonic_loss = KarmonicFilterLoss(
            torus_dim=torus_dim,
            n_modes=n_modes,
            grid_size=grid_size,
            t_uniformity=t_uniformity,
            spread_weight=spread_weight,
        )

    def forward(
        self,
        predicted_512: torch.Tensor,
        target_512: torch.Tensor,
        predicted_angles: torch.Tensor,
        target_angles: torch.Tensor,
        encoder_output: torch.Tensor,
        online_angles: torch.Tensor,
        online_fourier: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Path 1: 512D prediction loss
        pred_512_loss = F.mse_loss(predicted_512, target_512.detach())

        # Path 2: torus angle prediction loss (circular distance)
        pred_torus_loss = (
            1.0 - torch.cos(predicted_angles - target_angles.detach())
        ).mean()

        # Std loss on 512D encoder
        std_loss = self.std_loss(encoder_output)

        # Karmonic-filtered uniformity on Fourier coords
        karmonic_losses = self.karmonic_loss(online_angles, online_fourier)

        total = (
            pred_512_loss
            + self.lambda_torus_pred * pred_torus_loss
            + self.lambda_std * std_loss
            + self.lambda_karmonic * karmonic_losses["total"]
        )

        return {
            "total": total,
            "pred_512": pred_512_loss,
            "pred_torus": pred_torus_loss,
            "std": std_loss,
            "uniformity": karmonic_losses["uniformity"],
            "spread": karmonic_losses["spread"],
            "mode_uniformities": karmonic_losses["mode_uniformities"],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_v6(config: dict) -> Path:
    """V6 training: dual-path with Karmonic torus branch."""
    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    model_cfg = config.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 512)
    torus_dim = model_cfg.get("torus_dim", 2)
    n_modes = model_cfg.get("n_modes", 6)
    fourier_dim = 2 * torus_dim * n_modes

    model = EBJEPA_V6(
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        ema_decay=model_cfg.get("ema_decay", 0.996),
        torus_dim=torus_dim,
        n_modes=n_modes,
        torus_hidden=model_cfg.get("torus_hidden", 128),
        predictor_hidden=model_cfg.get("predictor_hidden", 256),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Path 1: encoder (512D) -> predictor (512D)")
    print(f"Path 2: encoder (512D) -> T^{torus_dim} x {n_modes} modes = {fourier_dim}D")

    # Print Karmonic weights
    loss_cfg = config.get("loss", {})
    grid_size = loss_cfg.get("grid_size", 12)
    print(f"\nKarmonic spectral weights (N={grid_size}):")
    eigenvalues = []
    for n in range(1, n_modes + 1):
        lam = 2.0 - 2.0 * math.cos(2.0 * math.pi * n / grid_size)
        eigenvalues.append(lam)
    lam_1 = eigenvalues[0]
    lam_max = max(eigenvalues)
    for n, lam in enumerate(eigenvalues, 1):
        w = (lam - lam_1) / (lam_max - lam_1) if lam_max > lam_1 else 0.0
        print(f"  Mode {n}: lambda={lam:.4f}, weight={w:.4f}")

    # Data
    data_cfg = config.get("data", {})
    augmentation = JEPAAugmentation(image_size=32)
    train_dataset = datasets.CIFAR10(
        root=data_cfg.get("data_dir", "./data"),
        train=True,
        download=True,
        transform=augmentation,
    )
    train_cfg = config.get("training", {})
    dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 256),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # Loss
    criterion = DualPathLoss(
        torus_dim=torus_dim,
        n_modes=n_modes,
        grid_size=grid_size,
        lambda_std=loss_cfg.get("lambda_std", 25.0),
        lambda_torus_pred=loss_cfg.get("lambda_torus_pred", 0.5),
        lambda_karmonic=loss_cfg.get("lambda_karmonic", 5.0),
        t_uniformity=loss_cfg.get("t_uniformity", 2.0),
        spread_weight=loss_cfg.get("spread_weight", 1.0),
    )

    print(f"\nLoss weights:")
    print(f"  pred_512: 1.0 (implicit)")
    print(f"  pred_torus: {loss_cfg.get('lambda_torus_pred', 0.5)}")
    print(f"  std: {loss_cfg.get('lambda_std', 25.0)}")
    print(f"  karmonic: {loss_cfg.get('lambda_karmonic', 5.0)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.get("lr", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )

    total_epochs = train_cfg.get("epochs", 300)
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6,
    )

    out_cfg = config.get("output", {})
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v6_dual"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_loss = float("inf")
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"V6: Dual-Path + Karmonic Torus ({total_epochs} epochs)")
    print(f"  Path 1: 512D prediction (accuracy)")
    print(f"  Path 2: T^{torus_dim} x {n_modes} Fourier (topology)")
    print(f"  Karmonic filter: low modes = class, high modes = torus")
    print(f"{'='*60}\n")

    for epoch in range(1, total_epochs + 1):
        model.train()

        # Warmup
        if epoch <= warmup_epochs:
            warmup_lr = train_cfg.get("lr", 0.001) * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            scheduler.step()

        # EMA decay schedule
        model.ema_decay = 1.0 - (1.0 - model_cfg.get("ema_decay", 0.996)) * (
            math.cos(math.pi * epoch / total_epochs) + 1
        ) / 2

        loss_accum = {}
        n_batches = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            x1, x2 = images
            x1, x2 = x1.to(device), x2.to(device)

            out = model(x1, x2)
            losses = criterion(*out)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.update_target()

            for k, v in losses.items():
                if k == "mode_uniformities":
                    continue
                loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                mu = losses.get("mode_uniformities", [])
                mu_str = " ".join(f"{u:.2f}" for u in mu[:3]) if mu else ""
                print(
                    f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"p512={losses['pred_512'].item():.4f} "
                    f"ptor={losses['pred_torus'].item():.4f} "
                    f"unif={losses['uniformity'].item():.4f} "
                    f"modes=[{mu_str}...]"
                )

        avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
        history.append({"epoch": epoch, **avg_losses})

        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch}/{total_epochs} | "
            f"total={avg_losses['total']:.4f} | "
            f"p512={avg_losses['pred_512']:.4f} | "
            f"ptor={avg_losses['pred_torus']:.4f} | "
            f"unif={avg_losses['uniformity']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"time={elapsed:.0f}s"
        )

        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                },
                out_dir / "best.pt",
            )

        if epoch % 50 == 0 or epoch == total_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "history": history,
                },
                out_dir / f"checkpoint_{epoch:04d}.pt",
            )

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"V6 Dual-Path training complete")
    print(f"  {total_epochs} epochs, {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="V6: Dual-path 512D + Fourier Torus"
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    p = Path(args.config)
    with open(p) as f:
        if p.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("pip install pyyaml")
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    train_v6(config)


if __name__ == "__main__":
    main()
