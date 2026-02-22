"""V8b: Gradient-Scaled Karmonic — partial detach.

V6b (full coupling): encoder acc 34.8%, torus acc 21.0%, β₂=460
V8a (full detach):   encoder acc 32.9%, torus acc 33.8%, β₂=446

The encoder NEEDS some karmonic signal (V6b > V8a on accuracy).
But too much hurts (V6a < V6b). V8b finds the sweet spot:

    torus_input = GradientScale(encoder_output, scale=0.1)

The encoder gets 10% of karmonic gradients — enough to guide
structure, not enough to dominate. The torus head gets full gradients.

Usage:
    python -m src.train_v8b --config configs/toroidal_v8b_scaled.yaml
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
from .train_v6 import DualPathLoss
from .toroidal_loss import StandardDeviationLoss


# ---------------------------------------------------------------------------
# Gradient scaling autograd function
# ---------------------------------------------------------------------------


class GradientScale(torch.autograd.Function):
    """Scale gradients in backward pass without affecting forward pass."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def gradient_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply gradient scaling: forward is identity, backward multiplies by scale."""
    return GradientScale.apply(x, scale)


# ---------------------------------------------------------------------------
# V8b Model
# ---------------------------------------------------------------------------


class EBJEPA_V8b(nn.Module):
    """V8b: Dual-path with gradient-scaled torus branch.

    Same as V6 but torus head receives gradient_scale(encoder_output, 0.1).
    Encoder gets 10% of karmonic/torus gradients — a gentle guide, not a drag.
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
        karmonic_grad_scale: float = 0.1,
    ):
        super().__init__()
        self.online_encoder = ResNetEncoder(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.ema_decay = ema_decay
        self.karmonic_grad_scale = karmonic_grad_scale

        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.fourier_dim = 2 * torus_dim * n_modes

        # Path 1: 512D prediction
        self.predictor = Predictor(embed_dim, hidden_dim)

        # Path 2: Fourier torus branch
        self.torus_projection = FourierTorusHead(
            input_dim=embed_dim,
            hidden_dim=torus_hidden,
            torus_dim=torus_dim,
            n_modes=n_modes,
        )
        self.target_torus_projection = copy.deepcopy(self.torus_projection)

        self.torus_predictor = FourierPredictor(
            fourier_dim=self.fourier_dim,
            torus_dim=torus_dim,
            n_modes=n_modes,
            hidden_dim=predictor_hidden,
        )

        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_torus_projection.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
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
        encoder_output = self.online_encoder(x1)

        # Path 1: 512D prediction (full gradients to encoder)
        predicted_512 = self.predictor(encoder_output)

        # Path 2: Fourier torus — SCALED gradients to encoder
        scaled_input = gradient_scale(encoder_output, self.karmonic_grad_scale)
        online_angles, online_fourier = self.torus_projection(scaled_input)
        predicted_angles, _ = self.torus_predictor(online_fourier)

        # Target
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
# Training
# ---------------------------------------------------------------------------


def train_v8b(config: dict) -> Path:
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
    grad_scale = model_cfg.get("karmonic_grad_scale", 0.1)

    model = EBJEPA_V8b(
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        ema_decay=model_cfg.get("ema_decay", 0.996),
        torus_dim=torus_dim,
        n_modes=n_modes,
        torus_hidden=model_cfg.get("torus_hidden", 128),
        predictor_hidden=model_cfg.get("predictor_hidden", 256),
        karmonic_grad_scale=grad_scale,
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Path 1: encoder (512D) -> predictor (512D) [FULL GRADIENTS]")
    print(f"Path 2: encoder (512D) -> [SCALE={grad_scale}] -> T^{torus_dim} x {n_modes} = {fourier_dim}D")

    loss_cfg = config.get("loss", {})
    grid_size = loss_cfg.get("grid_size", 12)

    data_cfg = config.get("data", {})
    augmentation = JEPAAugmentation(image_size=32)
    train_dataset = datasets.CIFAR10(
        root=data_cfg.get("data_dir", "./data"),
        train=True, download=True, transform=augmentation,
    )
    train_cfg = config.get("training", {})
    dataloader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 256),
        shuffle=True, num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True, drop_last=True,
    )

    criterion = DualPathLoss(
        torus_dim=torus_dim, n_modes=n_modes, grid_size=grid_size,
        lambda_std=loss_cfg.get("lambda_std", 25.0),
        lambda_torus_pred=loss_cfg.get("lambda_torus_pred", 0.5),
        lambda_karmonic=loss_cfg.get("lambda_karmonic", 5.0),
        t_uniformity=loss_cfg.get("t_uniformity", 2.0),
        spread_weight=loss_cfg.get("spread_weight", 1.0),
    )

    print(f"\nLoss weights (encoder sees {grad_scale}x of torus gradients):")
    print(f"  pred_512: 1.0 (full)")
    print(f"  pred_torus: {loss_cfg.get('lambda_torus_pred', 0.5)} (x{grad_scale})")
    print(f"  std: {loss_cfg.get('lambda_std', 25.0)} (full)")
    print(f"  karmonic: {loss_cfg.get('lambda_karmonic', 5.0)} (x{grad_scale})")

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
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v8b_scaled"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_loss = float("inf")
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"V8b: Gradient-Scaled Karmonic ({total_epochs} epochs)")
    print(f"  Encoder: gets {grad_scale}x torus gradients")
    print(f"  Torus head: full gradients")
    print(f"{'='*60}\n")

    for epoch in range(1, total_epochs + 1):
        model.train()

        if epoch <= warmup_epochs:
            warmup_lr = train_cfg.get("lr", 0.001) * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            scheduler.step()

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
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": config},
                out_dir / "best.pt",
            )

        if epoch % 50 == 0 or epoch == total_epochs:
            torch.save(
                {
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config, "history": history,
                },
                out_dir / f"checkpoint_{epoch:04d}.pt",
            )

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"V8b Gradient-Scaled training complete")
    print(f"  {total_epochs} epochs, {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="V8b: Gradient-Scaled Karmonic")
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
    train_v8b(config)


if __name__ == "__main__":
    main()
