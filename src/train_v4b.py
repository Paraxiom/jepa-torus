"""V4b: Higher-dimensional torus bottleneck (T^k instead of T^2).

V4 proved the bottleneck concept works (best β₀=816, encoder eff.rank=4.22)
but T^2 (4D) is too tight for 10-class discrimination (27.81% accuracy).

V4b generalizes to T^k = S¹×...×S¹ (k circles, 2k-dimensional embedding).
T^5 gives 10D — enough capacity for CIFAR-10 while maintaining torus topology.

Usage:
    python -m src.train_v4b --config configs/toroidal_v4b_T5.yaml
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

from .train import ResNetEncoder, JEPAAugmentation
from .toroidal_loss import StandardDeviationLoss


class GeneralizedTorusHead(nn.Module):
    """Projects encoder output onto T^k = S¹×...×S¹ ⊂ R^{2k}.

    Architecture: Linear(512→hidden)→BN→ReLU→Linear(hidden→k)→sigmoid*2π
    Output: k angles → 2k-dim (cos θ₁, sin θ₁, ..., cos θ_k, sin θ_k)

    For k=2: equivalent to TorusProjectionHead (T², 4D)
    For k=5: T⁵, 10D embedding
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128, torus_dim: int = 5):
        super().__init__()
        self.torus_dim = torus_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, torus_dim),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map encoder output to T^k.

        Returns:
            angles: (B, k) raw angles in [0, 2π).
            torus_embed: (B, 2k) interleaved cos/sin pairs.
        """
        raw = self.net(z)  # (B, k)
        angles = 2.0 * math.pi * torch.sigmoid(raw)  # [0, 2π)

        # Build (cos θ₁, sin θ₁, cos θ₂, sin θ₂, ..., cos θ_k, sin θ_k)
        cos_vals = torch.cos(angles)  # (B, k)
        sin_vals = torch.sin(angles)  # (B, k)
        # Interleave: (B, 2k)
        torus_embed = torch.stack([cos_vals, sin_vals], dim=2).reshape(-1, 2 * self.torus_dim)
        return angles, torus_embed


class GeneralizedTorusPredictor(nn.Module):
    """Predicts target T^k coordinates from online T^k coordinates.

    Input: 2k-dim on T^k, Output: 2k-dim on T^k (guaranteed by cos/sin).
    """

    def __init__(self, torus_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.torus_dim = torus_dim
        embed_dim = 2 * torus_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, torus_dim),  # predict k angles
        )

    def forward(self, torus_embed: torch.Tensor) -> torch.Tensor:
        raw = self.net(torus_embed)  # (B, k)
        angles = 2.0 * math.pi * torch.sigmoid(raw)
        cos_vals = torch.cos(angles)
        sin_vals = torch.sin(angles)
        return torch.stack([cos_vals, sin_vals], dim=2).reshape(-1, 2 * self.torus_dim)


class GeneralizedTorusStructureLoss(nn.Module):
    """Uniformity + spread loss for T^k embeddings.

    Uniformity: Wang-Isola log E[exp(-t·||z_i - z_j||²)] in R^{2k}
    Spread: average pairwise circular decorrelation across k angles
    """

    def __init__(self, torus_dim: int = 5, t_uniformity: float = 2.0, spread_weight: float = 1.0):
        super().__init__()
        self.torus_dim = torus_dim
        self.t = t_uniformity
        self.spread_weight = spread_weight

    def forward(self, angles: torch.Tensor, torus_embed: torch.Tensor) -> dict[str, torch.Tensor]:
        B = torus_embed.shape[0]
        k = self.torus_dim

        # --- Uniformity in R^{2k} ---
        sq_dists = torch.cdist(torus_embed, torus_embed, p=2).pow(2)
        mask = ~torch.eye(B, dtype=torch.bool, device=torus_embed.device)
        neg_dists = -self.t * sq_dists
        neg_dists = neg_dists.masked_select(mask).view(B, B - 1)
        uniformity = torch.logsumexp(neg_dists, dim=1).mean() - math.log(B - 1)

        # --- Spread: pairwise circular decorrelation ---
        # For each pair (θ_i, θ_j), compute circular correlation and penalize
        spread = torch.tensor(0.0, device=angles.device)
        n_pairs = 0
        for i in range(k):
            for j in range(i + 1, k):
                theta_i = angles[:, i]
                theta_j = angles[:, j]
                mu_i = torch.atan2(torch.sin(theta_i).mean(), torch.cos(theta_i).mean())
                mu_j = torch.atan2(torch.sin(theta_j).mean(), torch.cos(theta_j).mean())
                s_i = torch.sin(theta_i - mu_i)
                s_j = torch.sin(theta_j - mu_j)
                num = (s_i * s_j).mean()
                den = torch.sqrt(s_i.pow(2).mean() * s_j.pow(2).mean() + 1e-8)
                spread = spread + (num / den).pow(2)
                n_pairs += 1
        if n_pairs > 0:
            spread = spread / n_pairs

        total = uniformity + self.spread_weight * spread
        return {"uniformity": uniformity, "spread": spread, "total": total}


class EBJEPA_V4b(nn.Module):
    """V4b: Prediction through T^k bottleneck (generalized torus)."""

    def __init__(
        self,
        embed_dim: int = 512,
        ema_decay: float = 0.996,
        torus_dim: int = 5,
        torus_hidden: int = 128,
        predictor_hidden: int = 128,
    ):
        super().__init__()
        self.online_encoder = ResNetEncoder(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.ema_decay = ema_decay
        self.torus_dim = torus_dim

        self.torus_projection = GeneralizedTorusHead(
            input_dim=embed_dim, hidden_dim=torus_hidden, torus_dim=torus_dim,
        )
        self.target_torus_projection = copy.deepcopy(self.torus_projection)

        self.torus_predictor = GeneralizedTorusPredictor(
            torus_dim=torus_dim, hidden_dim=predictor_hidden,
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
            self.torus_projection.parameters(), self.target_torus_projection.parameters()
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        encoder_output = self.online_encoder(x1)
        online_angles, online_torus = self.torus_projection(encoder_output)
        predicted_torus = self.torus_predictor(online_torus)

        with torch.no_grad():
            target_enc = self.target_encoder(x2)
            _, target_torus = self.target_torus_projection(target_enc)

        return predicted_torus, target_torus, encoder_output, online_angles, online_torus


class ToroidalBottleneckLossV4b(nn.Module):
    """Loss for V4b with T^k bottleneck."""

    def __init__(
        self,
        torus_dim: int = 5,
        lambda_std: float = 5.0,
        lambda_torus: float = 25.0,
        t_uniformity: float = 2.0,
        spread_weight: float = 1.0,
    ):
        super().__init__()
        self.lambda_std = lambda_std
        self.lambda_torus = lambda_torus
        self.std_loss = StandardDeviationLoss()
        self.torus_loss = GeneralizedTorusStructureLoss(
            torus_dim=torus_dim, t_uniformity=t_uniformity, spread_weight=spread_weight,
        )

    def forward(self, predicted_torus, target_torus, encoder_output, online_angles, online_torus):
        pred_loss = F.mse_loss(predicted_torus, target_torus.detach())
        std_loss = self.std_loss(encoder_output)
        torus_losses = self.torus_loss(online_angles, online_torus)

        total = (
            pred_loss
            + self.lambda_std * std_loss
            + self.lambda_torus * torus_losses["total"]
        )
        return {
            "total": total,
            "prediction": pred_loss,
            "std": std_loss,
            "uniformity": torus_losses["uniformity"],
            "spread": torus_losses["spread"],
        }


def train_v4b(config: dict) -> Path:
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
    torus_dim = model_cfg.get("torus_dim", 5)

    model = EBJEPA_V4b(
        embed_dim=embed_dim,
        ema_decay=model_cfg.get("ema_decay", 0.996),
        torus_dim=torus_dim,
        torus_hidden=model_cfg.get("torus_hidden", 128),
        predictor_hidden=model_cfg.get("predictor_hidden", 128),
    ).to(device)

    # Load pretrained encoder
    pretrained_path = config.get("pretrained_checkpoint")
    if pretrained_path:
        print(f"\nLoading pretrained encoder: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        pretrained_state = ckpt["model_state_dict"]
        own_state = model.state_dict()
        loaded = 0
        for key, value in pretrained_state.items():
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key] = value
                loaded += 1
        model.load_state_dict(own_state)
        print(f"Loaded {loaded} pretrained weight tensors")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Torus: T^{torus_dim} ({2*torus_dim}D embedding on {'×'.join(['S¹']*torus_dim)})")

    # Data
    data_cfg = config.get("data", {})
    augmentation = JEPAAugmentation(image_size=32)
    train_dataset = datasets.CIFAR10(
        root=data_cfg.get("data_dir", "./data"), train=True, download=True, transform=augmentation,
    )
    train_cfg = config.get("training", {})
    dataloader = DataLoader(
        train_dataset, batch_size=train_cfg.get("batch_size", 256), shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), pin_memory=True, drop_last=True,
    )

    # Loss
    loss_cfg = config.get("loss", {})
    criterion = ToroidalBottleneckLossV4b(
        torus_dim=torus_dim,
        lambda_std=loss_cfg.get("lambda_std", 5.0),
        lambda_torus=loss_cfg.get("lambda_torus", 25.0),
        t_uniformity=loss_cfg.get("t_uniformity", 2.0),
        spread_weight=loss_cfg.get("spread_weight", 1.0),
    )

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
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v4b_T5"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_loss = float("inf")
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"V4b: T^{torus_dim} Bottleneck ({total_epochs} epochs)")
    print(f"  encoder -> T^{torus_dim} ({2*torus_dim}D) -> predictor -> target T^{torus_dim}")
    print(f"  lambda_std={loss_cfg.get('lambda_std', 5.0)}, "
          f"lambda_torus={loss_cfg.get('lambda_torus', 25.0)}")
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

            predicted_torus, target_torus, encoder_output, online_angles, online_torus = model(x1, x2)
            losses = criterion(predicted_torus, target_torus, encoder_output, online_angles, online_torus)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target()

            for k, v in losses.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                      f"loss={losses['total'].item():.4f} "
                      f"pred={losses['prediction'].item():.4f} "
                      f"unif={losses['uniformity'].item():.4f}")

        avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
        history.append({"epoch": epoch, **avg_losses})

        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch}/{total_epochs} | "
            f"loss={avg_losses['total']:.4f} | "
            f"pred={avg_losses['prediction']:.4f} | "
            f"unif={avg_losses['uniformity']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"time={elapsed:.0f}s"
        )

        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, out_dir / "best.pt")

        if epoch % 50 == 0 or epoch == total_epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "history": history,
            }, out_dir / f"checkpoint_{epoch:04d}.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"V4b T^{torus_dim} training complete")
    print(f"  {total_epochs} epochs, {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="V4b: Higher-dimensional torus bottleneck")
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
    train_v4b(config)


if __name__ == "__main__":
    main()
