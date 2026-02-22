"""V8c: Contrastive Torus — class-aware toroidal structure.

Replace Karmonic uniformity (push ALL points apart) with supervised
contrastive loss on the torus (same class close, different class far).

This aligns topology with class structure instead of fighting it.
The torus geometry becomes directly meaningful: each class occupies
a distinct region on the torus, naturally creating topology.

Architecture: same as V8a (detached torus) but with contrastive loss.

Usage:
    python -m src.train_v8c --config configs/toroidal_v8c_contrastive.yaml
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
# Contrastive Torus Loss
# ---------------------------------------------------------------------------


class ContrastiveTorusLoss(nn.Module):
    """Supervised contrastive loss on torus angles.

    For each sample i with label y_i:
    - Positive pairs: samples j where y_j == y_i (pull close on torus)
    - Negative pairs: samples j where y_j != y_i (push far on torus)

    Uses circular distance: d(θ_i, θ_j) = 1 - cos(θ_i - θ_j) averaged over dims.
    Temperature-scaled softmax like SupCon.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, angles: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Args:
            angles: (B, k) torus angles in [0, 2π]
            labels: (B,) integer class labels

        Returns:
            dict with 'contrastive' loss and diagnostics
        """
        B = angles.shape[0]
        device = angles.device

        # Pairwise circular similarity: cos(θ_i - θ_j) averaged over dims
        # Shape: (B, B)
        diff = angles.unsqueeze(0) - angles.unsqueeze(1)  # (B, B, k)
        sim = torch.cos(diff).mean(dim=-1)  # (B, B) in [-1, 1]

        # Scale by temperature
        sim = sim / self.temperature

        # Mask: same class = positive, different class = negative
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        # Remove self-similarity
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        positives = labels_eq & self_mask
        negatives = ~labels_eq & self_mask

        # SupCon loss: for each anchor, log(sum_pos / (sum_pos + sum_neg))
        # Numerically stable version
        exp_sim = torch.exp(sim) * self_mask.float()

        pos_sum = (exp_sim * positives.float()).sum(dim=1)  # (B,)
        all_sum = exp_sim.sum(dim=1)  # (B,)

        # Avoid log(0)
        loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8)

        # Only count anchors that have at least one positive
        has_pos = positives.any(dim=1)
        loss = loss[has_pos].mean() if has_pos.any() else torch.tensor(0.0, device=device)

        # Diagnostics
        with torch.no_grad():
            raw_sim = torch.cos(diff).mean(dim=-1)
            pos_sim = (raw_sim * positives.float()).sum() / positives.float().sum().clamp(min=1)
            neg_sim = (raw_sim * negatives.float()).sum() / negatives.float().sum().clamp(min=1)

        return {
            "contrastive": loss,
            "pos_similarity": pos_sim.item(),
            "neg_similarity": neg_sim.item(),
        }


# ---------------------------------------------------------------------------
# V8c Loss: p512 + contrastive torus + std
# ---------------------------------------------------------------------------


class ContrastiveDualPathLoss(nn.Module):
    """Loss for V8c: accuracy path + contrastive torus.

    L = L_pred_512
      + lambda_contrastive * L_contrastive_torus
      + lambda_std * L_std
      + lambda_spread * L_spread (mild uniformity to prevent collapse)
    """

    def __init__(
        self,
        torus_dim: int = 2,
        n_modes: int = 6,
        lambda_std: float = 25.0,
        lambda_contrastive: float = 1.0,
        lambda_spread: float = 0.5,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.lambda_std = lambda_std
        self.lambda_contrastive = lambda_contrastive
        self.lambda_spread = lambda_spread

        self.std_loss = StandardDeviationLoss()
        self.contrastive_loss = ContrastiveTorusLoss(temperature=temperature)

    def forward(
        self,
        predicted_512: torch.Tensor,
        target_512: torch.Tensor,
        encoder_output: torch.Tensor,
        online_angles: torch.Tensor,
        online_fourier: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        # Path 1: 512D prediction
        pred_512_loss = F.mse_loss(predicted_512, target_512.detach())

        # Std loss on encoder
        std_loss = self.std_loss(encoder_output)

        # Contrastive torus loss (uses labels)
        cont_result = self.contrastive_loss(online_angles, labels)
        cont_loss = cont_result["contrastive"]

        # Mild spread to prevent all angles collapsing to same point
        # Variance of angles across batch (per dim)
        angle_var = torch.var(online_angles, dim=0).mean()
        spread_loss = F.relu(1.0 - angle_var)  # penalize if variance < 1.0

        total = (
            pred_512_loss
            + self.lambda_contrastive * cont_loss
            + self.lambda_std * std_loss
            + self.lambda_spread * spread_loss
        )

        return {
            "total": total,
            "pred_512": pred_512_loss,
            "contrastive": cont_loss,
            "std": std_loss,
            "spread": spread_loss,
            "pos_sim": cont_result["pos_similarity"],
            "neg_sim": cont_result["neg_similarity"],
        }


# ---------------------------------------------------------------------------
# V8c Model — detached torus with contrastive (no predictor needed)
# ---------------------------------------------------------------------------


class EBJEPA_V8c(nn.Module):
    """V8c: Contrastive torus — class-aware toroidal structure.

    Same encoder + torus head as V8a (detached), but:
    - No torus predictor (not predicting target angles)
    - Instead, contrastive loss on online angles using class labels
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        ema_decay: float = 0.996,
        torus_dim: int = 2,
        n_modes: int = 6,
        torus_hidden: int = 128,
    ):
        super().__init__()
        self.online_encoder = ResNetEncoder(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.ema_decay = ema_decay

        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.fourier_dim = 2 * torus_dim * n_modes

        # Path 1: 512D prediction
        self.predictor = Predictor(embed_dim, hidden_dim)

        # Path 2: Fourier torus (detached, contrastive)
        self.torus_projection = FourierTorusHead(
            input_dim=embed_dim,
            hidden_dim=torus_hidden,
            torus_dim=torus_dim,
            n_modes=n_modes,
        )

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        for online, target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        encoder_output = self.online_encoder(x1)

        # Path 1: 512D prediction (full gradients)
        predicted_512 = self.predictor(encoder_output)

        # Path 2: Torus (detached from encoder)
        online_angles, online_fourier = self.torus_projection(encoder_output.detach())

        # Target (only 512D needed)
        with torch.no_grad():
            target_512 = self.target_encoder(x2)

        return (
            predicted_512,
            target_512,
            encoder_output,
            online_angles,
            online_fourier,
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_v8c(config: dict) -> Path:
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

    model = EBJEPA_V8c(
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        ema_decay=model_cfg.get("ema_decay", 0.996),
        torus_dim=torus_dim,
        n_modes=n_modes,
        torus_hidden=model_cfg.get("torus_hidden", 128),
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Path 1: encoder (512D) -> predictor (512D) [FULL GRADIENTS]")
    print(f"Path 2: encoder (512D) -> [DETACH] -> T^{torus_dim} x {n_modes} = {fourier_dim}D [CONTRASTIVE]")

    loss_cfg = config.get("loss", {})

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

    criterion = ContrastiveDualPathLoss(
        torus_dim=torus_dim,
        n_modes=n_modes,
        lambda_std=loss_cfg.get("lambda_std", 25.0),
        lambda_contrastive=loss_cfg.get("lambda_contrastive", 1.0),
        lambda_spread=loss_cfg.get("lambda_spread", 0.5),
        temperature=loss_cfg.get("temperature", 0.1),
    )

    print(f"\nLoss weights:")
    print(f"  pred_512: 1.0 (encoder)")
    print(f"  contrastive: {loss_cfg.get('lambda_contrastive', 1.0)} (torus, uses labels)")
    print(f"  std: {loss_cfg.get('lambda_std', 25.0)} (encoder)")
    print(f"  spread: {loss_cfg.get('lambda_spread', 0.5)} (torus)")

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
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v8c_contrastive"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_loss = float("inf")
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"V8c: Contrastive Torus ({total_epochs} epochs)")
    print(f"  Encoder: p512 + std (self-supervised)")
    print(f"  Torus: contrastive + spread (supervised)")
    print(f"  Same class -> close on torus, different -> far")
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

        for batch_idx, (images, labels) in enumerate(dataloader):
            x1, x2 = images
            x1, x2 = x1.to(device), x2.to(device)
            labels = labels.to(device)

            out = model(x1, x2)
            # out = (predicted_512, target_512, encoder_output, online_angles, online_fourier)
            losses = criterion(*out, labels=labels)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target()

            for k, v in losses.items():
                if isinstance(v, (int, float)):
                    loss_accum[k] = loss_accum.get(k, 0.0) + v
                elif torch.is_tensor(v):
                    loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                print(
                    f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"p512={losses['pred_512'].item():.4f} "
                    f"cont={losses['contrastive'].item():.4f} "
                    f"pos={losses['pos_sim']:.3f} "
                    f"neg={losses['neg_sim']:.3f}"
                )

        avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
        history.append({"epoch": epoch, **avg_losses})

        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch}/{total_epochs} | "
            f"total={avg_losses['total']:.4f} | "
            f"p512={avg_losses['pred_512']:.4f} | "
            f"cont={avg_losses['contrastive']:.4f} | "
            f"pos={avg_losses['pos_sim']:.3f} neg={avg_losses['neg_sim']:.3f} | "
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
    print(f"V8c Contrastive Torus training complete")
    print(f"  {total_epochs} epochs, {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="V8c: Contrastive Torus")
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
    train_v8c(config)


if __name__ == "__main__":
    main()
