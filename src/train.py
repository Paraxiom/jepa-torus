"""EB-JEPA training with toroidal regularization.

Implements a simplified EB-JEPA (Energy-Based JEPA) with:
- Online encoder + EMA target encoder
- Linear predictor with masking
- Pluggable loss: VICReg, SIGReg, or Toroidal

Designed for CIFAR-10 with ResNet-18 backbone. Runs on single GPU.

Usage:
    python -m src.train --config configs/toroidal_N12.yaml
    python -m src.train --config configs/baseline_vicreg.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

try:
    import yaml
except ImportError:
    yaml = None

from .toroidal_loss import ToroidalJEPALoss, ToroidalJEPALossV2, VICRegLoss, TorusProjectionHead


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class ResNetEncoder(nn.Module):
    """ResNet-18 encoder (removes classification head)."""

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).flatten(1)  # (B, 512)
        return self.projector(h)  # (B, embed_dim)


class Predictor(nn.Module):
    """Linear predictor for JEPA."""

    def __init__(self, embed_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EBJEPA(nn.Module):
    """Energy-Based JEPA with online encoder, EMA target, and predictor.

    The predictor maps online encoder output to target encoder space.
    Target encoder is updated via exponential moving average.

    When torus_head=True, adds a TorusProjectionHead branch that maps
    encoder output to S¹×S¹ ⊂ R⁴ for topology-aware training.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        ema_decay: float = 0.996,
        torus_head: bool = False,
    ):
        super().__init__()
        self.online_encoder = ResNetEncoder(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.predictor = Predictor(embed_dim, hidden_dim)
        self.ema_decay = ema_decay

        self.torus_projection = None
        if torus_head:
            self.torus_projection = TorusProjectionHead(input_dim=embed_dim)

        # Freeze target encoder
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """EMA update of target encoder."""
        for online, target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """Forward pass with two augmented views.

        Returns (3-tuple or 5-tuple):
            predictions: Predictor output from view 1 (B, D)
            targets: Target encoder output from view 2 (B, D)
            encoder_output: Online encoder output from view 1 (B, D)
            torus_angles: (B, 2) angles on T² — only when torus_head is active
            torus_embed: (B, 4) cos/sin on S¹×S¹ — only when torus_head is active
        """
        encoder_output = self.online_encoder(x1)
        predictions = self.predictor(encoder_output)

        with torch.no_grad():
            targets = self.target_encoder(x2)

        if self.torus_projection is not None:
            torus_angles, torus_embed = self.torus_projection(encoder_output)
            return predictions, targets, encoder_output, torus_angles, torus_embed

        return predictions, targets, encoder_output


# ---------------------------------------------------------------------------
# Data augmentation (JEPA-style for CIFAR-10)
# ---------------------------------------------------------------------------


class JEPAAugmentation:
    """Two-crop augmentation for JEPA training."""

    def __init__(self, image_size: int = 32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def cosine_schedule(epoch: int, total_epochs: int, base_value: float, final_value: float) -> float:
    """Cosine annealing schedule."""
    return final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * epoch / total_epochs))


def train_one_epoch(
    model: EBJEPA,
    criterion,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> dict:
    """Train for one epoch. Returns dict of average losses."""
    model.train()
    loss_accum = {}
    n_batches = 0

    for batch_idx, (images, _) in enumerate(dataloader):
        x1, x2 = images
        x1, x2 = x1.to(device), x2.to(device)

        outputs = model(x1, x2)
        if len(outputs) == 5:
            predictions, targets, encoder_output, torus_angles, torus_embed = outputs
            losses = criterion(
                predictions, targets, encoder_output,
                torus_angles=torus_angles, torus_embed=torus_embed,
            )
        else:
            predictions, targets, encoder_output = outputs
            losses = criterion(predictions, targets, encoder_output)

        optimizer.zero_grad()
        losses["total"].backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.update_target()

        for k, v in losses.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
        n_batches += 1

        if batch_idx % 100 == 0:
            total_loss = losses["total"].item()
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] loss={total_loss:.4f}")

    return {k: v / max(n_batches, 1) for k, v in loss_accum.items()}


def train(config: dict) -> Path:
    """Full training run from config dict. Returns path to saved checkpoint.

    Config keys:
        model.embed_dim, model.hidden_dim, model.ema_decay
        training.epochs, training.batch_size, training.lr, training.weight_decay
        training.warmup_epochs, training.seed
        loss.type ('vicreg' | 'toroidal'), loss.lambda_std, loss.lambda_cov/lambda_torus
        loss.grid_size, loss.penalty_mode
        data.dataset ('cifar10'), data.data_dir
        output.dir, output.save_every
    """
    # Seed
    seed = config.get("training", {}).get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Model
    model_cfg = config.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 512)
    loss_cfg = config.get("loss", {})
    loss_type = loss_cfg.get("type", "toroidal")
    use_torus_head = loss_type == "toroidal_v2"

    model = EBJEPA(
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        ema_decay=model_cfg.get("ema_decay", 0.996),
        torus_head=use_torus_head,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    if use_torus_head:
        print("Torus projection head: ENABLED (S¹×S¹ ⊂ R⁴)")

    # Loss
    if loss_type == "vicreg":
        criterion = VICRegLoss(
            embed_dim=embed_dim,
            lambda_std=loss_cfg.get("lambda_std", 25.0),
            lambda_cov=loss_cfg.get("lambda_cov", 1.0),
        )
    elif loss_type == "toroidal":
        criterion = ToroidalJEPALoss(
            grid_size=loss_cfg.get("grid_size", 12),
            embed_dim=embed_dim,
            lambda_std=loss_cfg.get("lambda_std", 25.0),
            lambda_torus=loss_cfg.get("lambda_torus", 1.0),
            penalty_mode=loss_cfg.get("penalty_mode", "distance"),
        )
    elif loss_type == "toroidal_v2":
        criterion = ToroidalJEPALossV2(
            embed_dim=embed_dim,
            lambda_std=loss_cfg.get("lambda_std", 10.0),
            lambda_torus=loss_cfg.get("lambda_torus", 10.0),
            t_uniformity=loss_cfg.get("t_uniformity", 2.0),
            spread_weight=loss_cfg.get("spread_weight", 1.0),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    print(f"Loss: {loss_type}")

    # Data
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "./data")
    augmentation = JEPAAugmentation(image_size=32)
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=augmentation,
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

    # Optimizer
    param_groups = list(model.online_encoder.parameters()) + list(model.predictor.parameters())
    if model.torus_projection is not None:
        param_groups += list(model.torus_projection.parameters())
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.05),
    )

    # LR scheduler
    total_epochs = train_cfg.get("epochs", 300)
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6,
    )

    # Output
    out_cfg = config.get("output", {})
    out_dir = Path(out_cfg.get("dir", "./checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = out_cfg.get("save_every", 50)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = []
    best_loss = float("inf")
    t_start = time.time()

    for epoch in range(1, total_epochs + 1):
        # Warmup
        if epoch <= warmup_epochs:
            warmup_lr = train_cfg.get("lr", 1e-3) * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
        else:
            scheduler.step()

        # EMA decay schedule (linearly increase from 0.996 to 1.0)
        model.ema_decay = 1.0 - (1.0 - model_cfg.get("ema_decay", 0.996)) * (
            math.cos(math.pi * epoch / total_epochs) + 1
        ) / 2

        epoch_losses = train_one_epoch(
            model, criterion, optimizer, dataloader, device, epoch, total_epochs,
        )
        history.append({"epoch": epoch, **epoch_losses})

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch}/{total_epochs} | "
            f"loss={epoch_losses['total']:.4f} | "
            f"lr={current_lr:.6f} | "
            f"ema={model.ema_decay:.4f} | "
            f"time={elapsed:.0f}s"
        )

        # Save checkpoint
        if epoch % save_every == 0 or epoch == total_epochs:
            ckpt_path = out_dir / f"checkpoint_{epoch:04d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "history": history,
            }, ckpt_path)
            print(f"Saved: {ckpt_path}")

        if epoch_losses["total"] < best_loss:
            best_loss = epoch_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, out_dir / "best.pt")

    # Save training history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\nTraining complete. Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {out_dir}")

    return out_dir


def load_config(path: str) -> dict:
    """Load YAML or JSON config."""
    p = Path(path)
    with open(p) as f:
        if p.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("pip install pyyaml")
            return yaml.safe_load(f)
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Train EB-JEPA with toroidal regularization")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML/JSON")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--lambda-torus", type=float, default=None, help="Override λ_torus")
    parser.add_argument("--grid-size", type=int, default=None, help="Override grid size N")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)

    # Apply overrides
    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["lr"] = args.lr
    if args.lambda_torus is not None:
        config.setdefault("loss", {})["lambda_torus"] = args.lambda_torus
    if args.grid_size is not None:
        config.setdefault("loss", {})["grid_size"] = args.grid_size
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = args.seed
    if args.output_dir is not None:
        config.setdefault("output", {})["dir"] = args.output_dir

    train(config)


if __name__ == "__main__":
    main()
