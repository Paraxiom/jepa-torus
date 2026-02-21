"""V4 Toroidal Bottleneck Training for JEPA.

Key insight from V1-V3: prediction path must flow THROUGH the torus.
In V2/V3, encoder->predictor bypasses the torus head, so the encoder
has no incentive to maintain torus structure.

V4 architecture:
  x1 -> encoder (512D) -> torus_head (4D on S1xS1) -> torus_predictor -> predicted (4D)
  x2 -> target_encoder (EMA) -> torus_head (4D on S1xS1) -> target (4D)

  Loss = MSE(predicted, target) on S1xS1
       + lambda_torus * uniformity on torus embeddings
       + lambda_std * std on 512D encoder (prevents collapse for linear probe)

Usage:
    python -m src.train_v4 --config configs/toroidal_v4_N12.yaml
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
from .toroidal_loss import TorusProjectionHead, ToroidalStructureLoss, StandardDeviationLoss


class TorusPredictor(nn.Module):
    """Predicts target torus coordinates from online torus coordinates.

    Input: 4D (cos θ1, sin θ1, cos θ2, sin θ2) on S1×S1
    Output: 4D predicted target on S1×S1 (via angle prediction + cos/sin)

    The output is guaranteed to lie on S1×S1 by construction.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # predict 2 raw angles
        )

    def forward(self, torus_embed: torch.Tensor) -> torch.Tensor:
        """Predict target torus embedding from online torus embedding.

        Args:
            torus_embed: (B, 4) online torus coordinates on S1×S1.

        Returns:
            predicted: (B, 4) predicted target coordinates on S1×S1.
        """
        raw = self.net(torus_embed)  # (B, 2)
        angles = 2.0 * math.pi * torch.sigmoid(raw)  # [0, 2π)
        cos1 = torch.cos(angles[:, 0])
        sin1 = torch.sin(angles[:, 0])
        cos2 = torch.cos(angles[:, 1])
        sin2 = torch.sin(angles[:, 1])
        return torch.stack([cos1, sin1, cos2, sin2], dim=1)  # (B, 4)


class EBJEPA_V4(nn.Module):
    """V4: Prediction flows through the torus bottleneck.

    Architecture:
        Online:  encoder -> torus_head -> torus_predictor -> predicted_torus
        Target:  target_encoder (EMA) -> target_torus_head (EMA) -> target_torus

    The encoder must produce torus-compatible representations because
    ALL prediction signal flows through the 4D torus bottleneck.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        ema_decay: float = 0.996,
        torus_predictor_hidden: int = 64,
    ):
        super().__init__()
        self.online_encoder = ResNetEncoder(embed_dim)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.ema_decay = ema_decay

        # Torus projection: encoder (512D) -> S1×S1 (4D)
        self.torus_projection = TorusProjectionHead(input_dim=embed_dim)
        self.target_torus_projection = copy.deepcopy(self.torus_projection)

        # Predictor operates in torus space (4D -> 4D)
        self.torus_predictor = TorusPredictor(hidden_dim=torus_predictor_hidden)

        # Freeze target networks
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_torus_projection.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        """EMA update of target encoder AND target torus head."""
        for online, target in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)
        for online, target in zip(
            self.torus_projection.parameters(), self.target_torus_projection.parameters()
        ):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1 - self.ema_decay)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        """Forward pass: both views go through torus bottleneck.

        Returns:
            predicted_torus: (B, 4) predicted target torus coords
            target_torus: (B, 4) actual target torus coords (detached)
            encoder_output: (B, 512) online encoder output (for std loss)
            online_angles: (B, 2) online torus angles
            online_torus: (B, 4) online torus embedding
        """
        # Online path: encoder -> torus -> predictor
        encoder_output = self.online_encoder(x1)
        online_angles, online_torus = self.torus_projection(encoder_output)
        predicted_torus = self.torus_predictor(online_torus)

        # Target path: target_encoder -> target_torus
        with torch.no_grad():
            target_enc = self.target_encoder(x2)
            _, target_torus = self.target_torus_projection(target_enc)

        return predicted_torus, target_torus, encoder_output, online_angles, online_torus


class ToroidalBottleneckLoss(nn.Module):
    """Loss for V4: prediction in torus space + uniformity + std.

    L = L_pred_torus + lambda_torus * L_uniformity + lambda_std * L_std

    where:
    - L_pred_torus: MSE between predicted and target torus embeddings (chord distance)
    - L_uniformity: Wang-Isola uniformity on torus embeddings
    - L_std: VICReg std loss on 512D encoder output (prevents collapse)
    """

    def __init__(
        self,
        lambda_std: float = 5.0,
        lambda_torus: float = 25.0,
        t_uniformity: float = 2.0,
        spread_weight: float = 1.0,
    ):
        super().__init__()
        self.lambda_std = lambda_std
        self.lambda_torus = lambda_torus

        self.std_loss = StandardDeviationLoss()
        self.torus_loss = ToroidalStructureLoss(
            t_uniformity=t_uniformity,
            spread_weight=spread_weight,
        )

    def forward(
        self,
        predicted_torus: torch.Tensor,
        target_torus: torch.Tensor,
        encoder_output: torch.Tensor,
        online_angles: torch.Tensor,
        online_torus: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Prediction loss: MSE on torus chord distance
        pred_loss = F.mse_loss(predicted_torus, target_torus.detach())

        # Std loss on 512D encoder (keep representations non-degenerate)
        std_loss = self.std_loss(encoder_output)

        # Torus uniformity + spread on online torus embeddings
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


def train_v4(config: dict) -> Path:
    """V4 training: prediction through torus bottleneck."""
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

    # Build V4 model
    model = EBJEPA_V4(
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        ema_decay=model_cfg.get("ema_decay", 0.996),
        torus_predictor_hidden=model_cfg.get("torus_predictor_hidden", 64),
    ).to(device)

    # Optionally load pretrained encoder
    pretrained_path = config.get("pretrained_checkpoint")
    if pretrained_path:
        print(f"\n{'='*60}")
        print(f"Loading pretrained encoder: {pretrained_path}")
        print(f"{'='*60}")
        ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        pretrained_state = ckpt["model_state_dict"]

        # Map pretrained weights to V4 model (encoder + target_encoder)
        own_state = model.state_dict()
        loaded = 0
        for key, value in pretrained_state.items():
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key] = value
                loaded += 1
        model.load_state_dict(own_state)
        print(f"Loaded {loaded} pretrained weight tensors")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable:,}")
    print(f"Torus predictor: 4D -> {model_cfg.get('torus_predictor_hidden', 64)}D -> 4D (on S1xS1)")

    # Data
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "./data")
    augmentation = JEPAAugmentation(image_size=32)
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=augmentation,
    )
    train_cfg = config.get("training", {})
    batch_size = train_cfg.get("batch_size", 256)
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True, drop_last=True,
    )

    # Loss
    loss_cfg = config.get("loss", {})
    criterion = ToroidalBottleneckLoss(
        lambda_std=loss_cfg.get("lambda_std", 5.0),
        lambda_torus=loss_cfg.get("lambda_torus", 25.0),
        t_uniformity=loss_cfg.get("t_uniformity", 2.0),
        spread_weight=loss_cfg.get("spread_weight", 1.0),
    )

    # Optimizer — all trainable params
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

    # Output
    out_cfg = config.get("output", {})
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v4_N12"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = out_cfg.get("save_every", 50)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = []
    best_loss = float("inf")
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"V4 Toroidal Bottleneck Training ({total_epochs} epochs)")
    print(f"  Prediction: encoder -> torus (4D) -> predictor -> target torus")
    print(f"  lambda_std={loss_cfg.get('lambda_std', 5.0)}, "
          f"lambda_torus={loss_cfg.get('lambda_torus', 25.0)}")
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

            predicted_torus, target_torus, encoder_output, online_angles, online_torus = model(x1, x2)
            losses = criterion(
                predicted_torus, target_torus, encoder_output,
                online_angles, online_torus,
            )

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

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start
        print(
            f"Epoch {epoch}/{total_epochs} | "
            f"loss={avg_losses['total']:.4f} | "
            f"pred={avg_losses['prediction']:.4f} | "
            f"unif={avg_losses['uniformity']:.4f} | "
            f"lr={current_lr:.6f} | "
            f"ema={model.ema_decay:.4f} | "
            f"time={elapsed:.0f}s"
        )

        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, out_dir / "best.pt")

        if epoch % save_every == 0 or epoch == total_epochs:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "history": history,
            }, out_dir / f"checkpoint_{epoch:04d}.pt")

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"V4 training complete")
    print(f"  {total_epochs} epochs, {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="V4 Toroidal Bottleneck Training")
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

    train_v4(config)


if __name__ == "__main__":
    main()
