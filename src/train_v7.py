"""V7: Adaptive Karmonic Constraint (ERLHS Coherence Thermostat).

From ERLHS (Cormier 2026): H(z_{t+1}) <= H(z_t) + epsilon
The torus penalty should only activate when coherence degrades.

V6 applied Karmonic uniformity unconditionally every batch. This wastes
encoder capacity on maintaining torus coverage even when it's already fine.

V7 implements the coherence thermostat:
  Phase 1 (warmup, epochs 1-N): Full Karmonic penalty to ESTABLISH torus coverage
  Phase 2 (adaptive, epochs N+1-300): Only apply Karmonic when uniformity
    degrades past the level established during warmup

    gate = clamp(relu(uniformity - threshold) / gate_width, 0, 1)
    effective_karmonic = gate * karmonic_loss

    When torus is healthy (uniformity < threshold): gate=0, encoder free for accuracy
    When torus degrades (uniformity > threshold): gate>0, corrective penalty

Architecture: same as V6 (dual-path 512D + Fourier torus).
Only the loss application changes.

Usage:
    python -m src.train_v7 --config configs/toroidal_v7_adaptive.yaml
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
from .train_v6 import EBJEPA_V6
from .toroidal_loss import StandardDeviationLoss


# ---------------------------------------------------------------------------
# Adaptive Loss (Coherence Thermostat)
# ---------------------------------------------------------------------------


class AdaptiveDualPathLoss(nn.Module):
    """V7 Loss: dual-path with adaptive Karmonic gating.

    The coherence thermostat from ERLHS:
    - During warmup: full Karmonic penalty (establish torus)
    - After warmup: only apply penalty when uniformity degrades

    gate = clamp(relu(uniformity - threshold) / gate_width, 0, 1)

    When gate=0: torus is healthy, encoder focuses purely on 512D accuracy
    When gate=1: torus degrading, full corrective Karmonic penalty

    The threshold is auto-set from the EMA of uniformity at end of warmup.
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
        warmup_karmonic_epochs: int = 30,
        gate_width: float = 0.5,
    ):
        super().__init__()
        self.lambda_std = lambda_std
        self.lambda_torus_pred = lambda_torus_pred
        self.lambda_karmonic = lambda_karmonic
        self.warmup_karmonic_epochs = warmup_karmonic_epochs
        self.gate_width = gate_width

        # Coherence thermostat state
        self.coherence_threshold = None  # auto-set after warmup
        self.ema_uniformity = None
        self.ema_decay = 0.99

        self.std_loss = StandardDeviationLoss()
        self.karmonic_loss = KarmonicFilterLoss(
            torus_dim=torus_dim,
            n_modes=n_modes,
            grid_size=grid_size,
            t_uniformity=t_uniformity,
            spread_weight=spread_weight,
        )

    def update_ema(self, uniformity_value: float):
        """Update exponential moving average of uniformity."""
        if self.ema_uniformity is None:
            self.ema_uniformity = uniformity_value
        else:
            self.ema_uniformity = (
                self.ema_decay * self.ema_uniformity
                + (1 - self.ema_decay) * uniformity_value
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
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        # Path 1: 512D prediction (always active)
        pred_512_loss = F.mse_loss(predicted_512, target_512.detach())

        # Path 2: torus angle prediction (always active — creates class structure)
        pred_torus_loss = (
            1.0 - torch.cos(predicted_angles - target_angles.detach())
        ).mean()

        # Std loss (always active)
        std_loss = self.std_loss(encoder_output)

        # Karmonic loss computation (always computed for monitoring)
        karmonic_losses = self.karmonic_loss(online_angles, online_fourier)
        raw_uniformity = karmonic_losses["uniformity"]

        # Update EMA
        self.update_ema(raw_uniformity.item())

        # --- Coherence Thermostat ---
        if epoch <= self.warmup_karmonic_epochs:
            # Warmup: full Karmonic to establish torus coverage
            effective_karmonic = karmonic_losses["total"]
            gate_value = 1.0
        else:
            # Auto-set threshold at end of warmup
            if self.coherence_threshold is None:
                self.coherence_threshold = self.ema_uniformity
                # Will be printed by training loop

            # Adaptive gating: only penalize when uniformity degrades
            # uniformity is negative; more negative = better
            # deficit > 0 means uniformity is worse than threshold
            deficit = F.relu(raw_uniformity - self.coherence_threshold)
            gate = torch.clamp(deficit / self.gate_width, 0.0, 1.0)
            effective_karmonic = gate * karmonic_losses["total"]
            gate_value = gate.item()

        total = (
            pred_512_loss
            + self.lambda_torus_pred * pred_torus_loss
            + self.lambda_std * std_loss
            + self.lambda_karmonic * effective_karmonic
        )

        return {
            "total": total,
            "pred_512": pred_512_loss,
            "pred_torus": pred_torus_loss,
            "std": std_loss,
            "uniformity": karmonic_losses["uniformity"],
            "spread": karmonic_losses["spread"],
            "gate": gate_value,
            "mode_uniformities": karmonic_losses["mode_uniformities"],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_v7(config: dict) -> Path:
    """V7 training: dual-path with adaptive Karmonic thermostat."""
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

    # Reuse V6 model architecture
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

    # Adaptive loss
    warmup_karmonic = loss_cfg.get("warmup_karmonic_epochs", 30)
    gate_width = loss_cfg.get("gate_width", 0.5)
    criterion = AdaptiveDualPathLoss(
        torus_dim=torus_dim,
        n_modes=n_modes,
        grid_size=grid_size,
        lambda_std=loss_cfg.get("lambda_std", 25.0),
        lambda_torus_pred=loss_cfg.get("lambda_torus_pred", 0.5),
        lambda_karmonic=loss_cfg.get("lambda_karmonic", 5.0),
        t_uniformity=loss_cfg.get("t_uniformity", 2.0),
        spread_weight=loss_cfg.get("spread_weight", 1.0),
        warmup_karmonic_epochs=warmup_karmonic,
        gate_width=gate_width,
    )

    print(f"\nLoss weights:")
    print(f"  pred_512: 1.0 (implicit)")
    print(f"  pred_torus: {loss_cfg.get('lambda_torus_pred', 0.5)}")
    print(f"  std: {loss_cfg.get('lambda_std', 25.0)}")
    print(f"  karmonic: {loss_cfg.get('lambda_karmonic', 5.0)} (gated)")
    print(f"\nCoherence thermostat (ERLHS):")
    print(f"  Warmup: epochs 1-{warmup_karmonic} (full Karmonic)")
    print(f"  Adaptive: epochs {warmup_karmonic+1}-{train_cfg.get('epochs', 300)} (gated)")
    print(f"  Gate width: {gate_width}")
    print(f"  Threshold: auto-set from warmup EMA")

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
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v7_adaptive"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_loss = float("inf")
    t_start = time.time()
    threshold_printed = False

    print(f"\n{'='*60}")
    print(f"V7: Adaptive Karmonic (ERLHS Thermostat) ({total_epochs} epochs)")
    print(f"  Path 1: 512D prediction (always active)")
    print(f"  Path 2: T^{torus_dim} x {n_modes} Fourier (always active)")
    print(f"  Karmonic: full during warmup, gated after")
    print(f"{'='*60}\n")

    for epoch in range(1, total_epochs + 1):
        model.train()

        # LR warmup
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
        gate_accum = 0.0
        n_batches = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            x1, x2 = images
            x1, x2 = x1.to(device), x2.to(device)

            out = model(x1, x2)
            losses = criterion(*out, epoch=epoch)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.update_target()

            for k, v in losses.items():
                if k in ("mode_uniformities",):
                    continue
                if k == "gate":
                    gate_accum += v
                else:
                    loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                phase = "WARMUP" if epoch <= warmup_karmonic else "ADAPTIVE"
                print(
                    f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                    f"p512={losses['pred_512'].item():.4f} "
                    f"ptor={losses['pred_torus'].item():.4f} "
                    f"gate={losses['gate']:.3f} [{phase}]"
                )

        avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
        avg_gate = gate_accum / max(n_batches, 1)
        history.append({"epoch": epoch, "avg_gate": avg_gate, **avg_losses})

        # Print threshold when it's first set
        if (
            epoch == warmup_karmonic + 1
            and criterion.coherence_threshold is not None
            and not threshold_printed
        ):
            print(f"\n  >>> COHERENCE THRESHOLD SET: {criterion.coherence_threshold:.4f}")
            print(f"  >>> Switching to adaptive mode (gate activates above this)\n")
            threshold_printed = True

        elapsed = time.time() - t_start
        phase = "WARMUP" if epoch <= warmup_karmonic else "ADAPT"
        print(
            f"Epoch {epoch}/{total_epochs} [{phase}] | "
            f"total={avg_losses['total']:.4f} | "
            f"p512={avg_losses['pred_512']:.4f} | "
            f"ptor={avg_losses['pred_torus']:.4f} | "
            f"gate={avg_gate:.3f} | "
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
    print(f"V7 Adaptive Karmonic training complete")
    print(f"  {total_epochs} epochs, {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Coherence threshold: {criterion.coherence_threshold:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="V7: Adaptive Karmonic (ERLHS Coherence Thermostat)"
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
    train_v7(config)


if __name__ == "__main__":
    main()
