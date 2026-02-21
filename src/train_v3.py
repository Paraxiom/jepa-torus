"""V3 Curriculum Training for Toroidal JEPA.

Two-phase approach:
  Phase 1: Load pretrained encoder (baseline_vicreg), freeze it,
           train only the torus projection head with high uniformity weight.
  Phase 2: Unfreeze encoder, fine-tune everything with balanced losses.

Usage:
    python -m src.train_v3 --config configs/toroidal_v3_N12.yaml
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import yaml
except ImportError:
    yaml = None

from .train import EBJEPA, JEPAAugmentation
from .toroidal_loss import ToroidalJEPALossV2


def train_curriculum(config: dict) -> Path:
    """Curriculum training: pretrained encoder -> torus head -> fine-tune.

    Config keys:
        pretrained_checkpoint: path to baseline_vicreg/best.pt
        phase1.epochs, phase1.lr, phase1.lambda_torus, phase1.t_uniformity
        phase2.epochs, phase2.lr, phase2.lambda_torus, phase2.lambda_std
        + standard model/data/output keys
    """
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

    # --- Load pretrained encoder ---
    pretrained_path = config.get("pretrained_checkpoint", "checkpoints/baseline_vicreg/best.pt")
    print(f"\n{'='*60}")
    print(f"Loading pretrained encoder: {pretrained_path}")
    print(f"{'='*60}")

    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)

    # Build model WITH torus head
    model = EBJEPA(
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        ema_decay=model_cfg.get("ema_decay", 0.996),
        torus_head=True,
    ).to(device)

    # Load pretrained weights (strict=False because torus_projection is new)
    pretrained_state = ckpt["model_state_dict"]
    missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
    print(f"Loaded pretrained weights. Missing (new): {len(missing)}, Unexpected: {len(unexpected)}")
    for k in missing:
        print(f"  + {k}")

    # Data
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir", "./data")
    augmentation = JEPAAugmentation(image_size=32)
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=augmentation,
    )
    batch_size = config.get("training", {}).get("batch_size", 256)
    dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=config.get("training", {}).get("num_workers", 4),
        pin_memory=True, drop_last=True,
    )

    # Output
    out_cfg = config.get("output", {})
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v3_N12"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    t_start = time.time()

    # =========================================================================
    # PHASE 1: Freeze encoder, train only torus head
    # =========================================================================
    p1 = config.get("phase1", {})
    p1_epochs = p1.get("epochs", 200)
    p1_lr = p1.get("lr", 0.003)
    p1_lambda_torus = p1.get("lambda_torus", 50.0)
    p1_lambda_std = p1.get("lambda_std", 1.0)
    p1_t_uniformity = p1.get("t_uniformity", 2.0)
    p1_spread = p1.get("spread_weight", 2.0)

    print(f"\n{'='*60}")
    print(f"PHASE 1: Train torus head only ({p1_epochs} epochs)")
    print(f"  Encoder: FROZEN")
    print(f"  lr={p1_lr}, lambda_torus={p1_lambda_torus}, t={p1_t_uniformity}")
    print(f"{'='*60}\n")

    # Freeze encoder + predictor
    for p in model.online_encoder.parameters():
        p.requires_grad = False
    for p in model.predictor.parameters():
        p.requires_grad = False

    # Only optimize torus head
    torus_params = list(model.torus_projection.parameters())
    print(f"Torus head params: {sum(p.numel() for p in torus_params):,}")

    optimizer_p1 = torch.optim.AdamW(torus_params, lr=p1_lr, weight_decay=0.01)
    scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=p1_epochs, eta_min=1e-5,
    )

    criterion_p1 = ToroidalJEPALossV2(
        embed_dim=embed_dim,
        lambda_std=p1_lambda_std,
        lambda_torus=p1_lambda_torus,
        t_uniformity=p1_t_uniformity,
        spread_weight=p1_spread,
    )

    best_loss = float("inf")

    for epoch in range(1, p1_epochs + 1):
        model.train()
        loss_accum = {}
        n_batches = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            x1, x2 = images
            x1, x2 = x1.to(device), x2.to(device)

            predictions, targets, encoder_output, torus_angles, torus_embed = model(x1, x2)
            losses = criterion_p1(
                predictions, targets, encoder_output,
                torus_angles=torus_angles, torus_embed=torus_embed,
            )

            optimizer_p1.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(torus_params, max_norm=1.0)
            optimizer_p1.step()

            # Still update EMA target (frozen encoder doesn't change, but keeps it consistent)
            model.update_target()

            for k, v in losses.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                print(f"  P1 Epoch {epoch} [{batch_idx}/{len(dataloader)}] loss={losses['total'].item():.4f}")

        scheduler_p1.step()
        avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
        history.append({"phase": 1, "epoch": epoch, **avg_losses})

        elapsed = time.time() - t_start
        print(
            f"P1 Epoch {epoch}/{p1_epochs} | "
            f"loss={avg_losses['total']:.4f} | "
            f"lr={optimizer_p1.param_groups[0]['lr']:.6f} | "
            f"time={elapsed:.0f}s"
        )

        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            torch.save({
                "epoch": epoch,
                "phase": 1,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, out_dir / "best.pt")

        if epoch % 50 == 0:
            torch.save({
                "epoch": epoch,
                "phase": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p1.state_dict(),
                "config": config,
                "history": history,
            }, out_dir / f"checkpoint_p1_{epoch:04d}.pt")

    print(f"\nPhase 1 complete. Best loss: {best_loss:.4f}")

    # =========================================================================
    # PHASE 2: Unfreeze encoder, fine-tune everything
    # =========================================================================
    p2 = config.get("phase2", {})
    p2_epochs = p2.get("epochs", 100)
    p2_lr = p2.get("lr", 0.0003)
    p2_lambda_torus = p2.get("lambda_torus", 20.0)
    p2_lambda_std = p2.get("lambda_std", 10.0)
    p2_t_uniformity = p2.get("t_uniformity", 2.0)
    p2_spread = p2.get("spread_weight", 1.0)

    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tune everything ({p2_epochs} epochs)")
    print(f"  Encoder: UNFROZEN")
    print(f"  lr={p2_lr}, lambda_std={p2_lambda_std}, lambda_torus={p2_lambda_torus}")
    print(f"{'='*60}\n")

    # Unfreeze encoder + predictor
    for p in model.online_encoder.parameters():
        p.requires_grad = True
    for p in model.predictor.parameters():
        p.requires_grad = True

    all_params = (
        list(model.online_encoder.parameters())
        + list(model.predictor.parameters())
        + list(model.torus_projection.parameters())
    )
    print(f"Total trainable params: {sum(p.numel() for p in all_params if p.requires_grad):,}")

    optimizer_p2 = torch.optim.AdamW(all_params, lr=p2_lr, weight_decay=0.05)
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=p2_epochs, eta_min=1e-6,
    )

    criterion_p2 = ToroidalJEPALossV2(
        embed_dim=embed_dim,
        lambda_std=p2_lambda_std,
        lambda_torus=p2_lambda_torus,
        t_uniformity=p2_t_uniformity,
        spread_weight=p2_spread,
    )

    for epoch in range(1, p2_epochs + 1):
        model.train()

        # EMA decay schedule for phase 2
        model.ema_decay = 1.0 - (1.0 - model_cfg.get("ema_decay", 0.996)) * (
            math.cos(math.pi * epoch / p2_epochs) + 1
        ) / 2

        loss_accum = {}
        n_batches = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            x1, x2 = images
            x1, x2 = x1.to(device), x2.to(device)

            predictions, targets, encoder_output, torus_angles, torus_embed = model(x1, x2)
            losses = criterion_p2(
                predictions, targets, encoder_output,
                torus_angles=torus_angles, torus_embed=torus_embed,
            )

            optimizer_p2.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_p2.step()

            model.update_target()

            for k, v in losses.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                print(f"  P2 Epoch {epoch} [{batch_idx}/{len(dataloader)}] loss={losses['total'].item():.4f}")

        scheduler_p2.step()
        avg_losses = {k: v / max(n_batches, 1) for k, v in loss_accum.items()}
        history.append({"phase": 2, "epoch": epoch, **avg_losses})

        elapsed = time.time() - t_start
        print(
            f"P2 Epoch {epoch}/{p2_epochs} | "
            f"loss={avg_losses['total']:.4f} | "
            f"lr={optimizer_p2.param_groups[0]['lr']:.6f} | "
            f"ema={model.ema_decay:.4f} | "
            f"time={elapsed:.0f}s"
        )

        if avg_losses["total"] < best_loss:
            best_loss = avg_losses["total"]
            torch.save({
                "epoch": p1_epochs + epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "config": config,
            }, out_dir / "best.pt")

        if epoch % 50 == 0 or epoch == p2_epochs:
            torch.save({
                "epoch": p1_epochs + epoch,
                "phase": 2,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer_p2.state_dict(),
                "config": config,
                "history": history,
            }, out_dir / f"checkpoint_p2_{epoch:04d}.pt")

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"V3 Curriculum training complete")
    print(f"  Phase 1: {p1_epochs} epochs (torus head only)")
    print(f"  Phase 2: {p2_epochs} epochs (fine-tune all)")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")

    return out_dir


def main():
    parser = argparse.ArgumentParser(description="V3 Curriculum Training for Toroidal JEPA")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML/JSON")
    args = parser.parse_args()

    p = Path(args.config)
    with open(p) as f:
        if p.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("pip install pyyaml")
            config = yaml.safe_load(f)
        else:
            config = json.load(f)

    train_curriculum(config)


if __name__ == "__main__":
    main()
