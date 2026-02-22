"""V8a: Detached Karmonic — encoder free, torus head independent.

Key insight from V5-V7: karmonic gradients flowing into the encoder degrade
its representations. The accuracy-topology tension is caused by the encoder
fighting two objectives through the same parameters.

V8a surgically removes this tension:
  - Encoder gets gradients from: p512 prediction + std loss ONLY
  - Torus head gets gradients from: karmonic + torus prediction ONLY
  - The torus head operates on encoder_output.detach()

The encoder is completely free to maximize 512D accuracy (~70%).
The torus head independently learns to arrange encoder features on a torus.
If the encoder's features have natural structure that maps to a torus,
the torus accuracy will be high AND the encoder stays near baseline.

Architecture (same as V6, one line changed):
    Online:  encoder (512D) ---> predictor_512 ---------> predicted_512
                            \\--> [DETACH] -> FourierTorusHead -> FourierPredictor -> predicted_angles
    Target:  EMA encoder ----> target_512
                           \\--> [DETACH] -> EMA FourierTorusHead --> target_angles

Usage:
    python -m src.train_v8a --config configs/toroidal_v8a_detached.yaml
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
# V8a Model — V6 with detached torus branch
# ---------------------------------------------------------------------------


class EBJEPA_V8a(nn.Module):
    """V8a: Dual-path with DETACHED torus branch.

    Same architecture as V6, but the torus head operates on
    encoder_output.detach(). This means:
    - The encoder ONLY gets gradients from p512 + std (accuracy)
    - The torus head ONLY gets gradients from karmonic + torus_pred (topology)
    - No gradient conflict between accuracy and topology
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

        # Path 2: Fourier torus branch (gets DETACHED encoder output)
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
        """Forward pass — torus branch gets DETACHED encoder output.

        Returns same tuple as V6 for eval compatibility.
        """
        # Online path
        encoder_output = self.online_encoder(x1)

        # Path 1: 512D prediction (encoder gets gradients)
        predicted_512 = self.predictor(encoder_output)

        # Path 2: Fourier torus — DETACHED from encoder
        # Torus head trains independently, encoder doesn't see karmonic gradients
        online_angles, online_fourier = self.torus_projection(encoder_output.detach())
        predicted_angles, _ = self.torus_predictor(online_fourier)

        # Target path (also detached for torus)
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
# Training (reuses V6 training loop with V8a model)
# ---------------------------------------------------------------------------


def train_v8a(config: dict) -> Path:
    """V8a training: dual-path with detached karmonic."""
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

    model = EBJEPA_V8a(
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
    print(f"Path 1: encoder (512D) -> predictor (512D) [FULL GRADIENTS]")
    print(f"Path 2: encoder (512D) -> [DETACH] -> T^{torus_dim} x {n_modes} modes = {fourier_dim}D")

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

    # Loss — same as V6, but karmonic only trains torus head (via detach)
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
    print(f"  pred_512: 1.0 (encoder gets these gradients)")
    print(f"  pred_torus: {loss_cfg.get('lambda_torus_pred', 0.5)} (torus head only)")
    print(f"  std: {loss_cfg.get('lambda_std', 25.0)} (encoder gets these gradients)")
    print(f"  karmonic: {loss_cfg.get('lambda_karmonic', 5.0)} (torus head only — DETACHED)")

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
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v8a_detached"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_loss = float("inf")
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"V8a: Detached Karmonic ({total_epochs} epochs)")
    print(f"  Encoder: FREE (p512 + std only)")
    print(f"  Torus head: INDEPENDENT (karmonic + torus_pred)")
    print(f"  Gradient bridge: NONE (encoder_output.detach())")
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
    print(f"V8a Detached Karmonic training complete")
    print(f"  {total_epochs} epochs, {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="V8a: Detached Karmonic (encoder free, torus independent)"
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
    train_v8a(config)


if __name__ == "__main__":
    main()
