"""V5: Multi-mode Fourier Torus with Karmonic Spectral Filtering.

From V1-V4b we learned:
  - Global uniformity erases class structure (V4b: 20% accuracy)
  - No uniformity causes fragmentation (V2: beta_0=845)
  - The prediction path must flow through the torus (V4 insight)

V5 resolves the accuracy-topology tradeoff via the Karmonic Mesh spectral filter
(Cormier 2026, Theorem 12.1): low-frequency Fourier modes on the torus preserve
class-level discrimination, while high-frequency modes are attenuated toward
uniformity, enforcing torus coverage without destroying class structure.

Architecture:
    encoder (512D) -> FourierTorusHead (k angles -> 2km Fourier coords)
                   -> FourierPredictor (2km -> k angles -> 2km) -> target

    Uniformity weight for mode n: w(n) = (lambda_n - lambda_1) / (lambda_max - lambda_1)
    where lambda_n = 2 - 2*cos(2*pi*n/N) is the nth eigenvalue of the cycle Laplacian.

    Mode 1 (fundamental): w=0.00 -> NO uniformity, class clustering preserved
    Mode 3+:              w>0.46 -> STRONG uniformity, torus coverage enforced

Usage:
    python -m src.train_v5 --config configs/toroidal_v5_fourier.yaml
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


# ---------------------------------------------------------------------------
# Fourier Torus Head
# ---------------------------------------------------------------------------


class FourierTorusHead(nn.Module):
    """Projects encoder output to multi-mode Fourier coordinates on T^k.

    Architecture: Linear(512->hidden)->BN->ReLU->Linear(hidden->k)->sigmoid*2pi
    Then Fourier expansion: for each angle theta_i, compute
        (cos(theta_i), sin(theta_i), cos(2*theta_i), sin(2*theta_i), ..., cos(m*theta_i), sin(m*theta_i))

    Output is grouped by mode for easy slicing in the loss:
        [mode_1_all_circles | mode_2_all_circles | ... | mode_m_all_circles]
    where each mode_n block has 2k dims: (cos(n*theta_1), sin(n*theta_1), ..., cos(n*theta_k), sin(n*theta_k))

    Total output dimension: 2 * k * m.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        torus_dim: int = 2,
        n_modes: int = 6,
    ):
        super().__init__()
        self.torus_dim = torus_dim  # k circles
        self.n_modes = n_modes      # m Fourier modes per circle
        self.embed_dim = 2 * torus_dim * n_modes  # 2km total dims

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, torus_dim),
        )

        # Mode indices: [1, 2, ..., m]
        self.register_buffer("modes", torch.arange(1, n_modes + 1).float())

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map encoder output to Fourier torus coordinates.

        Args:
            z: Encoder output (B, input_dim).

        Returns:
            angles: Raw angles in [0, 2pi) of shape (B, k).
            fourier_embed: Fourier coordinates of shape (B, 2km), grouped by mode.
        """
        raw = self.net(z)  # (B, k)
        angles = 2.0 * math.pi * torch.sigmoid(raw)  # [0, 2pi)

        fourier_embed = self._fourier_expand(angles)
        return angles, fourier_embed

    def _fourier_expand(self, angles: torch.Tensor) -> torch.Tensor:
        """Expand k angles into 2km Fourier coordinates.

        Args:
            angles: (B, k) angles in [0, 2pi).

        Returns:
            (B, 2km) Fourier coordinates grouped by mode.
        """
        # angles: (B, k) -> (B, k, 1) * modes (m,) -> (B, k, m)
        n_angles = angles.unsqueeze(-1) * self.modes  # (B, k, m)

        cos_vals = torch.cos(n_angles)  # (B, k, m)
        sin_vals = torch.sin(n_angles)  # (B, k, m)

        # Group by mode: permute to (B, m, k), then interleave cos/sin
        cos_by_mode = cos_vals.permute(0, 2, 1)  # (B, m, k)
        sin_by_mode = sin_vals.permute(0, 2, 1)  # (B, m, k)

        # Stack cos/sin pairs: (B, m, k, 2) -> reshape to (B, 2km)
        fourier = torch.stack([cos_by_mode, sin_by_mode], dim=-1)  # (B, m, k, 2)
        return fourier.reshape(angles.shape[0], -1)  # (B, 2km)


# ---------------------------------------------------------------------------
# Fourier Predictor
# ---------------------------------------------------------------------------


class FourierPredictor(nn.Module):
    """Predicts target angles from online Fourier coordinates.

    Input: 2km Fourier embedding (rich multi-scale information).
    Output: 2km predicted Fourier embedding (via angle prediction + expansion).

    The output is guaranteed to lie on T^k by construction (cos/sin of angles).
    """

    def __init__(
        self,
        fourier_dim: int,
        torus_dim: int = 2,
        n_modes: int = 6,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.torus_dim = torus_dim
        self.n_modes = n_modes

        self.net = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, torus_dim),  # predict k angles
        )

        self.register_buffer("modes", torch.arange(1, n_modes + 1).float())

    def forward(self, fourier_embed: torch.Tensor) -> torch.Tensor:
        """Predict target Fourier coordinates.

        Args:
            fourier_embed: (B, 2km) online Fourier coordinates.

        Returns:
            (B, 2km) predicted target Fourier coordinates.
        """
        raw = self.net(fourier_embed)  # (B, k)
        angles = 2.0 * math.pi * torch.sigmoid(raw)

        # Fourier expansion (same as head)
        n_angles = angles.unsqueeze(-1) * self.modes  # (B, k, m)
        cos_vals = torch.cos(n_angles).permute(0, 2, 1)  # (B, m, k)
        sin_vals = torch.sin(n_angles).permute(0, 2, 1)  # (B, m, k)
        fourier = torch.stack([cos_vals, sin_vals], dim=-1)  # (B, m, k, 2)
        return fourier.reshape(fourier_embed.shape[0], -1)  # (B, 2km)


# ---------------------------------------------------------------------------
# Karmonic Spectral Filter Loss
# ---------------------------------------------------------------------------


class KarmonicFilterLoss(nn.Module):
    """Mode-weighted uniformity implementing the Karmonic spectral filter.

    From Karmonic Mesh (Cormier 2026), Theorem 12.1:
        The Karmonic Constraint acts as a low-pass filter on the torus manifold.
        Low-frequency modes (global coherence / class discrimination) are preserved;
        high-frequency modes (noise / intra-class variation) are attenuated.

    Uniformity weight for Fourier mode n:
        w(n) = (lambda_n - lambda_1) / (lambda_max - lambda_1)

    where lambda_n = 2 - 2*cos(2*pi*n/N) is the nth eigenvalue of the
    N-cycle graph Laplacian. This gives:
        - Mode 1: w = 0.000 (no uniformity -> preserve class positions)
        - Mode 2: w ~ 0.196 (mild uniformity)
        - Mode 3: w ~ 0.464 (moderate)
        - Mode 4+: w > 0.73 (strong -> enforce torus coverage)

    Also includes circular decorrelation (spread) across circles,
    same as V4b.

    Args:
        torus_dim: Number of circles k.
        n_modes: Number of Fourier modes m per circle.
        grid_size: Torus grid size N for eigenvalue computation.
        t_uniformity: Temperature for Wang-Isola uniformity.
        spread_weight: Weight for circular decorrelation loss.
    """

    def __init__(
        self,
        torus_dim: int = 2,
        n_modes: int = 6,
        grid_size: int = 12,
        t_uniformity: float = 2.0,
        spread_weight: float = 1.0,
    ):
        super().__init__()
        self.torus_dim = torus_dim
        self.n_modes = n_modes
        self.t = t_uniformity
        self.spread_weight = spread_weight

        # Compute Karmonic spectral weights from cycle Laplacian eigenvalues
        eigenvalues = []
        for n in range(1, n_modes + 1):
            lam = 2.0 - 2.0 * math.cos(2.0 * math.pi * n / grid_size)
            eigenvalues.append(lam)

        lam_1 = eigenvalues[0]  # smallest (mode 1)
        lam_max = max(eigenvalues)

        weights = []
        for lam in eigenvalues:
            w = (lam - lam_1) / (lam_max - lam_1) if lam_max > lam_1 else 0.0
            weights.append(w)

        self.register_buffer("karmonic_weights", torch.tensor(weights))

    def forward(
        self, angles: torch.Tensor, fourier_embed: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute Karmonic-filtered uniformity + spread.

        Args:
            angles: (B, k) raw angles.
            fourier_embed: (B, 2km) Fourier coordinates grouped by mode.

        Returns:
            Dict with 'uniformity', 'spread', 'total', and per-mode uniformities.
        """
        B = fourier_embed.shape[0]
        k = self.torus_dim
        m = self.n_modes

        # Mode-weighted uniformity
        total_uniformity = torch.tensor(0.0, device=fourier_embed.device)
        mode_uniformities = []

        for n in range(m):
            # Extract mode n slice: (B, 2k)
            start = 2 * k * n
            end = 2 * k * (n + 1)
            mode_slice = fourier_embed[:, start:end]

            # Wang-Isola uniformity on this 2k-dim slice
            sq_dists = torch.cdist(mode_slice, mode_slice, p=2).pow(2)
            mask = ~torch.eye(B, dtype=torch.bool, device=fourier_embed.device)
            neg_dists = -self.t * sq_dists
            neg_dists = neg_dists.masked_select(mask).view(B, B - 1)
            unif_n = torch.logsumexp(neg_dists, dim=1).mean() - math.log(B - 1)

            weighted_unif = self.karmonic_weights[n] * unif_n
            total_uniformity = total_uniformity + weighted_unif
            mode_uniformities.append(unif_n.item())

        # Spread: circular decorrelation across circles
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

        total = total_uniformity + self.spread_weight * spread
        return {
            "uniformity": total_uniformity,
            "spread": spread,
            "total": total,
            "mode_uniformities": mode_uniformities,
        }


# ---------------------------------------------------------------------------
# V5 Model
# ---------------------------------------------------------------------------


class EBJEPA_V5(nn.Module):
    """V5: Multi-mode Fourier torus with Karmonic spectral filtering.

    Architecture:
        Online:  encoder -> FourierTorusHead -> FourierPredictor -> predicted (2km)
        Target:  EMA encoder -> EMA FourierTorusHead -> target (2km)

    The multi-mode Fourier expansion provides:
    1. Richer representation (24D vs 4D for k=2,m=6) for class discrimination
    2. Natural multi-scale structure for mode-weighted uniformity
    3. Guaranteed torus topology (all outputs parameterized by k angles)
    """

    def __init__(
        self,
        embed_dim: int = 512,
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

        # Fourier torus projection
        self.torus_projection = FourierTorusHead(
            input_dim=embed_dim,
            hidden_dim=torus_hidden,
            torus_dim=torus_dim,
            n_modes=n_modes,
        )
        self.target_torus_projection = copy.deepcopy(self.torus_projection)

        # Predictor in Fourier space
        self.fourier_predictor = FourierPredictor(
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
        """Forward pass through Fourier torus bottleneck.

        Returns:
            predicted_fourier: (B, 2km) predicted target Fourier coords
            target_fourier: (B, 2km) actual target Fourier coords
            encoder_output: (B, 512) online encoder output (for std loss)
            online_angles: (B, k) online torus angles
            online_fourier: (B, 2km) online Fourier embedding
        """
        # Online path
        encoder_output = self.online_encoder(x1)
        online_angles, online_fourier = self.torus_projection(encoder_output)
        predicted_fourier = self.fourier_predictor(online_fourier)

        # Target path
        with torch.no_grad():
            target_enc = self.target_encoder(x2)
            _, target_fourier = self.target_torus_projection(target_enc)

        return (
            predicted_fourier,
            target_fourier,
            encoder_output,
            online_angles,
            online_fourier,
        )


# ---------------------------------------------------------------------------
# V5 Loss
# ---------------------------------------------------------------------------


class KarmonicBottleneckLoss(nn.Module):
    """Loss for V5: prediction in Fourier space + Karmonic-filtered uniformity.

    L = L_pred + lambda_std * L_std + lambda_karmonic * L_karmonic

    where:
    - L_pred: MSE between predicted and target Fourier coords (all modes)
    - L_std: VICReg std loss on 512D encoder (prevents collapse)
    - L_karmonic: mode-weighted uniformity + spread (Karmonic spectral filter)
    """

    def __init__(
        self,
        torus_dim: int = 2,
        n_modes: int = 6,
        grid_size: int = 12,
        lambda_std: float = 5.0,
        lambda_karmonic: float = 15.0,
        t_uniformity: float = 2.0,
        spread_weight: float = 1.0,
    ):
        super().__init__()
        self.lambda_std = lambda_std
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
        predicted_fourier: torch.Tensor,
        target_fourier: torch.Tensor,
        encoder_output: torch.Tensor,
        online_angles: torch.Tensor,
        online_fourier: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Prediction loss: MSE in full Fourier space
        pred_loss = F.mse_loss(predicted_fourier, target_fourier.detach())

        # Std loss on 512D encoder
        std_loss = self.std_loss(encoder_output)

        # Karmonic-filtered uniformity + spread
        karmonic_losses = self.karmonic_loss(online_angles, online_fourier)

        total = (
            pred_loss
            + self.lambda_std * std_loss
            + self.lambda_karmonic * karmonic_losses["total"]
        )

        return {
            "total": total,
            "prediction": pred_loss,
            "std": std_loss,
            "uniformity": karmonic_losses["uniformity"],
            "spread": karmonic_losses["spread"],
            "mode_uniformities": karmonic_losses["mode_uniformities"],
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_v5(config: dict) -> Path:
    """V5 training: Fourier torus with Karmonic spectral filtering."""
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

    model = EBJEPA_V5(
        embed_dim=embed_dim,
        ema_decay=model_cfg.get("ema_decay", 0.996),
        torus_dim=torus_dim,
        n_modes=n_modes,
        torus_hidden=model_cfg.get("torus_hidden", 128),
        predictor_hidden=model_cfg.get("predictor_hidden", 256),
    ).to(device)

    # Optionally load pretrained encoder
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
    print(f"Torus: T^{torus_dim} with {n_modes} Fourier modes -> {fourier_dim}D embedding")

    # Print Karmonic weights
    loss_cfg = config.get("loss", {})
    grid_size = loss_cfg.get("grid_size", 12)
    print(f"\nKarmonic spectral weights (N={grid_size}):")
    for n in range(1, n_modes + 1):
        lam = 2.0 - 2.0 * math.cos(2.0 * math.pi * n / grid_size)
        lam_1 = 2.0 - 2.0 * math.cos(2.0 * math.pi / grid_size)
        lam_max = max(
            2.0 - 2.0 * math.cos(2.0 * math.pi * nn_i / grid_size)
            for nn_i in range(1, n_modes + 1)
        )
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
    criterion = KarmonicBottleneckLoss(
        torus_dim=torus_dim,
        n_modes=n_modes,
        grid_size=grid_size,
        lambda_std=loss_cfg.get("lambda_std", 5.0),
        lambda_karmonic=loss_cfg.get("lambda_karmonic", 15.0),
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
    out_dir = Path(out_cfg.get("dir", "./checkpoints/toroidal_v5_fourier"))
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = []
    best_loss = float("inf")
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"V5: Fourier Torus + Karmonic Filter ({total_epochs} epochs)")
    print(f"  T^{torus_dim} x {n_modes} modes = {fourier_dim}D embedding")
    print(f"  lambda_std={loss_cfg.get('lambda_std', 5.0)}, "
          f"lambda_karmonic={loss_cfg.get('lambda_karmonic', 15.0)}")
    print(f"  Low modes: class discrimination (w~0)")
    print(f"  High modes: torus coverage (w~1)")
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
                    f"loss={losses['total'].item():.4f} "
                    f"pred={losses['prediction'].item():.4f} "
                    f"unif={losses['uniformity'].item():.4f} "
                    f"modes=[{mu_str}...]"
                )

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
    print(f"V5 Fourier Torus training complete")
    print(f"  {total_epochs} epochs, {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Checkpoints: {out_dir}")
    print(f"{'='*60}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="V5: Multi-mode Fourier Torus + Karmonic Filter"
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
    train_v5(config)


if __name__ == "__main__":
    main()
