"""Toroidal covariance regularizer for JEPA latent spaces.

Replaces VICReg's isotropic covariance penalty with a geometry-aware one
based on the N×N torus Laplacian (Tonnetz geometry). Nearby dimensions on
the torus may covary; distant dimensions are penalized more heavily.

The spectral gap λ₁ = 2 - 2·cos(2π/N) provides a provable lower bound on
representation diversity — collapse prevention with geometric structure.

References:
    - Cormier (2026), "Topological Constraints for Coherent Language Models"
    - Bardes et al. (2022), VICReg
    - Gardner et al. (2022), toroidal grid cells in entorhinal cortex
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _build_torus_distance_matrix(grid_size: int) -> np.ndarray:
    """Build N²×N² Manhattan distance matrix on the N×N torus.

    d_T(i, j) = min(|x_i-x_j|, N-|x_i-x_j|) + min(|y_i-y_j|, N-|y_i-y_j|)
    """
    n = grid_size
    n2 = n * n
    rows = np.arange(n2)
    x = rows % n
    y = rows // n
    dx = np.abs(x[:, None] - x[None, :])
    dy = np.abs(y[:, None] - y[None, :])
    dx = np.minimum(dx, n - dx)
    dy = np.minimum(dy, n - dy)
    return dx + dy


def _build_torus_adjacency(grid_size: int) -> np.ndarray:
    """Build adjacency matrix for N×N torus (4-connected grid with wraparound)."""
    n = grid_size
    n2 = n * n
    adj = np.zeros((n2, n2), dtype=np.float64)
    for i in range(n2):
        x, y = i % n, i // n
        neighbors = [
            ((x + 1) % n) + y * n,
            ((x - 1) % n) + y * n,
            x + ((y + 1) % n) * n,
            x + ((y - 1) % n) * n,
        ]
        for j in neighbors:
            adj[i, j] = 1.0
    return adj


def _build_torus_laplacian(grid_size: int) -> np.ndarray:
    """Build graph Laplacian L = D - A for the N×N torus.

    Each node has degree 4 (4-connected grid). L is positive semidefinite
    with smallest eigenvalue 0 (constant vector). Spectral gap λ₁ provides
    the lower bound for our regularizer.
    """
    adj = _build_torus_adjacency(grid_size)
    degree = np.diag(adj.sum(axis=1))
    return degree - adj


def torus_spectral_gap(grid_size: int) -> float:
    """Analytic spectral gap of the N×N torus: λ₁ = 2 - 2·cos(2π/N).

    This is the smallest nonzero eigenvalue of the graph Laplacian.
    For N=12: λ₁ ≈ 0.268
    For N=8:  λ₁ ≈ 0.586
    For N=16: λ₁ ≈ 0.152
    """
    return 2.0 - 2.0 * math.cos(2.0 * math.pi / grid_size)


@lru_cache(maxsize=8)
def _get_torus_penalty_matrix(grid_size: int, device_str: str) -> torch.Tensor:
    """Cached torus penalty weight matrix W[i,j] based on Laplacian.

    W[i,j] = L[i,j]² — penalizes covariance between dimensions based
    on their torus Laplacian coupling. Adjacent dims (L=-1) get weight 1,
    non-adjacent (L=0) get weight 0, self (L=4) gets weight 16.

    We normalize so that Tr(W) = N² (scale-invariant).
    """
    L = _build_torus_laplacian(grid_size)
    # Absolute value of Laplacian entries gives natural penalty weights:
    # adjacent (L=-1) → 1, non-adjacent (L=0) → 0, diagonal (L=4) → 4
    # We use |L| so that adjacent dim covariance IS penalized (less than diagonal)
    W = np.abs(L).astype(np.float32)
    # Normalize: mean off-diagonal weight = 1
    n2 = grid_size * grid_size
    off_diag_mask = ~np.eye(n2, dtype=bool)
    off_diag_sum = W[off_diag_mask].sum()
    if off_diag_sum > 0:
        W[off_diag_mask] *= (n2 * (n2 - 1)) / off_diag_sum
    device = torch.device(device_str)
    return torch.from_numpy(W).to(device)


@lru_cache(maxsize=8)
def _get_distance_penalty_matrix(
    grid_size: int, device_str: str
) -> torch.Tensor:
    """Distance-based penalty: W[i,j] = d_T(i,j) / max_dist.

    This gives a softer version: nearby dims penalized less, far dims more.
    """
    D = _build_torus_distance_matrix(grid_size).astype(np.float32)
    max_dist = D.max()
    if max_dist > 0:
        D /= max_dist
    device = torch.device(device_str)
    return torch.from_numpy(D).to(device)


class ToroidalCovarianceLoss(nn.Module):
    """Toroidal covariance regularizer for JEPA representations.

    Given batch embeddings z ∈ R^{B×D}, computes covariance matrix C ∈ R^{D×D}
    and penalizes off-diagonal entries weighted by torus distance:

        E_torus(z) = (1 / D²) Σ_{i≠j} W[pos(i), pos(j)] · C[i,j]²

    where pos(d) = d % N² maps embedding dimension d to a torus position,
    and W is the torus distance penalty matrix.

    Args:
        grid_size: Torus grid dimension N (torus is N×N). Default 12.
        embed_dim: Embedding dimension D. Used for position mapping.
        penalty_mode: 'distance' (soft, distance-based) or 'laplacian' (hard, adjacency-based).
        normalize: Whether to normalize the loss by D².
    """

    def __init__(
        self,
        grid_size: int = 12,
        embed_dim: int = 2048,
        penalty_mode: str = "distance",
        normalize: bool = True,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.embed_dim = embed_dim
        self.penalty_mode = penalty_mode
        self.normalize = normalize
        self.n_positions = grid_size * grid_size

        # Pre-compute dimension-to-torus-position mapping
        positions = torch.arange(embed_dim) % self.n_positions
        self.register_buffer("positions", positions, persistent=False)

    def _get_weight_matrix(self, device: torch.device) -> torch.Tensor:
        """Get the N²×N² penalty weight matrix (cached)."""
        device_str = str(device)
        if self.penalty_mode == "laplacian":
            return _get_torus_penalty_matrix(self.grid_size, device_str)
        else:
            return _get_distance_penalty_matrix(self.grid_size, device_str)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute toroidal covariance loss.

        Args:
            z: Embeddings of shape (B, D) where B is batch size.

        Returns:
            Scalar loss value.
        """
        B, D = z.shape
        assert D == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {D}"

        # Center embeddings
        z_centered = z - z.mean(dim=0, keepdim=True)

        # Covariance matrix: (D, D)
        cov = (z_centered.T @ z_centered) / (B - 1)

        # Get torus penalty weights: (N², N²)
        W = self._get_weight_matrix(z.device)

        # Map embedding dims to torus positions
        pos = self.positions.to(z.device)  # (D,)

        # Build D×D weight matrix from torus positions
        W_full = W[pos[:, None], pos[None, :]]  # (D, D)

        # Zero out diagonal (we only penalize off-diagonal covariance)
        mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
        loss = (W_full * cov.pow(2) * mask.float()).sum()

        if self.normalize:
            loss = loss / (D * (D - 1))

        return loss


class StandardDeviationLoss(nn.Module):
    """VICReg-style standard deviation loss (collapse prevention).

    Ensures each embedding dimension has std >= gamma (default 1).
    This is dimension-independent and complements the toroidal covariance loss.

    Args:
        gamma: Target standard deviation. Default 1.0.
        eps: Small constant for numerical stability.
    """

    def __init__(self, gamma: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute std loss: mean(max(0, gamma - std(z_d)))."""
        std = torch.sqrt(z.var(dim=0) + self.eps)
        return F.relu(self.gamma - std).mean()


class ToroidalJEPALoss(nn.Module):
    """Complete loss function for toroidal EB-JEPA.

    Combines:
    1. Prediction loss (L2 between predictor output and target)
    2. Standard deviation loss (collapse prevention)
    3. Toroidal covariance loss (geometry-aware decorrelation)

    L = L_pred + λ_std · L_std + λ_torus · E_torus

    Args:
        grid_size: Torus grid dimension N.
        embed_dim: Embedding dimension D.
        lambda_std: Weight for standard deviation loss.
        lambda_torus: Weight for toroidal covariance loss.
        penalty_mode: 'distance' or 'laplacian'.
    """

    def __init__(
        self,
        grid_size: int = 12,
        embed_dim: int = 2048,
        lambda_std: float = 25.0,
        lambda_torus: float = 1.0,
        penalty_mode: str = "distance",
    ):
        super().__init__()
        self.lambda_std = lambda_std
        self.lambda_torus = lambda_torus

        self.std_loss = StandardDeviationLoss()
        self.torus_loss = ToroidalCovarianceLoss(
            grid_size=grid_size,
            embed_dim=embed_dim,
            penalty_mode=penalty_mode,
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            predictions: Predictor output (B, D).
            targets: EMA target encoder output (B, D). Detached.
            encoder_output: Online encoder output (B, D). For regularization.

        Returns:
            Dict with 'total', 'prediction', 'std', 'toroidal' losses.
        """
        # Prediction loss: MSE between predictor and target
        pred_loss = F.mse_loss(predictions, targets.detach())

        # Regularization on online encoder output
        std_loss = self.std_loss(encoder_output)
        torus_loss = self.torus_loss(encoder_output)

        total = pred_loss + self.lambda_std * std_loss + self.lambda_torus * torus_loss

        return {
            "total": total,
            "prediction": pred_loss,
            "std": std_loss,
            "toroidal": torus_loss,
        }


class TorusProjectionHead(nn.Module):
    """Projects 512D encoder output onto T² = S¹×S¹ via cos/sin.

    Architecture: Linear(512→128) → BN → ReLU → Linear(128→2) → scale by 2π
    Output: angles (B, 2) and torus_embed (B, 4) = (cos θ₁, sin θ₁, cos θ₂, sin θ₂)

    The 4D output lives exactly on S¹×S¹ by construction — no regularization
    needed to enforce topology; it's a hard architectural constraint.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map encoder output to torus.

        Args:
            z: Encoder output (B, 512).

        Returns:
            angles: Raw angles in [0, 2π) of shape (B, 2).
            torus_embed: (cos θ₁, sin θ₁, cos θ₂, sin θ₂) of shape (B, 4).
        """
        raw = self.net(z)  # (B, 2)
        angles = 2.0 * math.pi * torch.sigmoid(raw)  # Map to [0, 2π)
        cos1 = torch.cos(angles[:, 0])
        sin1 = torch.sin(angles[:, 0])
        cos2 = torch.cos(angles[:, 1])
        sin2 = torch.sin(angles[:, 1])
        torus_embed = torch.stack([cos1, sin1, cos2, sin2], dim=1)  # (B, 4)
        return angles, torus_embed


class ToroidalStructureLoss(nn.Module):
    """Topology-aware loss for points on T² embedded in R⁴.

    Two components:
    1. **Uniformity** (Wang & Isola, 2020): log E[exp(-t·||z_i - z_j||²)]
       Pushes points to spread uniformly across T². Euclidean distance
       in R⁴ between points on S¹×S¹ is the correct torus chord metric.

    2. **Spread** (circular correlation penalty): Penalizes correlation
       between θ₁ and θ₂ to prevent degenerate collapse to a 1D circle.
       Uses Fisher & Lee (1983) circular correlation coefficient:
       ρ_circ = E[sin(θ₁-μ₁)·sin(θ₂-μ₂)] / √(E[sin²(θ₁-μ₁)]·E[sin²(θ₂-μ₂)])

    Args:
        t_uniformity: Temperature for uniformity loss. Default 2.0.
        spread_weight: Relative weight of spread vs uniformity. Default 1.0.
    """

    def __init__(self, t_uniformity: float = 2.0, spread_weight: float = 1.0):
        super().__init__()
        self.t = t_uniformity
        self.spread_weight = spread_weight

    def forward(
        self,
        angles: torch.Tensor,
        torus_embed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute uniformity + spread losses.

        Args:
            angles: (B, 2) raw angles on T².
            torus_embed: (B, 4) cos/sin embedding on S¹×S¹ ⊂ R⁴.

        Returns:
            Dict with 'uniformity', 'spread', 'total' losses.
        """
        B = torus_embed.shape[0]

        # --- Uniformity loss: log E[exp(-t·||z_i - z_j||²)] ---
        # Pairwise squared distances in R⁴
        sq_dists = torch.cdist(torus_embed, torus_embed, p=2).pow(2)  # (B, B)
        # Mask diagonal
        mask = ~torch.eye(B, dtype=torch.bool, device=torus_embed.device)
        # Log-mean-exp for numerical stability
        neg_dists = -self.t * sq_dists
        neg_dists = neg_dists.masked_select(mask).view(B, B - 1)
        uniformity = torch.logsumexp(neg_dists, dim=1).mean() - math.log(B - 1)

        # --- Spread loss: circular correlation penalty ---
        theta1 = angles[:, 0]  # (B,)
        theta2 = angles[:, 1]  # (B,)
        # Circular means
        mu1 = torch.atan2(torch.sin(theta1).mean(), torch.cos(theta1).mean())
        mu2 = torch.atan2(torch.sin(theta2).mean(), torch.cos(theta2).mean())
        # Centered sin
        s1 = torch.sin(theta1 - mu1)
        s2 = torch.sin(theta2 - mu2)
        # Circular correlation coefficient squared
        num = (s1 * s2).mean()
        den = torch.sqrt((s1.pow(2).mean()) * (s2.pow(2).mean()) + 1e-8)
        rho_sq = (num / den).pow(2)
        spread = rho_sq  # Minimize: push correlation to 0

        total = uniformity + self.spread_weight * spread
        return {
            "uniformity": uniformity,
            "spread": spread,
            "total": total,
        }


class ToroidalJEPALossV2(nn.Module):
    """Complete loss for toroidal EB-JEPA with hard torus projection head.

    L = L_pred + λ_std · L_std + λ_torus · (L_uniformity + L_spread)

    The torus losses operate on the 4D S¹×S¹ embedding from the
    TorusProjectionHead, not on the raw 512D encoder output.

    Args:
        embed_dim: Embedding dimension for std loss.
        lambda_std: Weight for std collapse prevention. Default 10.0.
        lambda_torus: Weight for torus uniformity + spread. Default 10.0.
        t_uniformity: Temperature for uniformity loss. Default 2.0.
        spread_weight: Relative weight of spread within torus loss. Default 1.0.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        lambda_std: float = 10.0,
        lambda_torus: float = 10.0,
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
        predictions: torch.Tensor,
        targets: torch.Tensor,
        encoder_output: torch.Tensor,
        torus_angles: torch.Tensor | None = None,
        torus_embed: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute all losses.

        Args:
            predictions: Predictor output (B, D).
            targets: EMA target encoder output (B, D).
            encoder_output: Online encoder output (B, D).
            torus_angles: (B, 2) angles from TorusProjectionHead.
            torus_embed: (B, 4) cos/sin embedding from TorusProjectionHead.

        Returns:
            Dict with 'total', 'prediction', 'std', 'uniformity', 'spread' losses.
        """
        pred_loss = F.mse_loss(predictions, targets.detach())
        std_loss = self.std_loss(encoder_output)

        result = {
            "prediction": pred_loss,
            "std": std_loss,
        }

        if torus_angles is not None and torus_embed is not None:
            torus_losses = self.torus_loss(torus_angles, torus_embed)
            total = (
                pred_loss
                + self.lambda_std * std_loss
                + self.lambda_torus * torus_losses["total"]
            )
            result["uniformity"] = torus_losses["uniformity"]
            result["spread"] = torus_losses["spread"]
        else:
            total = pred_loss + self.lambda_std * std_loss

        result["total"] = total
        return result


class VICRegLoss(nn.Module):
    """Standard VICReg loss for baseline comparison.

    L = L_pred + λ_std · L_std + λ_cov · L_cov

    where L_cov = (1/D) Σ_{i≠j} C[i,j]² (isotropic, no geometry).
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        lambda_std: float = 25.0,
        lambda_cov: float = 1.0,
    ):
        super().__init__()
        self.lambda_std = lambda_std
        self.lambda_cov = lambda_cov
        self.embed_dim = embed_dim
        self.std_loss = StandardDeviationLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, D = encoder_output.shape

        pred_loss = F.mse_loss(predictions, targets.detach())
        std_loss = self.std_loss(encoder_output)

        # Isotropic covariance loss
        z = encoder_output - encoder_output.mean(dim=0, keepdim=True)
        cov = (z.T @ z) / (B - 1)
        mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
        cov_loss = cov.pow(2).masked_select(mask).sum() / (D * (D - 1))

        total = pred_loss + self.lambda_std * std_loss + self.lambda_cov * cov_loss

        return {
            "total": total,
            "prediction": pred_loss,
            "std": std_loss,
            "covariance": cov_loss,
        }
