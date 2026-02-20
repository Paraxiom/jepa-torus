"""Topological analysis of JEPA embedding spaces.

Phase 1 tools: persistent homology, spectral analysis, UMAP visualization.
Checks whether pretrained EB-JEPA representations exhibit toroidal structure.

Requirements: ripser, persim, umap-learn, scikit-learn, matplotlib
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class TopologyMetrics:
    """Results from topological analysis of an embedding space."""

    # Betti numbers from persistent homology
    betti_0: int = 0  # Connected components
    betti_1: int = 0  # 1-cycles (loops)
    betti_2: int = 0  # 2-cycles (voids)

    # Persistence diagram statistics
    persistence_h0: list[tuple[float, float]] = field(default_factory=list)
    persistence_h1: list[tuple[float, float]] = field(default_factory=list)
    persistence_h2: list[tuple[float, float]] = field(default_factory=list)

    # Spectral properties of k-NN graph
    spectral_gap: float = 0.0  # λ₁ of k-NN Laplacian
    effective_rank: float = 0.0  # exp(entropy(eigenvalues))
    num_eigenvalues: int = 0

    # Comparison with theoretical torus
    spectral_gap_ratio: float = 0.0  # λ₁(embedding) / λ₁(torus)
    bottleneck_distance_h1: float = 0.0  # Bottleneck dist to torus H₁

    # Embedding statistics
    mean_norm: float = 0.0
    std_norm: float = 0.0
    intrinsic_dim_estimate: float = 0.0

    @property
    def has_torus_signature(self) -> bool:
        """Check if Betti numbers match torus: β₀=1, β₁=2, β₂=1."""
        return self.betti_0 == 1 and self.betti_1 == 2 and self.betti_2 == 1

    @property
    def torus_score(self) -> float:
        """Score from 0-1 measuring how toroidal the embedding is.

        Components:
        - Betti number match (0 or 1)
        - Spectral gap ratio closeness to 1.0
        - Low bottleneck distance to torus persistence
        """
        betti_score = float(self.has_torus_signature)
        gap_score = max(0.0, 1.0 - abs(1.0 - self.spectral_gap_ratio))
        return 0.5 * betti_score + 0.5 * gap_score


def extract_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 10000,
) -> np.ndarray:
    """Extract encoder embeddings from a trained JEPA model.

    Args:
        model: Trained JEPA model with .encoder attribute.
        dataloader: DataLoader for evaluation data.
        device: Torch device.
        max_samples: Maximum number of embeddings to extract.

    Returns:
        Embeddings array of shape (N, D).
    """
    model.eval()
    embeddings = []
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            # Get encoder output (before predictor)
            if hasattr(model, "encoder"):
                z = model.encoder(x)
            elif hasattr(model, "online_encoder"):
                z = model.online_encoder(x)
            else:
                z = model(x)

            # Handle sequence outputs: take mean pool
            if z.dim() == 3:
                z = z.mean(dim=1)

            embeddings.append(z.cpu().numpy())
            total += z.shape[0]
            if total >= max_samples:
                break

    embeddings = np.concatenate(embeddings, axis=0)[:max_samples]
    return embeddings


def extract_torus_embeddings(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 10000,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract torus angles and 4D embeddings from a model with torus head.

    Args:
        model: Trained JEPA model with .torus_projection attribute.
        dataloader: DataLoader for evaluation data.
        device: Torch device.
        max_samples: Maximum samples.

    Returns:
        (angles, torus_embed) — each (N, 2) and (N, 4), or (None, None)
        if the model has no torus head.
    """
    if not hasattr(model, "torus_projection") or model.torus_projection is None:
        return None, None

    model.eval()
    all_angles = []
    all_embed = []
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            z = model.online_encoder(x)
            angles, torus_embed = model.torus_projection(z)

            all_angles.append(angles.cpu().numpy())
            all_embed.append(torus_embed.cpu().numpy())
            total += z.shape[0]
            if total >= max_samples:
                break

    angles = np.concatenate(all_angles, axis=0)[:max_samples]
    embed = np.concatenate(all_embed, axis=0)[:max_samples]
    return angles, embed


def compute_torus_topology(
    torus_embed: np.ndarray,
    n_subsample: int = 1000,
    persistence_threshold: float = 0.05,
    verbose: bool = True,
) -> TopologyMetrics:
    """Run persistent homology on 4D torus embeddings (S¹×S¹ ⊂ R⁴).

    This is the correct analysis space for V2 models — 3000 points in
    4D is massive overkill for detecting T² topology (Betti: 1,2,1).

    Args:
        torus_embed: (N, 4) array of (cos θ₁, sin θ₁, cos θ₂, sin θ₂).
        n_subsample: Points to subsample (Rips O(n³)).
        persistence_threshold: Minimum lifetime for significant features.
        verbose: Print results.

    Returns:
        TopologyMetrics with Betti numbers from 4D analysis.
    """
    from ripser import ripser

    metrics = TopologyMetrics()

    N = torus_embed.shape[0]
    if N > n_subsample:
        indices = np.random.choice(N, n_subsample, replace=False)
        X = torus_embed[indices]
    else:
        X = torus_embed

    if verbose:
        print(f"Running persistent homology on {X.shape[0]} pts in R^{X.shape[1]} (torus branch)")

    # No normalization needed — points already on S¹×S¹ (norm ≈ √2)
    result = ripser(X, maxdim=2)
    diagrams = result["dgms"]

    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])
        significant = (lifetimes > persistence_threshold).sum()

        if dim == 0:
            metrics.betti_0 = 1 + int(significant)
            metrics.persistence_h0 = [tuple(p) for p in finite]
        elif dim == 1:
            metrics.betti_1 = int(significant)
            metrics.persistence_h1 = [tuple(p) for p in finite]
        elif dim == 2:
            metrics.betti_2 = int(significant)
            metrics.persistence_h2 = [tuple(p) for p in finite]

    metrics.mean_norm = float(np.linalg.norm(torus_embed, axis=1).mean())
    metrics.std_norm = float(np.linalg.norm(torus_embed, axis=1).std())
    metrics.intrinsic_dim_estimate = 2.0  # By construction

    if verbose:
        print(f"Torus 4D analysis — β₀={metrics.betti_0}, β₁={metrics.betti_1}, β₂={metrics.betti_2}")

    return metrics


def compute_persistent_homology(
    embeddings: np.ndarray,
    max_dim: int = 2,
    n_subsample: int = 1000,
    max_edge_length: float = float("inf"),
    use_pca: int | None = None,
) -> TopologyMetrics:
    """Compute persistent homology of embedding point cloud.

    Uses Vietoris-Rips filtration. Subsamples for tractability.

    Args:
        embeddings: (N, D) array of embeddings.
        max_dim: Maximum homology dimension (2 for torus detection).
        n_subsample: Number of points to subsample (Rips is O(n³)).
        max_edge_length: Maximum edge length in Rips complex.
        use_pca: If set, PCA-reduce to this many dimensions before persistence.
            Helps with high-D embeddings where covering number is exponential.

    Returns:
        TopologyMetrics with Betti numbers and persistence diagrams.
    """
    from ripser import ripser

    metrics = TopologyMetrics()

    # Subsample if needed
    N = embeddings.shape[0]
    if N > n_subsample:
        indices = np.random.choice(N, n_subsample, replace=False)
        X = embeddings[indices]
    else:
        X = embeddings

    # Optional PCA reduction (helps high-D → tractable persistence)
    if use_pca is not None and X.shape[1] > use_pca:
        from sklearn.decomposition import PCA
        X = PCA(n_components=use_pca).fit_transform(X)

    # Normalize to unit sphere for better persistence
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_norm = X / norms

    # Compute persistence
    result = ripser(
        X_norm,
        maxdim=max_dim,
        thresh=max_edge_length,
    )

    diagrams = result["dgms"]

    # Extract Betti numbers (count features with persistence > threshold)
    persistence_threshold = 0.1  # Features living longer than this
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        # Filter out infinite death (H₀ always has one)
        finite = dgm[np.isfinite(dgm[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0] if len(finite) > 0 else np.array([])
        significant = (lifetimes > persistence_threshold).sum()

        if dim == 0:
            # β₀ includes the one infinite bar (connected component)
            metrics.betti_0 = 1 + int(significant)
            metrics.persistence_h0 = [tuple(p) for p in finite]
        elif dim == 1:
            metrics.betti_1 = int(significant)
            metrics.persistence_h1 = [tuple(p) for p in finite]
        elif dim == 2:
            metrics.betti_2 = int(significant)
            metrics.persistence_h2 = [tuple(p) for p in finite]

    # Embedding statistics
    metrics.mean_norm = float(np.linalg.norm(embeddings, axis=1).mean())
    metrics.std_norm = float(np.linalg.norm(embeddings, axis=1).std())

    return metrics


def compute_spectral_analysis(
    embeddings: np.ndarray,
    k: int = 15,
    grid_size: int = 12,
    n_subsample: int = 2000,
) -> TopologyMetrics:
    """Spectral analysis of embedding k-NN graph.

    Builds a k-NN graph, computes graph Laplacian eigenvalues, and compares
    the spectral gap to the theoretical torus spectral gap.

    Args:
        embeddings: (N, D) array.
        k: Number of nearest neighbors.
        grid_size: Torus grid size for comparison.
        n_subsample: Subsample size.

    Returns:
        TopologyMetrics with spectral properties.
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix

    metrics = TopologyMetrics()

    # Subsample
    N = embeddings.shape[0]
    if N > n_subsample:
        indices = np.random.choice(N, n_subsample, replace=False)
        X = embeddings[indices]
    else:
        X = embeddings

    # Build k-NN graph (symmetric)
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False)
    A = ((A + A.T) > 0).astype(float)  # Symmetrize

    # Graph Laplacian: L = D - A
    degrees = np.array(A.sum(axis=1)).flatten()
    D = csr_matrix(np.diag(degrees))
    L = D - A

    # Compute smallest eigenvalues
    n_eigs = min(20, X.shape[0] - 2)
    try:
        eigenvalues, _ = eigsh(L.toarray(), k=n_eigs, which="SM")
        eigenvalues = np.sort(np.real(eigenvalues))

        # Spectral gap: second smallest eigenvalue
        # (smallest should be ~0 for connected graph)
        metrics.spectral_gap = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        metrics.num_eigenvalues = len(eigenvalues)

        # Effective rank from eigenvalue distribution
        pos_eigs = eigenvalues[eigenvalues > 1e-10]
        if len(pos_eigs) > 0:
            p = pos_eigs / pos_eigs.sum()
            entropy = -np.sum(p * np.log(p + 1e-15))
            metrics.effective_rank = float(np.exp(entropy))
    except Exception:
        pass

    # Theoretical torus spectral gap
    torus_gap = 2.0 - 2.0 * math.cos(2.0 * math.pi / grid_size)
    if torus_gap > 0:
        metrics.spectral_gap_ratio = metrics.spectral_gap / torus_gap

    return metrics


def compute_covariance_metrics(embeddings: np.ndarray) -> dict:
    """Compute covariance matrix properties of embeddings.

    Returns:
        Dict with effective_rank, condition_number, explained_variance_ratio.
    """
    # Center
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    N, D = X.shape

    # Covariance
    cov = (X.T @ X) / (N - 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending
    eigenvalues = np.maximum(eigenvalues, 0)  # Clip negative

    # Effective rank: exp(entropy of normalized eigenvalues)
    total = eigenvalues.sum()
    if total > 0:
        p = eigenvalues / total
        p = p[p > 1e-15]
        entropy = -np.sum(p * np.log(p))
        effective_rank = float(np.exp(entropy))
    else:
        effective_rank = 0.0

    # Condition number
    nonzero = eigenvalues[eigenvalues > 1e-10]
    condition_number = float(nonzero[0] / nonzero[-1]) if len(nonzero) > 1 else float("inf")

    # Explained variance
    cumsum = np.cumsum(eigenvalues)
    if total > 0:
        explained_90 = int(np.searchsorted(cumsum, 0.9 * total)) + 1
        explained_99 = int(np.searchsorted(cumsum, 0.99 * total)) + 1
    else:
        explained_90 = D
        explained_99 = D

    return {
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "dims_for_90pct_var": explained_90,
        "dims_for_99pct_var": explained_99,
        "top_eigenvalue": float(eigenvalues[0]),
        "eigenvalue_sum": float(total),
    }


def compute_intrinsic_dimension(
    embeddings: np.ndarray, k: int = 10, n_subsample: int = 2000
) -> float:
    """Estimate intrinsic dimensionality using MLE (Levina & Bickel, 2004).

    For a torus, intrinsic dim should be ~2.
    """
    N = embeddings.shape[0]
    if N > n_subsample:
        indices = np.random.choice(N, n_subsample, replace=False)
        X = embeddings[indices]
    else:
        X = embeddings

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    distances = distances[:, 1:]  # Exclude self

    # MLE estimator
    T_k = distances[:, -1:]  # k-th neighbor distance
    T_k = np.maximum(T_k, 1e-10)
    log_ratios = np.log(distances / T_k)
    log_ratios = log_ratios[:, :-1]  # Exclude last (log(1) = 0)

    # m_hat = 1 / mean(log(T_k / T_j)) for j < k
    mean_log = log_ratios.mean(axis=1)
    mean_log = mean_log[mean_log < -1e-10]  # Only negative values (T_j < T_k)

    if len(mean_log) > 0:
        return float(-1.0 / mean_log.mean())
    return 0.0


def umap_projection(
    embeddings: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_subsample: int = 5000,
) -> np.ndarray:
    """UMAP dimensionality reduction for visualization.

    Args:
        embeddings: (N, D) array.
        n_components: Output dimensions (3 for toroidal visualization).
        n_neighbors: UMAP neighbor parameter.
        min_dist: UMAP minimum distance.
        n_subsample: Subsample size.

    Returns:
        (N, n_components) projected embeddings.
    """
    import umap

    N = embeddings.shape[0]
    if N > n_subsample:
        indices = np.random.choice(N, n_subsample, replace=False)
        X = embeddings[indices]
    else:
        X = embeddings

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    return reducer.fit_transform(X)


def full_analysis(
    embeddings: np.ndarray,
    grid_size: int = 12,
    verbose: bool = True,
    torus_embed: np.ndarray | None = None,
) -> TopologyMetrics:
    """Run complete topological analysis pipeline.

    Combines persistent homology, spectral analysis, covariance metrics,
    and intrinsic dimension estimation.

    When torus_embed is provided (V2 models), runs persistent homology
    on the 4D S¹×S¹ data instead of the 512D encoder space.

    Args:
        embeddings: (N, D) array — 512D encoder output.
        grid_size: Torus grid size for comparison.
        verbose: Print results.
        torus_embed: (N, 4) array from TorusProjectionHead, or None.

    Returns:
        TopologyMetrics with all fields populated.
    """
    if verbose:
        print(f"Analyzing {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # Persistent homology — choose analysis space
    if torus_embed is not None:
        if verbose:
            print("Computing persistent homology on 4D torus branch...")
        metrics = compute_torus_topology(torus_embed, verbose=verbose)
    else:
        if verbose:
            print("Computing persistent homology (PCA→10D fallback)...")
        metrics = compute_persistent_homology(embeddings, use_pca=10)

    # Spectral analysis (always on encoder space)
    if verbose:
        print("Computing spectral analysis...")
    spectral = compute_spectral_analysis(embeddings, grid_size=grid_size)
    metrics.spectral_gap = spectral.spectral_gap
    metrics.spectral_gap_ratio = spectral.spectral_gap_ratio
    metrics.effective_rank = spectral.effective_rank
    metrics.num_eigenvalues = spectral.num_eigenvalues

    # Intrinsic dimension
    if verbose:
        print("Estimating intrinsic dimension...")
    if torus_embed is not None:
        metrics.intrinsic_dim_estimate = compute_intrinsic_dimension(torus_embed)
    else:
        metrics.intrinsic_dim_estimate = compute_intrinsic_dimension(embeddings)

    if verbose:
        analysis_space = "4D torus branch" if torus_embed is not None else "512D encoder"
        print(f"\n{'='*50}")
        print(f"TOPOLOGICAL ANALYSIS RESULTS ({analysis_space})")
        print(f"{'='*50}")
        print(f"Betti numbers: β₀={metrics.betti_0}, β₁={metrics.betti_1}, β₂={metrics.betti_2}")
        print(f"Torus signature (1,2,1): {'YES' if metrics.has_torus_signature else 'NO'}")
        print(f"Spectral gap (k-NN): {metrics.spectral_gap:.4f}")
        torus_gap = 2.0 - 2.0 * math.cos(2.0 * math.pi / grid_size)
        print(f"Torus spectral gap (N={grid_size}): {torus_gap:.4f}")
        print(f"Spectral gap ratio: {metrics.spectral_gap_ratio:.4f}")
        print(f"Effective rank: {metrics.effective_rank:.2f}")
        print(f"Intrinsic dimension: {metrics.intrinsic_dim_estimate:.2f}")
        print(f"Mean embedding norm: {metrics.mean_norm:.4f}")
        print(f"Torus score: {metrics.torus_score:.4f}")
        print(f"{'='*50}")

    return metrics
