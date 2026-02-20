"""Evaluation for EB-JEPA models: linear probing + topology metrics.

Usage:
    python -m src.eval --checkpoint checkpoints/best.pt --analysis
    python -m src.eval --checkpoint checkpoints/best.pt --linear-probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .train import EBJEPA
from .analysis import (
    extract_embeddings,
    extract_torus_embeddings,
    full_analysis,
    compute_covariance_metrics,
    umap_projection,
)


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------


class LinearProbe(nn.Module):
    """Single linear layer for evaluation."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def linear_probe(
    model: EBJEPA,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    embed_dim: int = 512,
    num_classes: int = 10,
    epochs: int = 100,
    lr: float = 0.01,
) -> dict:
    """Train a linear probe on frozen encoder features.

    Returns dict with train_acc, test_acc, and per-epoch history.
    """
    model.eval()
    probe = LinearProbe(embed_dim, num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_test_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        probe.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                features = model.online_encoder(images)
            logits = probe(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total

        # Test
        probe.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                features = model.online_encoder(images)
                logits = probe(features)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        test_acc = correct / total
        best_test_acc = max(best_test_acc, test_acc)

        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Probe epoch {epoch}/{epochs}: train={train_acc:.4f} test={test_acc:.4f}")

        history.append({"epoch": epoch, "train_acc": train_acc, "test_acc": test_acc})

    return {
        "best_test_acc": best_test_acc,
        "final_test_acc": test_acc,
        "final_train_acc": train_acc,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Embedding extraction with standard transforms
# ---------------------------------------------------------------------------


def get_eval_transforms():
    """Standard CIFAR-10 evaluation transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])


def get_dataloaders(data_dir: str = "./data", batch_size: int = 256):
    """Get CIFAR-10 train and test dataloaders with eval transforms."""
    transform = get_eval_transforms()
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------


def evaluate(
    checkpoint_path: str,
    data_dir: str = "./data",
    run_linear_probe: bool = True,
    run_analysis: bool = True,
    run_umap: bool = False,
    grid_size: int | None = None,
    output_dir: str | None = None,
) -> dict:
    """Full evaluation pipeline.

    Args:
        checkpoint_path: Path to model checkpoint.
        data_dir: CIFAR-10 data directory.
        run_linear_probe: Run linear probing evaluation.
        run_analysis: Run topological analysis.
        run_umap: Run UMAP visualization.
        grid_size: Torus grid size for analysis (None = auto-detect from checkpoint config).
        output_dir: Where to save results.

    Returns:
        Dict with all evaluation results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    # Auto-detect grid_size from training config
    loss_cfg = config.get("loss", {})
    if grid_size is None:
        grid_size = loss_cfg.get("grid_size", 12)
        print(f"Auto-detected grid_size={grid_size} from checkpoint config")

    # Reconstruct model — detect torus_head from loss type
    model_cfg = config.get("model", {})
    embed_dim = model_cfg.get("embed_dim", 512)
    loss_type = loss_cfg.get("type", "toroidal")
    use_torus_head = loss_type == "toroidal_v2"

    model = EBJEPA(
        embed_dim=embed_dim,
        hidden_dim=model_cfg.get("hidden_dim", 1024),
        torus_head=use_torus_head,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if use_torus_head:
        print("Torus projection head: DETECTED (will analyze 4D torus branch)")

    # Data
    train_loader, test_loader = get_dataloaders(data_dir)

    results = {
        "checkpoint": checkpoint_path,
        "config": config,
        "embed_dim": embed_dim,
    }

    # Linear probe (always on 512D encoder)
    if run_linear_probe:
        print("\n--- Linear Probing ---")
        probe_results = linear_probe(
            model, train_loader, test_loader, device,
            embed_dim=embed_dim, num_classes=10,
        )
        results["linear_probe"] = probe_results
        print(f"Best test accuracy: {probe_results['best_test_acc']:.4f}")

    # Extract embeddings for analysis
    torus_embed_np = None
    if run_analysis or run_umap:
        print("\n--- Extracting Embeddings ---")
        embeddings = extract_embeddings(model, test_loader, device, max_samples=10000)
        print(f"Extracted {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

        # Extract torus branch if available
        if use_torus_head:
            _, torus_embed_np = extract_torus_embeddings(model, test_loader, device, max_samples=10000)
            if torus_embed_np is not None:
                print(f"Extracted {torus_embed_np.shape[0]} torus embeddings of dim {torus_embed_np.shape[1]}")

    # Topological analysis
    if run_analysis:
        print("\n--- Topological Analysis ---")
        metrics = full_analysis(embeddings, grid_size=grid_size, torus_embed=torus_embed_np)
        results["topology"] = {
            "betti_0": metrics.betti_0,
            "betti_1": metrics.betti_1,
            "betti_2": metrics.betti_2,
            "has_torus_signature": metrics.has_torus_signature,
            "spectral_gap": metrics.spectral_gap,
            "spectral_gap_ratio": metrics.spectral_gap_ratio,
            "effective_rank": metrics.effective_rank,
            "intrinsic_dim": metrics.intrinsic_dim_estimate,
            "torus_score": metrics.torus_score,
        }
        if use_torus_head:
            results["topology"]["analysis_space"] = "4D_torus_branch"
        else:
            results["topology"]["analysis_space"] = "512D_encoder_pca10"

        # Covariance metrics
        cov_metrics = compute_covariance_metrics(embeddings)
        results["covariance"] = cov_metrics
        print(f"Effective rank: {cov_metrics['effective_rank']:.2f}")
        print(f"Condition number: {cov_metrics['condition_number']:.2f}")

    # UMAP
    if run_umap:
        print("\n--- UMAP Projection ---")
        umap_3d = umap_projection(embeddings, n_components=3)
        results["umap_shape"] = list(umap_3d.shape)

        if output_dir:
            np.save(Path(output_dir) / "umap_3d.npy", umap_3d)
            np.save(Path(output_dir) / "embeddings.npy", embeddings)
            print(f"Saved UMAP and embeddings to {output_dir}")

    # Save results
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Remove non-serializable items
        save_results = {
            k: v for k, v in results.items()
            if k not in ("umap_shape",)
        }
        with open(out_path / "eval_results.json", "w") as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"Results saved to {out_path / 'eval_results.json'}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate EB-JEPA with toroidal analysis")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data dir")
    parser.add_argument("--linear-probe", action="store_true", help="Run linear probing")
    parser.add_argument("--analysis", action="store_true", help="Run topological analysis")
    parser.add_argument("--umap", action="store_true", help="Run UMAP visualization")
    parser.add_argument("--grid-size", type=int, default=None, help="Torus grid size (auto-detect from checkpoint if not set)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    if not args.linear_probe and not args.analysis and not args.umap:
        args.linear_probe = True
        args.analysis = True

    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        run_linear_probe=args.linear_probe,
        run_analysis=args.analysis,
        run_umap=args.umap,
        grid_size=args.grid_size,
        output_dir=args.output_dir or str(Path(args.checkpoint).parent / "eval"),
    )


if __name__ == "__main__":
    main()
