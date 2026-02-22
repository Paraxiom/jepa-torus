"""Resume V7 training from checkpoint."""
import json, math, random, time
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

import sys
sys.path.insert(0, "/workspace/jepa-torus")
from src.train import ResNetEncoder, Predictor, JEPAAugmentation
from src.train_v6 import EBJEPA_V6
from src.train_v7 import AdaptiveDualPathLoss

# Load checkpoint
ckpt_path = "/workspace/jepa-torus/checkpoints/toroidal_v7_adaptive/checkpoint_0150.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
config = ckpt["config"]
start_epoch = ckpt["epoch"] + 1
history = ckpt.get("history", [])
print(f"Resuming from epoch {start_epoch}")

# Rebuild model
model_cfg = config["model"]
device = torch.device("cuda")
model = EBJEPA_V6(
    embed_dim=model_cfg["embed_dim"],
    hidden_dim=model_cfg.get("hidden_dim", 1024),
    ema_decay=model_cfg.get("ema_decay", 0.996),
    torus_dim=model_cfg.get("torus_dim", 2),
    n_modes=model_cfg.get("n_modes", 6),
    torus_hidden=model_cfg.get("torus_hidden", 128),
    predictor_hidden=model_cfg.get("predictor_hidden", 256),
).to(device)
model.load_state_dict(ckpt["model_state_dict"])

# Loss
loss_cfg = config["loss"]
criterion = AdaptiveDualPathLoss(
    grid_size=loss_cfg.get("grid_size", 12),
    torus_dim=model_cfg.get("torus_dim", 2),
    n_modes=model_cfg.get("n_modes", 6),
    lambda_std=loss_cfg.get("lambda_std", 25.0),
    lambda_torus_pred=loss_cfg.get("lambda_torus_pred", 0.5),
    lambda_karmonic=loss_cfg.get("lambda_karmonic", 5.0),
    t_uniformity=loss_cfg.get("t_uniformity", 2.0),
    spread_weight=loss_cfg.get("spread_weight", 1.0),
    warmup_karmonic_epochs=loss_cfg.get("warmup_karmonic_epochs", 30),
    gate_width=loss_cfg.get("gate_width", 0.5),
)
# Restore threshold from history
warmup_univs = [h["uniformity"] for h in history if h["epoch"] <= 30]
if warmup_univs:
    criterion.coherence_threshold = warmup_univs[-1]
    criterion.ema_uniformity = history[-1]["uniformity"]
    print(f"Restored threshold: {criterion.coherence_threshold:.4f}, ema_unif: {criterion.ema_uniformity:.4f}")

# Optimizer
train_cfg = config["training"]
optimizer = torch.optim.AdamW(
    model.parameters(), lr=train_cfg.get("lr", 0.001),
    weight_decay=train_cfg.get("weight_decay", 0.05),
)
try:
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print("Optimizer state restored")
except (ValueError, KeyError) as e:
    print(f"Skipping optimizer restore (param mismatch): {e}")

total_epochs = train_cfg.get("epochs", 300)
warmup_epochs = train_cfg.get("warmup_epochs", 10)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-6,
)
# Fast-forward scheduler to current epoch
for _ in range(start_epoch - warmup_epochs - 1):
    scheduler.step()

# Data
seed = train_cfg.get("seed", 42)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
aug = JEPAAugmentation()
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=aug)
dataloader = DataLoader(
    train_dataset, batch_size=train_cfg.get("batch_size", 256),
    shuffle=True, num_workers=train_cfg.get("num_workers", 4),
    pin_memory=True, drop_last=True,
)

out_dir = Path("checkpoints/toroidal_v7_adaptive")
best_loss = min((h["total"] for h in history), default=float("inf"))
t_start = time.time()

print(f"\nResuming V7 from epoch {start_epoch} to {total_epochs}")
print(f"Best loss so far: {best_loss:.4f}")

for epoch in range(start_epoch, total_epochs + 1):
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
            if k == "total" or isinstance(v, list):
                continue
            loss_accum[k] = loss_accum.get(k, 0.0) + (v.item() if torch.is_tensor(v) else v)
        loss_accum["total"] = loss_accum.get("total", 0.0) + losses["total"].item()
        gv = losses.get("gate", 0.0)
        gate_accum += gv.item() if torch.is_tensor(gv) else gv
        n_batches += 1

        if batch_idx % 100 == 0:
            gate_val = losses.get("gate", 0.0)
            if torch.is_tensor(gate_val):
                gate_val = gate_val.item()
            phase = "WARMUP" if epoch <= 30 else "ADAPTIVE"
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"p512={losses['pred_512'].item():.4f} "
                  f"ptor={losses['pred_torus'].item():.4f} "
                  f"gate={gate_val:.3f} [{phase}]")

    avg_losses = {k: v / n_batches for k, v in loss_accum.items()}
    avg_gate = gate_accum / n_batches
    elapsed = time.time() - t_start

    history.append({
        "epoch": epoch, **avg_losses, "gate": avg_gate,
        "lr": optimizer.param_groups[0]["lr"],
    })

    phase = "WARMUP" if epoch <= 30 else "ADAPT"
    print(
        f"Epoch {epoch}/{total_epochs} [{phase}] | "
        f"total={avg_losses['total']:.4f} | "
        f"p512={avg_losses.get('pred_512', 0):.4f} | "
        f"ptor={avg_losses.get('pred_torus', 0):.4f} | "
        f"gate={avg_gate:.3f} | "
        f"unif={avg_losses.get('uniformity', 0):.4f} | "
        f"lr={optimizer.param_groups[0]['lr']:.6f} | "
        f"time={elapsed:.0f}s"
    )

    if avg_losses["total"] < best_loss:
        best_loss = avg_losses["total"]
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, out_dir / "best.pt")

    if epoch % 50 == 0 or epoch == total_epochs:
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config, "history": history,
        }, out_dir / f"checkpoint_{epoch:04d}.pt")

with open(out_dir / "history.json", "w") as f:
    json.dump(history, f, indent=2)

total_time = time.time() - t_start
print(f"\nV7 resume complete: epochs {start_epoch}-{total_epochs}, {total_time:.0f}s ({total_time/3600:.1f}h)")
print(f"Best loss: {best_loss:.4f}")
