#!/bin/bash
set -euo pipefail
cd /workspace/jepa-torus

pip install -q ripser persim umap-learn pyyaml scikit-learn

echo "============================================"
echo "  V5b: Fourier Torus + Angle-Space Prediction"
echo "  Circular distance on angles (no high-mode bias)"
echo "  $(date)"
echo "============================================"

# Train V5b
python -m src.train_v5 --config configs/toroidal_v5b_angles.yaml 2>&1

echo ""
echo "============================================"
echo "  V5b training complete: $(date)"
echo "  Starting evaluation..."
echo "============================================"

# Evaluate
python -m src.eval \
    --checkpoint checkpoints/toroidal_v5b_angles/best.pt \
    --linear-probe \
    --analysis \
    --output-dir results/toroidal_v5b_angles 2>&1

echo ""
echo "============================================"
echo "  V5b COMPLETE: $(date)"
echo "============================================"

# Summary
python3 << 'PYEOF'
import json
from pathlib import Path

print(f"\n{'Config':<25} {'Enc Acc':>10} {'Torus Acc':>10} {'Torus Dim':>10} {'b0':>6} {'b1':>6} {'b2':>6}")
print("=" * 80)
for d in sorted(Path("results").iterdir()):
    f = d / "eval_results.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    enc_acc = r.get("linear_probe", {}).get("best_test_acc", 0)
    torus_acc = r.get("torus_probe", {}).get("best_test_acc", 0)
    torus_dim = r.get("torus_probe", {}).get("torus_embed_dim", "-")
    topo = r.get("topology", {})
    b0 = topo.get("betti_0", "-")
    b1 = topo.get("betti_1", "-")
    b2 = topo.get("betti_2", "-")
    if torus_acc > 0:
        print(f"{d.name:<25} {enc_acc:>10.4f} {torus_acc:>10.4f} {torus_dim:>10} {b0:>6} {b1:>6} {b2:>6}")
    else:
        print(f"{d.name:<25} {enc_acc:>10.4f} {'n/a':>10} {'n/a':>10} {b0:>6} {b1:>6} {b2:>6}")
PYEOF
