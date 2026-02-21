#!/bin/bash
set -euo pipefail
cd /workspace/jepa-torus

pip install -q ripser persim umap-learn pyyaml scikit-learn

echo "============================================"
echo "  V4 Toroidal Bottleneck Training"
echo "  Prediction flows THROUGH the torus"
echo "  $(date)"
echo "============================================"

mkdir -p checkpoints/toroidal_v4_N12 results/toroidal_v4_N12

python -m src.train_v4 --config configs/toroidal_v4_N12.yaml 2>&1

echo ""
echo "=== Evaluating V4 N12 $(date) ==="

python -m src.eval \
    --checkpoint checkpoints/toroidal_v4_N12/best.pt \
    --linear-probe --analysis \
    --output-dir results/toroidal_v4_N12 2>&1

echo ""
echo "============================================"
echo "  V4 DONE: $(date)"
echo "============================================"

# Summary table (all configs)
python3 << 'PYEOF'
import json
from pathlib import Path

header = f"{'Config':<25} {'Accuracy':>10} {'Eff.Rank':>10} {'Spec.Gap':>10} {'Torus':>8} {'Space':>12}"
print(header)
print("=" * 80)
for d in sorted(Path("results").iterdir()):
    f = d / "eval_results.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    acc = r.get("linear_probe", {}).get("best_test_acc", 0)
    er = r.get("covariance", {}).get("effective_rank", 0)
    sg = r.get("topology", {}).get("spectral_gap", 0)
    ts = r.get("topology", {}).get("torus_score", 0)
    sp = r.get("topology", {}).get("analysis_space", "encoder")
    b0 = r.get("topology", {}).get("betti_0", "-")
    b1 = r.get("topology", {}).get("betti_1", "-")
    b2 = r.get("topology", {}).get("betti_2", "-")
    idim = r.get("topology", {}).get("intrinsic_dim", 0)
    print(f"{d.name:<25} {acc:>10.4f} {er:>10.2f} {sg:>10.4f} {ts:>8.4f} {sp:>12}  b=({b0},{b1},{b2})  dim={idim:.1f}")
PYEOF
