#!/bin/bash
set -euo pipefail
cd /workspace/jepa-torus

pip install -q ripser persim umap-learn pyyaml scikit-learn

# V2 N12: training done, just needs eval
if [ -f "checkpoints/toroidal_v2_N12/best.pt" ] && [ ! -f "results/toroidal_v2_N12/eval_results.json" ]; then
    mkdir -p results/toroidal_v2_N12
    echo "=== Evaluating: toroidal_v2_N12 $(date) ==="
    python -m src.eval --checkpoint checkpoints/toroidal_v2_N12/best.pt --linear-probe --analysis --output-dir results/toroidal_v2_N12
    echo "=== Done: toroidal_v2_N12 $(date) ==="
fi

# V2 N8: needs training + eval
mkdir -p checkpoints/toroidal_v2_N8 results/toroidal_v2_N8
echo "=== Training: toroidal_v2_N8 $(date) ==="
python -m src.train --config configs/toroidal_v2_N8.yaml
echo "=== Evaluating: toroidal_v2_N8 $(date) ==="
python -m src.eval --checkpoint checkpoints/toroidal_v2_N8/best.pt --linear-probe --analysis --output-dir results/toroidal_v2_N8
echo "=== Done: toroidal_v2_N8 $(date) ==="

# Summary table
echo ""
echo "============================================"
echo "  ALL DONE: $(date)"
echo "============================================"
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
    print(f"{d.name:<25} {acc:>10.4f} {er:>10.2f} {sg:>10.4f} {ts:>8.4f} {sp:>12}")
PYEOF
