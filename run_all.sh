#!/bin/bash
set -euo pipefail

# Toroidal JEPA — Run all 5 training configs + evaluation
# Usage: cd /workspace/jepa-torus && bash run_all.sh
# RTX 4090: ~1h per config, ~5h total

echo "============================================"
echo "  jepa-torus: Full Training Pipeline"
echo "  $(date)"
echo "============================================"

# Install deps (runpod-torch-v240 has PyTorch already)
pip install -q ripser persim umap-learn pyyaml scikit-learn

# Download CIFAR-10 once
python -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True)"

CONFIGS=(
    "baseline_vicreg"
    "baseline_sigreg"
    "toroidal_N12"
    "toroidal_N8"
    "toroidal_N16"
)

for cfg in "${CONFIGS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Training: ${cfg}"
    echo "  Started: $(date)"
    echo "============================================"

    python -m src.train --config "configs/${cfg}.yaml" 2>&1 | tee "checkpoints/${cfg}/train.log"

    echo ""
    echo "  Evaluating: ${cfg}"
    echo "--------------------------------------------"

    python -m src.eval \
        --checkpoint "checkpoints/${cfg}/best.pt" \
        --linear-probe --analysis \
        --output-dir "results/${cfg}" 2>&1 | tee "results/${cfg}/eval.log"

    echo "  Finished: ${cfg} at $(date)"
done

echo ""
echo "============================================"
echo "  All runs complete: $(date)"
echo "============================================"
echo ""

# Print summary table
python -c "
import json
from pathlib import Path

print(f\"{'Config':<25} {'Accuracy':>10} {'Eff.Rank':>10} {'Spec.Gap':>10} {'Torus':>8}\")
print('='*68)
for d in sorted(Path('results').iterdir()):
    f = d / 'eval_results.json'
    if not f.exists(): continue
    r = json.loads(f.read_text())
    acc = r.get('linear_probe',{}).get('best_test_acc', 0)
    er = r.get('covariance',{}).get('effective_rank', 0)
    sg = r.get('topology',{}).get('spectral_gap', 0)
    ts = r.get('topology',{}).get('torus_score', 0)
    print(f'{d.name:<25} {acc:>10.4f} {er:>10.2f} {sg:>10.4f} {ts:>8.4f}')
"
