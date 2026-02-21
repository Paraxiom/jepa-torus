#!/bin/bash
set -euo pipefail
cd /workspace/jepa-torus

pip install -q ripser persim umap-learn pyyaml scikit-learn

echo "============================================"
echo "  Torus Coordinate Linear Probing"
echo "  Re-evaluating V2, V4, V4b checkpoints"
echo "  $(date)"
echo "============================================"

# Re-eval all configs that have torus heads
for config in toroidal_v2_N12 toroidal_v2_N8 toroidal_v4_N12 toroidal_v4b_T5; do
    ckpt="checkpoints/${config}/best.pt"
    if [ -f "$ckpt" ]; then
        echo ""
        echo "=== ${config} ==="
        python -m src.eval \
            --checkpoint "$ckpt" \
            --linear-probe \
            --output-dir "results/${config}" 2>&1
    else
        echo "SKIP: $ckpt not found"
    fi
done

echo ""
echo "============================================"
echo "  TORUS PROBE DONE: $(date)"
echo "============================================"

# Summary
python3 << 'PYEOF'
import json
from pathlib import Path

print(f"\n{'Config':<25} {'Enc Acc':>10} {'Torus Acc':>10} {'Torus Dim':>10}")
print("=" * 60)
for d in sorted(Path("results").iterdir()):
    f = d / "eval_results.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    enc_acc = r.get("linear_probe", {}).get("best_test_acc", 0)
    torus_acc = r.get("torus_probe", {}).get("best_test_acc", 0)
    torus_dim = r.get("torus_probe", {}).get("torus_embed_dim", "-")
    if torus_acc > 0:
        print(f"{d.name:<25} {enc_acc:>10.4f} {torus_acc:>10.4f} {torus_dim:>10}")
    else:
        print(f"{d.name:<25} {enc_acc:>10.4f} {'n/a':>10} {'n/a':>10}")
PYEOF
