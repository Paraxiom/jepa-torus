#!/bin/bash
set -euo pipefail
cd /workspace/jepa-torus

pip install -q ripser persim umap-learn pyyaml scikit-learn

for SCALE in 0.05 0.2 0.3; do
    SCALE_PCT=$(python3 -c "print(int(${SCALE}*100))")
    DIR="toroidal_v8b_scale${SCALE_PCT}"

    echo ""
    echo "============================================"
    echo "  V8b Sweep: gradient_scale=${SCALE} (${SCALE_PCT}%)"
    echo "  $(date)"
    echo "============================================"

    # Generate config on the fly
    cat > /tmp/v8b_scale${SCALE_PCT}.yaml << YAMLEOF
model:
  embed_dim: 512
  hidden_dim: 1024
  ema_decay: 0.996
  torus_dim: 2
  n_modes: 6
  torus_hidden: 128
  predictor_hidden: 256
  karmonic_grad_scale: ${SCALE}

training:
  epochs: 300
  batch_size: 256
  lr: 0.001
  weight_decay: 0.05
  warmup_epochs: 10
  num_workers: 4
  seed: 42

loss:
  type: toroidal_v8b
  grid_size: 12
  lambda_std: 25.0
  lambda_torus_pred: 0.5
  lambda_karmonic: 5.0
  t_uniformity: 2.0
  spread_weight: 1.0

data:
  dataset: cifar10
  data_dir: ./data

output:
  dir: ./checkpoints/${DIR}
  save_every: 50
YAMLEOF

    python -m src.train_v8b --config /tmp/v8b_scale${SCALE_PCT}.yaml 2>&1

    echo "  V8b (${SCALE_PCT}%) training complete: $(date)"

    python -m src.eval \
        --checkpoint checkpoints/${DIR}/best.pt \
        --linear-probe \
        --analysis \
        --output-dir results/${DIR} 2>&1

    echo "============================================"
    echo "  V8b (${SCALE_PCT}%) COMPLETE: $(date)"
    echo "============================================"
done

# ===================== Summary =====================
echo ""
echo "============================================"
echo "  SWEEP COMPLETE — ALL RESULTS"
echo "============================================"

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
