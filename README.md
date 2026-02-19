# jepa-torus

Toroidal Geometry in Joint-Embedding Predictive Architectures: From Logit Bias to Representation Regularization.

Replaces VICReg's isotropic covariance penalty in EB-JEPA with a geometry-aware regularizer based on the N x N torus Laplacian (Tonnetz geometry). The torus spectral gap provides a provable lower bound on representation diversity.

## Quick Start

```bash
pip install -r requirements.txt

# Train baseline VICReg
python -m src.train --config configs/baseline_vicreg.yaml

# Train with toroidal regularization (N=12, Tonnetz geometry)
python -m src.train --config configs/toroidal_N12.yaml

# Evaluate: linear probing + topological analysis
python -m src.eval --checkpoint checkpoints/toroidal_N12/best.pt --linear-probe --analysis
```

## Training Configs

| Config | Loss | Grid | Notes |
|--------|------|------|-------|
| `baseline_vicreg` | VICReg | - | Standard isotropic covariance penalty |
| `baseline_sigreg` | VICReg (low cov) | - | SIGReg proxy |
| `toroidal_N8` | Toroidal | 8x8 | Strong penalty gradient (lambda_1=0.586) |
| `toroidal_N12` | Toroidal | 12x12 | Tonnetz geometry (lambda_1=0.268) |
| `toroidal_N16` | Toroidal | 16x16 | Gentle penalty gradient (lambda_1=0.152) |

## Ablations

Override hyperparameters from the command line:

```bash
# Lambda sweep
python -m src.train --config configs/toroidal_N12.yaml --lambda-torus 0.01
python -m src.train --config configs/toroidal_N12.yaml --lambda-torus 0.1
python -m src.train --config configs/toroidal_N12.yaml --lambda-torus 10.0

# Grid size sweep
python -m src.train --config configs/toroidal_N12.yaml --grid-size 8
python -m src.train --config configs/toroidal_N12.yaml --grid-size 16

# Multi-seed
for seed in 42 123 456; do
  python -m src.train --config configs/toroidal_N12.yaml --seed $seed --output-dir checkpoints/toroidal_N12_s$seed
done
```

## Evaluation Metrics

- **Linear probing accuracy**: CIFAR-10 classification with frozen encoder
- **Effective rank**: exp(entropy(eigenvalues(Cov))) — higher means less collapse
- **Betti numbers**: Persistent homology. Torus signature: beta_0=1, beta_1=2, beta_2=1
- **Spectral gap ratio**: lambda_1(embedding k-NN graph) / lambda_1(theoretical torus)
- **Intrinsic dimension**: MLE estimator (torus should give ~2)

## Lean 4 Proofs

67 theorems with zero sorries (Mathlib v4.27.0):

```bash
cd lean
lake build
```

Proves: torus Laplacian PSD properties, spectral gap ordering, dimension-to-torus coverage, Betti number constraints, Euler characteristic.

## Project Structure

```
jepa-torus/
├── configs/           # Training YAML configs
├── src/
│   ├── toroidal_loss.py   # ToroidalCovarianceLoss + VICRegLoss
│   ├── analysis.py        # Persistent homology + spectral analysis
│   ├── train.py           # EB-JEPA training loop
│   └── eval.py            # Linear probing + topology evaluation
├── notebooks/
│   └── analysis.ipynb     # Self-contained RunPod/Colab notebook
├── lean/
│   └── JepaTorus/
│       └── ToroidalRegularizer.lean
└── requirements.txt
```

## References

- Cormier (2026), "Topological Constraints for Coherent Language Models" (Zenodo)
- Bardes et al. (2022), VICReg: Variance-Invariance-Covariance Regularization
- Assran et al. (2023), Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- Gardner et al. (2022), Toroidal topology of population activity in grid cells (Nature)
