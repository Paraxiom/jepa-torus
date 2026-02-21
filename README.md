# jepa-torus

Toroidal Geometry in Joint-Embedding Predictive Architectures: From Logit Bias to Representation Regularization.

Replaces VICReg's isotropic covariance penalty in EB-JEPA with a geometry-aware regularizer based on the N x N torus Laplacian (Tonnetz geometry). The torus spectral gap provides a provable lower bound on representation diversity.

## Results (Feb 20, 2026)

All 7 configs trained on CIFAR-10 (300 epochs, RTX 4090). V1 uses soft covariance shaping in 512D encoder space. V2 adds a hard torus projection head mapping to S1xS1 in R4.

| Config | Accuracy | Eff.Rank | Spec.Gap | Torus Score | Betti (b0,b1,b2) | Intrinsic Dim |
|--------|----------|----------|----------|-------------|-------------------|---------------|
| baseline_vicreg | **72.12%** | 93.85 | 1.3324 | 0.0000 | (1000,241,0) | 8.9 |
| baseline_sigreg | 72.11% | 68.29 | 1.1591 | 0.0000 | (998,195,0) | 9.2 |
| toroidal_N8 | 72.30% | 37.78 | 1.0115 | 0.0000 | (999,145,1) | 8.3 |
| toroidal_N16 | 72.13% | 41.92 | 1.0743 | 0.0000 | (1000,172,0) | 8.1 |
| toroidal_N12 | 71.47% | 41.45 | 0.9974 | 0.0000 | (999,164,3) | 8.6 |
| toroidal_v2_N12 | 44.68% | 7.09 | 0.0574 | **0.1071** | (845,112,**1**) | **1.9** |
| toroidal_v2_N8 | 42.54% | 7.26 | 0.0513 | **0.0437** | (872,120,**1**) | **1.9** |

### V1 Findings (Soft Covariance Shaping)

- All V1 configs achieve ~72% linear probe accuracy (comparable to baselines)
- Toroidal covariance penalty reduces effective rank (38-42 vs 68-94 for baselines)
- **No toroidal structure detected** (torus_score=0.0000 across all V1 configs)
- Betti numbers show ~1000 disconnected components — noise, not topology
- Intrinsic dimension ~8-9 (far from the target of 2)
- **Root cause**: Soft covariance shaping in 512D is too indirect. The torus penalty shapes eigenvalue distribution but doesn't constrain the topology of the embedding manifold

### V2 Findings (Hard Torus Projection Head)

V2 adds `TorusProjectionHead`: Linear(512->128)->BN->ReLU->Linear(128->2)->sigmoid*2pi->cos/sin, mapping to S1xS1 in R4. Analysis runs on the 4D torus branch.

**What worked:**
- **Intrinsic dim = 1.9** (target: 2.0) — points lie on a 2-manifold
- **b2 = 1** in both V2 configs — correct for a torus (vs 0 in all V1/baselines)
- **First non-zero torus scores** (0.107 for N12, 0.044 for N8)
- Hard constraint successfully places points on the torus surface

**What didn't work:**
- **b0 ~ 850** (should be 1) — point cloud is fragmented into ~850 disconnected clusters
- **b1 ~ 115** (should be 2) — too many spurious 1-cycles from fragmentation
- **Accuracy dropped to ~44%** (from ~72%) — too much capacity sacrificed to the torus head
- **Spectral gap ratio ~0.1-0.2** (should be ~1.0) — clusters don't form a connected manifold
- **Diagnosis**: Prediction loss dominates and creates class-conditional clusters on the torus. The Wang-Isola uniformity loss (lambda_torus=10) isn't strong enough to connect them

### V3 Design Direction (Next)

The V2 results confirm the hard constraint approach is correct (intrinsic dim, b2) but needs better training strategy.

**Curriculum learning (recommended):**
1. Phase 1 (epochs 1-200): Train encoder only with VICReg/SigReg loss -> 72% accuracy
2. Phase 2 (epochs 200-400): Freeze encoder, train only torus head with 10x uniformity weight
3. Phase 3 (epochs 400-500): Fine-tune both with balanced losses

**Alternative approaches:**
- **Contrastive torus loss**: Pull same-class points together on the torus, push different-class apart (supervised signal)
- **Spectral connectivity loss**: Penalize Fiedler value of k-NN graph on torus embeddings (directly optimizes b0->1)
- **Temperature annealing**: Start with very high uniformity weight, decay as torus structure forms
- **Larger torus head**: Add capacity (256->64->2 instead of 128->2) so encoder doesn't sacrifice representations

## Quick Start

```bash
pip install -r requirements.txt

# Train baseline VICReg
python -m src.train --config configs/baseline_vicreg.yaml

# Train with toroidal regularization (N=12, Tonnetz geometry)
python -m src.train --config configs/toroidal_N12.yaml

# Train V2 with hard torus projection head
python -m src.train --config configs/toroidal_v2_N12.yaml

# Evaluate: linear probing + topological analysis
python -m src.eval --checkpoint checkpoints/toroidal_v2_N12/best.pt --linear-probe --analysis

# Run all 7 configs (overnight, RTX 4090 ~7h)
nohup bash run_all.sh > run_all.log 2>&1 &
```

## Training Configs

| Config | Loss | Grid | Head | Notes |
|--------|------|------|------|-------|
| `baseline_vicreg` | VICReg | - | - | Standard isotropic covariance penalty |
| `baseline_sigreg` | VICReg (low cov) | - | - | SIGReg proxy |
| `toroidal_N8` | Toroidal V1 | 8x8 | - | Soft covariance shaping (lambda_1=0.586) |
| `toroidal_N12` | Toroidal V1 | 12x12 | - | Tonnetz geometry (lambda_1=0.268) |
| `toroidal_N16` | Toroidal V1 | 16x16 | - | Gentle penalty gradient (lambda_1=0.152) |
| `toroidal_v2_N12` | Toroidal V2 | 12x12 | S1xS1 | Hard torus projection + uniformity |
| `toroidal_v2_N8` | Toroidal V2 | 8x8 | S1xS1 | Hard torus projection + uniformity |

## Evaluation Metrics

- **Linear probing accuracy**: CIFAR-10 classification with frozen encoder
- **Effective rank**: exp(entropy(eigenvalues(Cov))) — higher means less collapse
- **Betti numbers**: Persistent homology. Torus signature: b0=1, b1=2, b2=1
- **Spectral gap ratio**: lambda_1(embedding k-NN graph) / lambda_1(theoretical torus)
- **Intrinsic dimension**: MLE estimator (torus should give ~2)
- **Torus score**: 0.5 * betti_match + 0.5 * spectral_gap_closeness

## Architecture

### V1: Soft Covariance Shaping
```
Input -> ResNet-18 Encoder (512D) -> Predictor -> Loss
                                  |
                                  +-> Torus Laplacian covariance penalty (512D)
```

### V2: Hard Torus Projection
```
Input -> ResNet-18 Encoder (512D) -> Predictor -> Prediction Loss
                |                                       |
                +-> TorusProjectionHead -> (cos,sin)x2 -> Uniformity + Spread Loss
                    Linear(512,128)->BN->ReLU               (4D on S1xS1)
                    Linear(128,2)->sigmoid*2pi
                    -> (theta1, theta2) -> (cos,sin,cos,sin)
```

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
├── configs/               # Training YAML configs (7 configs)
├── src/
│   ├── toroidal_loss.py   # V1 ToroidalCovarianceLoss + V2 ToroidalJEPALossV2
│   ├── analysis.py        # Persistent homology + spectral analysis + torus topology
│   ├── train.py           # EB-JEPA training loop + TorusProjectionHead
│   └── eval.py            # Linear probing + topology evaluation
├── notebooks/
│   └── analysis.ipynb     # Self-contained RunPod/Colab notebook
├── lean/
│   └── JepaTorus/
│       └── ToroidalRegularizer.lean
├── run_all.sh             # Full 7-config training pipeline
├── run_v2_remaining.sh    # V2 eval + N8 training
└── requirements.txt
```

## References

- Cormier (2026), "Topological Constraints for Coherent Language Models" (Zenodo)
- Bardes et al. (2022), VICReg: Variance-Invariance-Covariance Regularization
- Assran et al. (2023), Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture
- Gardner et al. (2022), Toroidal topology of population activity in grid cells (Nature)
- Wang & Isola (2020), Understanding Contrastive Representation Learning through Alignment and Uniformity
