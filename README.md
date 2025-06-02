# Orthogonal PCA-BO: Enhanced Bayesian Optimization for High-Dimensional Problems

A Python implementation of Orthogonal Principal Component Analysis-assisted Bayesian Optimization (O-PCA-BO), addressing fundamental exploration limitations in PCA-BO through systematic parallel orthogonal sampling.

## Overview

This repository implements the O-PCA-BO algorithm presented in "Enhancing PCA-Assisted Bayesian Optimization through Parallel Orthogonal Sampling" (Banny, 2025). The method addresses a critical limitation of PCA-BO: exploration restricted exclusively to the subspace spanned by selected principal components, potentially missing global optima in orthogonal directions.

### The Problem with PCA-BO

PCA-assisted Bayesian Optimization reduces computational complexity by operating in a lower-dimensional subspace. However, it exhibits a self-reinforcing limitation: as more points are sampled within the reduced subspace, variance along those dimensions increases, making them more likely to be retained in subsequent iterations. This feedback mechanism can permanently exclude optimal regions in the orthogonal complement space.

### Our Solution: Orthogonal Sampling

O-PCA-BO introduces **parallel orthogonal sampling** where for each candidate selected in the principal component subspace, multiple points are sampled in the $(d-r)$-dimensional orthogonal complement space using Hit-and-Run MCMC sampling. This maintains computational efficiency while enabling exploration of previously inaccessible regions.

## Key Features

### Core Algorithm
- **Systematic orthogonal exploration** in the $(d-r)$-dimensional complement space
- **Adaptive basis rotation** through orthogonal sample feedback
- **Parallel evaluation framework** for computational efficiency
- **Constraint-aware sampling** using polytope Hit-and-Run MCMC

### Algorithmic Enhancements
- **Squared rank-based weighting** to reduce influence of poor-performing points
- **Log Expected Improvement with penalization** for stable bound constraint handling
- **Selective GPR training** using composite value-distance ranking
- **Distance-controlled orthogonal sampling** with adaptive intensity

### Implementation
- **GPU acceleration** via PyTorch and BOTorch
- **Robust numerical handling** with comprehensive error recovery
- **Real-time visualization** for 2D optimization landscapes
- **Comprehensive benchmarking** on COCO BBOB functions F15-F24

## Installation

```bash
git clone https://github.com/IvanBanny/para-ortho-pca-bo.git
cd para-ortho-pca-bo
pip install -r requirements.txt
```

### Dependencies
- **Core**: `torch`, `botorch`, `gpytorch`, `numpy`
- **Optimization**: `ioh` (IOHexperimenter), `pyDOE`
- **Sampling**: Hit-and-Run MCMC implementation
- **Visualization**: `matplotlib`, `polars`, `pandas`

## Quick Start

### Basic Usage

```python
from ioh import get_problem
from Algorithms import O_PCA_BO

# Configure O-PCA-BO with optimized hyperparameters
optimizer = O_PCA_BO(
    budget=200,
    n_DoE=40,                   # 4d initial points
    q=1,                        # Candidates per iteration in reduced space
    ortho_samples=5,            # Orthogonal samples per candidate
    var_threshold=0.95,         # PCA variance threshold
    gpr_p=0.589,               # Fraction of points for GP training
    gpr_val_factor=0.101,      # Value vs distance ranking weight
    onorm_factor=3.027,        # Orthogonal sampling intensity
    acquisition_function="expected_improvement"
)

# Test on BBOB function with weak global structure
problem = get_problem(function_id=20, dimension=40, instance=0)
optimizer(problem)
```

### Reproducing Thesis Results

```bash
# Full experimental comparison on F15-F24
python main.py \
    --algorithms vanilla pca opca \
    --batch 1 5 \
    --dimensions 10 20 40 \
    --problems 15 16 17 18 19 20 21 22 23 24 \
    --runs 30

# Generate performance visualizations
python plots.py --experiment_dir experiment
```

## Algorithm Details

### Mathematical Foundation

O-PCA-BO decomposes any point in the original space as:
```
x = P_r * z_r + P_{d-r} * z_{d-r} + μ' + μ
```
where:
- `P_r`: First r principal components (reduced space)
- `P_{d-r}`: Remaining (d-r) components (orthogonal space)
- `z_r`: Coordinates in reduced space (optimized by acquisition function)
- `z_{d-r}`: Coordinates in orthogonal space (sampled via MCMC)

### Orthogonal Sampling Strategy

For each candidate point `x'` selected in the reduced space:
```
x'_ortho,j = x' + P_{d-r} * δ_j,  j = 1,...,m
```

The displacement vectors `δ_j` are generated using Hit-and-Run MCMC with adaptive distance control:

1. **Over-sampling**: Generate `m × s` samples where `s = max(1, ⌊onorm_factor × √(d-r)⌋)`
2. **Distance selection**: Choose `m` samples closest to origin in orthogonal space

### Key Hyperparameters

Based on Bayesian optimization over 8,640 configurations:

- **`gpr_p = 0.589`**: Proportion of points for GP training (avoids model confusion from overlapping projections)
- **`gpr_val_factor = 0.101`**: Weight for value ranking vs distance ranking in point selection
- **`onorm_factor = 3.027`**: Controls orthogonal sampling intensity (0 = uniform, >0 = closer to candidates)

## Experimental Results

### Performance on BBOB Functions

Testing on functions F15-F24 across dimensions 10, 20, 40:

**Functions with Adequate Global Structure (F15-F19)**:
- O-PCA-BO performs comparably to PCA-BO
- Faster initial convergence due to broader exploration

**Functions with Weak Global Structure (F20-F24)**:
- O-PCA-BO demonstrates dramatic superiority
- PCA-BO often plateaus while O-PCA-BO maintains consistent improvement
- Performance advantage increases with dimension

### Statistical Significance

Wilcoxon rank-sum tests across 30 independent runs confirm significant improvements, particularly on functions F20-F24 where O-PCA-BO often finds solutions orders of magnitude better than PCA-BO.

## Project Structure

```
para-ortho-pca-bo/
├── Algorithms/
│   ├── BayesianOptimization/
│   │   ├── O_PCA_BO.py              # Main O-PCA-BO implementation
│   │   ├── Vanilla_BO.py            # Standard BO baseline
│   │   ├── PenalizedAcqf.py         # Constraint-aware acquisition
│   │   └── AbstractBayesianOptimizer.py
│   ├── Experiment/                  # Benchmarking framework
│   └── utils/
│       ├── taylor.py                # Variance estimation for PCA mappings
│       ├── vis_utils.py             # Real-time 2D visualization
│       └── iohreader/               # BBOB data processing
├── meta-bo/                         # Hyperparameter optimization
├── main.py                          # Experiment runner
├── plots.py                         # Results visualization
└── example.py                       # Simple usage demo
```

## Implementation Highlights

### Robust Numerical Handling
- **Multiple fallback strategies** for GP fitting failures
- **Adaptive restart mechanisms** for acquisition optimization
- **Constraint-aware sampling** ensuring feasibility

### Advanced Features
- **Taylor series approximation** for uncertainty estimation in PCA mappings
- **Real-time 2D visualization** with acquisition function overlays
- **Parallel evaluation** with efficient GPU utilization
- **Comprehensive logging** for reproducibility

### Hyperparameter Optimization
The `meta-bo/` directory contains a complete Bayesian optimization system for tuning O-PCA-BO hyperparameters:

```bash
cd meta-bo
python meta-bo.py                # Run hyperparameter optimization
python meta-bo-slurms.py         # Generate cluster job scripts
python meta-bo-vis.py            # Visualize parameter landscape
```

## Citation

If you use this implementation, please cite:

```
@mastersthesis{banny2025opcabo,
    title={Enhancing PCA-Assisted Bayesian Optimization through Parallel Orthogonal Sampling},
    author={Ivan Banny},
    school={Leiden University},
    year={2025},
    type={Bachelor's thesis}
}
```

## Computational Requirements

Experiments performed using ALICE compute resources (Leiden University):
- **Hardware**: AMD EPYC 9534 (AMD.Zen4) processors
- **Memory**: Scales with dimension (1.5GB sufficient for d=40)
- **GPU**: Optional CUDA acceleration for GP operations

## License

MIT License - see `LICENSE` file for details.
