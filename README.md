# PCA-Assisted Bayesian Optimization

This repository implements PCA-assisted Bayesian Optimization (PCA-BO), a method for scaling up Bayesian Optimization to higher-dimensional problems by incorporating dimensionality reduction via Principal Component Analysis.

## Overview

Bayesian Optimization (BO) is limited in high-dimensional applications due to its:
1. Increasing computational complexity with dimension
2. Reduced convergence rate in higher dimensions
3. Difficulties in exploration-exploitation balance in large search spaces

PCA-BO addresses these limitations by:
1. Learning a linear transformation from evaluated points
2. Selecting dimensions in the transformed space based on data variability
3. Building surrogate models in a reduced space
4. Mapping points back to the original space for function evaluations

This approach enables BO to handle higher-dimensional problems while maintaining good convergence properties.

## Repository Structure

```
para-ortho-pca-bo/
|--- Algorithms/
|   |--- BayesianOptimization/
|   |   |--- AbstractBayesianOptimizer.py  # Base class for BO algorithms
|   |   |--- PCA_BO.py                     # PCA-assisted BO implementation
|   |   |--- Vanilla_BO.py                 # Standard BO implementation
|   |--- Experiment/
|   |   |--- ExperimentRunner.py           # Experiment framework
|   |   |--- Visualization.py              # Visualization tools
|   |--- utils/
|   |   |--- utilities.py                  # Utility functions
|   |--- AbstractAlgorithm.py              # Abstract optimization class
|   |--- __init__.py                       # Package initialization
|--- main.py                               # Command-line script for experiments
|--- plot_results.py                       # Command-line script for visualization
|--- requirements.txt                      # Package dependencies
|--- LICENSE                               # MIT License
|--- README.md                             # This file
```

## Installation

### Requirements

The code requires Python 3.10+ and the following packages:
- numpy==1.26.4
- torch==2.6.0
- pyDOE==0.3.8
- scikit-learn==1.6.1
- botorch==0.13.0
- gpytorch==1.14
- ioh==0.3.18
- iohinspector==0.0.3

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

Run experiments comparing Vanilla BO and PCA-BO:

```bash
python main.py
```

Command-line options:
```
--dimensions      : Problem dimensions to test [default: 10 20 40]
--functions       : BBOB function IDs to test [default: 15 16 17]
--runs            : Number of independent runs [default: 30]
--budget_factor   : Budget factor (budget = budget_factor * dim + 50) [default: 10]
--doe_factor      : Initial design size factor (n_doe = doe_factor * dim) [default: 3.0]
--experiment_dir  : Output directory [default: pca-bo-experiment]
--acquisition     : Acquisition function [default: expected_improvement]
--var_threshold   : PCA variance threshold [default: 0.95]
--verbose         : Enable detailed output
```

Example:
```bash
python main.py --dimensions 5 10 20 --functions 15 21 --runs 10 --verbose
```

### Visualizing Results

Analyze and visualize experiment results:

```bash
python plot_results.py
```

Command-line options:
```
--experiment_dir  : Directory containing experiment data [default: pca-bo-experiment]
--output_dir      : Directory for visualization outputs [default: experiment_dir/visualizations]
--dimensions      : Dimensions to analyze [default: all]
--functions       : Function IDs to analyze [default: all]
--no_save         : Don't save visualization files (display only)
--format          : Output file format (png, pdf, svg) [default: png]
--dpi             : DPI for raster outputs [default: 300]
```

Example:
```bash
python plot_results.py --experiment_dir pca-bo-experiment --functions 15 16 --format pdf
```

## Key Components

### PCA-BO Algorithm

The PCA-BO algorithm works through the following steps:

1. Initial sampling using Latin Hypercube Sampling (LHS)
2. Ranking-based weighting scheme for evaluated points
3. Weighted PCA to identify important dimensions
4. Dimensionality reduction maintaining specified variance
5. GPR modeling in reduced space
6. Acquisition function optimization in reduced space
7. Inverse mapping to original space for new evaluations
8. Model updating with new data

### Benchmark Problems

The implementation uses the BBOB benchmark suite via the IOH framework, with a focus on:
- Multi-modal functions with adequate global structure (F15-F19)
- Multi-modal functions with weak global structure (F20-F24)

The experimental setup follows methodologies from recent research in high-dimensional Bayesian Optimization.

## Development

This repository is part of ongoing research on high-dimensional Bayesian Optimization techniques. The codebase is designed to support additional algorithmic variants, benchmark functions, and analytical tools.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
