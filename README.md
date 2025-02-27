# Basic Bayesian Optimization Methods with Bo-Torch 
This repository contains an archetype to build algorithms by using BO-Torch. This 


This code compares these approaches on the 24 functions of the Black-Box Optimization Benchmarking (BBOB) suite from the [COCO](https://arxiv.org/pdf/1603.08785.pdf) benchmarking environment using their definition from [IOHprofiler](https://iohprofiler.github.io/). It is based on the original repositories and modules of the selected algorithms [vanilla Bayesian Optimization](https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html), [CMA-ES](https://github.com/CMA-ES/pycma), [random search](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html), [SAASBO](https://github.com/martinjankowiak/saasbo), [RDUCB](https://github.com/huawei-noah/HEBO/tree/master/RDUCB), [PCA-BO](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO), [KPCA-BO](https://github.com/wangronin/Bayesian-Optimization/tree/KPCA-BO) and [TuRBO](https://github.com/uber-research/TuRBO). 

# Libraries and dependencies

The implementation is in `Python 3.10.12` and all the libraries used are listed in `requirements.txt`.

# Structure
- 

# Execution from source
## Dependencies to run from source

Running this code from source requires Python 3.10.12, and the libraries given in `requirements.txt` (Warning: preferably use a virtual environment for this specific project, to avoid breaking the dependencies of your other projects). In Ubuntu, installing the dependencies can be done using the following command:

```
pip install -r requirements.txt
```
