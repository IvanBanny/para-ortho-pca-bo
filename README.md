# Basic Bayesian Optimization Methods with Bo-Torch 
This repository contains an archetype to build algorithms by using BO-Torch. In this moment this repository uses the `Bo-Torch`, `GPyTorch` and `PyTorch` as main ones to develop the forthcoming methods. Nevertheless, this library just includes a directory archetype and a main function to test the built algorithm with the BBOB problems called from IOH-Experimenter interface (see: https://iohprofiler.github.io/IOHexperimenter/)

# Libraries and dependencies

The implementation is in Python 3.10.12 and all the libraries used are listed in `requirements.txt`.

# Structure
- `main.py` -> An archetype file, which is an example on how to call an instance of one of the 24 BBOB problems by using IOH interface and use the Vanilla-BO Algorithm stored in the repository.
- _/Algorithms_

# Execution from source
## Dependencies to run from source

Running this code from source requires Python 3.10.12, and the libraries given in `requirements.txt` (Warning: preferably use a virtual environment for this specific project, to avoid breaking the dependencies of your other projects). In Ubuntu, installing the dependencies can be done using the following command:

```
pip install -r requirements.txt
```
