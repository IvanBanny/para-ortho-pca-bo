"""This is a handle to deal with the algorithm class to implement with Modular Problems."""

__author__ = ["Iván Olarte Rodríguez"]

from typing import Union, Callable, List, Tuple, Dict, Optional
from abc import ABC, abstractmethod
from ioh.iohcpp.problem import RealSingleObjective, BBOB
from ioh.iohcpp import MAX
from ioh.iohcpp import RealBounds
from math import inf
from collections import defaultdict
import numpy as np
import torch

import warnings
from botorch.models.utils.assorted import InputDataWarning

warnings.filterwarnings("ignore", category=InputDataWarning)


class AbstractAlgorithm(ABC):
    TIME_PROFILES = []

    @abstractmethod
    def __init__(self, **kwargs):
        """This is the constructor for any optimisation algorithm used within this framework.

        In this framework, the initializer might receive two keywords related to the
        maximization/minimisation behavior and the verbosity.
        """
        # Initialize the number of function evaluations
        self.__number_of_function_evaluations = 0
        
        # Set initial default variables
        verbose_init = kwargs.pop("verbose", False)
        maximization_init = kwargs.pop("maximization", False)

        # Initialise the bounds as None type
        self.__bounds = np.empty(shape=(1, 2))
            
        # Assign the variables as members of the class
        self.verbose = verbose_init
        self.__maximization = maximization_init
        self.__current_best_index = 0

        # Initialize a storage variable for the best sample
        if self.__maximization:
            self.__current_best = -inf 
        else:
            self.__current_best = inf

        self.__problem = None
        self.__dimension = None

        self.__random_states = None
        self.random_seed = None

        # If a pbar is passed - redirect writing output to it and update it every evaluation
        self._pbar = kwargs.get("pbar")

        # Execution time logging
        self.timing_logs = defaultdict(list)
        for time_profile in self.TIME_PROFILES:
            self.timing_logs[time_profile] = []
    
    @abstractmethod
    def __call__(self,
                 problem: Union[RealSingleObjective, BBOB, Callable],
                 dim: Optional[int], 
                 bounds: Optional[np.ndarray],
                 **kwargs):
        """This is a default function indicating the structure of the '_call_' method in the context of an algorithm."""
        if (isinstance(problem, BBOB) or isinstance(problem, RealSingleObjective) 
                or issubclass(type(problem), RealSingleObjective)):
            
            # Assign the problem
            self.__problem = problem
            # Get the dimensionality from this
            self.dimension = problem.meta_data.n_variables
            self.maximization = (problem.meta_data.optimization_type == MAX)  # This is a placeholder to modify

            # Pass the bounds from the IOH definition (the setter function will adapt the input)
            self.bounds = problem.bounds
        elif isinstance(problem, Callable):
            # Assign the problem
            self.__problem = problem

            # Set the dimension to be given by the parameters
            self.dimension = dim
            
            # In case a new maximization default is given as a parameter
            self.maximization = kwargs.pop("maximization", False)

            analyzed_bounds = bounds
            if isinstance(analyzed_bounds, (np.ndarray, List[float], Tuple[float])):
                # Assign in this case
                self.bounds = analyzed_bounds
            else:
                raise AttributeError("The bounds for a callable problem were not found", name="bounds")
        else:
            raise AttributeError("The problem input is not well defined",
                                 name="problem",
                                 obj=problem)
    
    @abstractmethod
    def __str__(self):
        pass

    def __repr__(self):
        return super().__repr__()
    
    @abstractmethod
    def reset(self):
        """Reset the algorithm state."""
        # Set the number of function evaluations to 0
        self.number_of_function_evaluations = 0

        if self.maximization:
            self.__current_best = -inf
        else:
            self.__current_best = inf
        
        self.__current_best_index = 0

        self.timing_logs = defaultdict(list)

    @property
    def time_profile_names(self) -> List[str]:
        """List all timing profile names logged so far."""
        return list(self.timing_logs.keys())

    @property
    def average_times(self) -> Dict[str, float]:
        """Calculate average execution time for each function."""
        return {func_name: sum(times) / len(times) for func_name, times in self.timing_logs.items() if times}

    @property
    def total_times(self) -> Dict[str, float]:
        """Calculate total time spent in each function."""
        return {func_name: sum(times) for func_name, times in self.timing_logs.items()}

    @property
    def number_of_function_evaluations(self) -> int:
        """This property handles the number of function evaluations."""
        return self.__number_of_function_evaluations
    
    @number_of_function_evaluations.setter 
    def number_of_function_evaluations(self, new_eval: int) -> None:
        """This is the setter for the number of function evaluations"""
        if isinstance(new_eval, int) and new_eval >= 0:
            self.__number_of_function_evaluations = new_eval
        else:
            raise ValueError("The number of function evaluations must be a positive integer")
    
    @property
    def verbose(self) -> bool:
        """Definition of verbosity of the algorithm."""
        return self.__verbose
    
    @verbose.setter
    def verbose(self, new_verbosity: bool) -> None:
        """Rewrite the verbosity parameter."""
        self.__verbose = bool(new_verbosity)

    @property 
    def dimension(self) -> Union[int, None]:
        """Get the dimension of the problem."""
        return self.__dimension

    @dimension.deleter
    def dimension(self) -> None:
        """Delete the dimension attribute."""
        del self.__dimension

    @dimension.setter
    def dimension(self, new_dimension: Union[int, None]) -> None:
        """Set the dimension of the problem."""
        if isinstance(new_dimension, int) and new_dimension > 0:
            self.__dimension = new_dimension
        elif new_dimension is None:
            self.__dimension = new_dimension
        else:
            raise ValueError("The new dimension is oddly set")
    
    @property
    def current_best(self) -> float:
        """Get the current best solution value."""
        return self.__current_best
    
    @current_best.setter
    def current_best(self, new_current_best: float):
        """Set the current best solution value."""
        if self.__maximization:
            if new_current_best >= self.__current_best:
                # Assign this new value
                self.__current_best = new_current_best
            else:
                raise ValueError("The assignment is incorrect")
        else:
            if new_current_best <= self.__current_best:
                # Assign this new value
                self.__current_best = new_current_best
            else:
                raise ValueError("The assignment is incorrect")
    
    @property 
    def current_best_index(self) -> int:
        """Get the index of the current best solution."""
        return self.__current_best_index
    
    @current_best_index.setter
    def current_best_index(self, new_current_best_index: int) -> None:
        """Set the index of the current best solution."""
        if isinstance(new_current_best_index, int) and new_current_best_index >= self.__current_best_index:
            # Assign in this case
            self.__current_best_index = new_current_best_index
        else:
            raise ValueError("Something is wrong with this assignment")
    
    @property 
    def maximization(self) -> bool:
        """Get whether the problem is a maximization problem."""
        return self.__maximization
    
    @maximization.setter
    def maximization(self, new_definition: bool) -> None:
        """Set whether the problem is a maximization problem."""
        # Assign the change in variable
        if self.__maximization != bool(new_definition):
            self.__maximization = bool(new_definition)
            # Change given this condition
            if self.__maximization:
                self.__current_best = -inf
            else:
                self.__current_best = inf

    @property
    def bounds(self) -> np.ndarray:
        """Bounds property 'getter'. This just a repeater from the token in memory."""
        return self.__bounds
    
    @bounds.setter
    def bounds(self, new_bounds: Union[np.ndarray, RealBounds, List[float], Tuple[float]]):
        """This is the bounds property setter."""
        # The first case is if the bounds come from an IOH defined problem
        if isinstance(new_bounds, RealBounds):
            lower_bounds = new_bounds.lb
            upper_bounds = new_bounds.ub
            
            if isinstance(lower_bounds, float):
                # this is in case the value is repeated per every dimension
                lower_bounds_arr = np.ones(shape=(self.dimension, 1)) * lower_bounds
                upper_bounds_arr = np.ones(shape=(self.dimension, 1)) * upper_bounds

                # Reshape the array
                self.__bounds = np.hstack(tup=(lower_bounds_arr, upper_bounds_arr))
            
            elif isinstance(lower_bounds, np.ndarray):
                if lower_bounds.size == 1:
                    # this is in case the value is repeated per every dimension
                    lower_bounds_arr = np.tile(lower_bounds, reps=self.dimension).reshape((self.dimension, 1))
                    upper_bounds_arr = np.tile(upper_bounds, reps=self.dimension).reshape((self.dimension, 1))
                else:
                    lower_bounds_arr = lower_bounds.reshape((self.dimension, 1))
                    upper_bounds_arr = upper_bounds.reshape((self.dimension, 1))

                # Reshape the array
                self.__bounds = np.hstack(tup=(lower_bounds_arr, upper_bounds_arr))
        else:
            try:
                new_bounds = np.array(new_bounds)
            except Exception as e:
                print("Error has occurred", e.args, flush=True)

            if new_bounds.size == 2:
                # This is the case all the bounds are the same
                new_bounds = new_bounds.ravel()

                lower_bounds = new_bounds[0]
                upper_bounds = new_bounds[1]

                lower_bounds_arr = np.ones(shape=(self.dimension, 1)) * lower_bounds
                upper_bounds_arr = np.ones(shape=(self.dimension, 1)) * upper_bounds

                # Reshape the array
                self.__bounds = np.hstack(tup=(lower_bounds_arr, upper_bounds_arr))
            
            elif np.remainder(new_bounds.size, 2) == 0 and new_bounds.size > 2:
                new_bounds = new_bounds.reshape((-1, 2))
                lower_bounds_arr = new_bounds[:, 0].reshape((-1, 1))
                upper_bounds_arr = new_bounds[:, 1].reshape((-1, 1))

                # Reshape the array
                self.__bounds = np.hstack(tup=(lower_bounds_arr, upper_bounds_arr))
            
            else:
                raise AttributeError("The bounds should be a given in pairs", name="bounds")

    @property
    def random_seed(self) -> int:
        """Get the random seed."""
        return self.__random_seed

    @random_seed.setter
    def random_seed(self, new_seed: int) -> None:
        """Set the random seed."""
        if isinstance(new_seed, int) and new_seed >= 0:
            self.__random_seed = new_seed

    def impose_random_seed(self) -> None:
        """Set random seeds for all randomness sources to ensure reproducibility.

        This method saves the current random states and then sets the random seed
        for numpy, torch CPU, and torch GPU if available.
        """
        # Save the current random states before setting a new seed
        self.save_random_states()

        # Set the numpy random seed
        np.random.seed(self.__random_seed)

        # Set PyTorch CPU seed
        torch.manual_seed(self.__random_seed)

        # Set PyTorch GPU seed if available
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.__random_seed)  # PyTorch GPU
            torch.cuda.manual_seed_all(self.__random_seed)  # Multi-GPU setups

    def save_random_states(self) -> None:
        """Save the current state of all random number generators as an attribute of self."""

        # Initialize the states dictionary, save numpy and PyTorch CPU state
        self.__random_states = {
            "numpy_state": np.random.get_state(),
            "torch_cpu_state": torch.get_rng_state()
        }

        # Save PyTorch GPU states if available
        if torch.cuda.is_available():
            self.__random_states["torch_gpu_states"] = [torch.cuda.get_rng_state(i) for i in
                                                        range(torch.cuda.device_count())]

    def restore_random_states(self) -> None:
        """Restore previously saved random number generator states from self.__random_states."""

        # Check if random states were previously saved
        if not hasattr(self, '__random_states'):
            raise "There are no saved random states to restore"

        # Restore numpy state
        if "numpy_state" not in self.__random_states:
            raise "There is not numpy_state saved in __random_states"

        np.random.set_state(self.__random_states["numpy_state"])

        # Restore PyTorch CPU state
        if "torch_cpu_state" not in self.__random_states:
            raise "There is not torch_cpu_state saved in __random_states"

        torch.set_rng_state(self.__random_states["torch_cpu_state"])

        # Restore PyTorch GPU states if available
        if torch.cuda.is_available() and "torch_gpu_states" in self.__random_states:
            for i, state in enumerate(self.__random_states["torch_gpu_states"]):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(state, i)

        # it does not make sense to re-restore these
        del self.__random_states

    def compute_space_volume(self) -> float:
        """This function computes the volume of the space defined by the bounds."""
        # Compute the volume of the space
        return np.prod(self.bounds[:, 1] - self.bounds[:, 0])
