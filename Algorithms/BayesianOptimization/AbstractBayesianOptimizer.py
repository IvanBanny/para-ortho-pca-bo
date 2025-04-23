from ..AbstractAlgorithm import AbstractAlgorithm
from typing import Union, Optional, List
from pyDOE import lhs
import numpy as np
from abc import abstractmethod


class LHS_sampler:
    """Latin Hypercube Sampling implementation."""

    __default_criteria = ("center", "maximin", "centermaximin", "correlation")
    __reduced_criteria = {
        "c": "center",
        "m": "maximin",
        "cm": "centermaximin",
        "corr": "correlation"
    }

    def __init__(
            self,
            criterion: str = "correlation",
            iterations: int = 1000,
            sample_zero: bool = False
    ):
        """Initialize the LHS sampler with given parameters"""
        self.criterion = criterion
        self.iterations = iterations
        self.sample_zero = sample_zero

    def __call__(self, dim: int, n_samples: int) -> np.ndarray:
        """This '__call__' overload runs the LHS Experiment and returns a 'NumPy array'.

        Args:
            dim: Dimensionality of the samples
            n_samples: Number of samples to generate

        Returns:
            NumPy array of samples
        """
        points = lhs(
            n=dim,
            samples=n_samples,
            criterion=self.criterion,
            iterations=self.iterations
        )

        if self.sample_zero:
            points[0, :] = np.zeros_like(points[0, :])

        # Return the points transformed as a Tensor
        return points.reshape((n_samples, dim))

    @property
    def criterion(self) -> str:
        """Get the LHS criterion."""
        return self.__criterion

    @criterion.setter
    def criterion(self, new_criterion: str) -> None:
        """This property holder checks if the criterion is well-defined."""
        if not isinstance(new_criterion, str):
            raise ValueError("The new criterion is not a string!")

        else:
            # Lower case
            new_criterion = new_criterion.lower().strip()
            if new_criterion not in self.__default_criteria:
                # Check if the criterion is in the reduced criterion
                if new_criterion in self.__reduced_criteria:
                    self.__criterion = self.__reduced_criteria[new_criterion]
                else:
                    raise ValueError("The criterion is not matching the set ones!")
            else:
                self.__criterion = new_criterion

    @property
    def iterations(self) -> int:
        """Get the number of iterations."""
        return self.__iterations

    @iterations.setter
    def iterations(self, new_n_iter: int) -> None:
        """Set the number of iterations."""
        if new_n_iter > 0:
            self.__iterations = int(new_n_iter)
        else:
            raise ValueError("Negative iterations not allowed")

    @property
    def sample_zero(self) -> bool:
        """Property for sampling the zero vector."""
        return self.__sample_zero

    @sample_zero.setter
    def sample_zero(self, new_change: bool) -> None:
        """Set whether to include a zero sample."""
        try:
            bool(new_change)
        except Exception as e:
            print(e.args)

        # set the new value
        self.__sample_zero = new_change


class AbstractBayesianOptimizer(AbstractAlgorithm):
    """Abstract base class for Bayesian optimization algorithms."""

    def __init__(
            self,
            budget: int,
            n_DoE: Optional[int] = 0,
            **kwargs
    ):
        """Initialize the Bayesian optimizer."""
        # call the initializer from super class
        super().__init__(**kwargs)
        self.budget = budget
        self.n_DoE = n_DoE

        DoE_parameters = None
        for key, item in kwargs.items():
            if key.lower().strip() == "doe_parameters":
                DoE_parameters = item

        full_parameters = self.__build_LHS_parameters(DoE_parameters)

        # Check that there is some dictionary with the name "LHS_configuration"
        self.__lhs_sampler = LHS_sampler(
            criterion=full_parameters['criterion'],
            iterations=full_parameters['iterations'],
            sample_zero=full_parameters['sample_zero']
        )

        # Store all the evaluations (x,y)
        self.__x_evals = []
        self.__f_evals = []

    def __str__(self):
        pass

    def __call__(self, problem, dim: int, bounds: np.ndarray, **kwargs) -> None:
        """Execute the optimization algorithm."""
        # Call the superclass
        super().__call__(problem, dim, bounds, **kwargs)

        if not isinstance(self.n_DoE, int) or self.n_DoE == 0:
            # Assign this equal to the dimension of the problem
            self.n_DoE = self.dimension

        # Sample the points
        init_points = self.lhs_sampler(
            self.dimension,
            self.n_DoE
        )

        # Rescale the initial points
        init_points = self._rescale_lhs_points(init_points)
        # perform a loop with each point
        for _, point in enumerate(init_points):
            # append the new points
            self.__x_evals.append(point)
            self.__f_evals.append(problem(point))

        # Redefine the best
        self.assign_new_best()

        self.number_of_function_evaluations = self.n_DoE

        # Print best to screen if verbose
        if self.verbose:
            print(
                "After Initial sampling...",
                f"Current Best: x:{self.__x_evals[self.current_best_index]} y:{self.current_best}",
                flush=True
            )

    def _rescale_lhs_points(self, raw_lhs_points: np.ndarray):
        """Project the Latin Hypercube Samples into the raw space defined by each dimension.

        Args:
            raw_lhs_points ('np.ndarray'): A NumPy array with the initial samples coming
                                          from DoE (some points between 0 and 1)
        """
        # Take a copy of the raw points
        new_array = np.empty_like(raw_lhs_points)

        # Perform a loop all over the dimensionality of the points
        for dim in range(self.dimension):
            # Compute the multiplier
            multiplier = self.bounds[dim, 1] - self.bounds[dim, 0]
            new_array[:, dim] = multiplier * raw_lhs_points[:, dim] + self.bounds[dim, 0]

        return new_array

    def assign_new_best(self):
        """Assign the new best solution found so far."""
        if self.maximization:
            self.current_best = max(self.__f_evals)
        else:
            self.current_best = min(self.__f_evals)

        # Assign the index
        self.current_best_index = self.__f_evals.index(
            self.current_best,  # Value
            self.current_best_index  # Starting search position
        )

    def __repr__(self):
        return super(AbstractAlgorithm, self).__repr__()

    def __build_LHS_parameters(self, params_dict: Union[dict, None]) -> dict:
        """This function builds the LHS parameters to initialize the optimisation loop."""
        complete_params_dict = {
            "criterion": "center",
            "iterations": 1000,
            "sample_zero": False
        }

        if isinstance(params_dict, dict):
            for key, item in params_dict.items():
                complete_params_dict[key] = item

        return complete_params_dict

    def reset(self) -> None:
        """Reset the optimizer state."""
        # Call the superclass reset
        super().reset()

        # Reset the evaluations
        self.__x_evals = []
        self.__f_evals = []

    @property
    def budget(self) -> int:
        """Get the optimization budget."""
        return self.__budget

    @budget.setter
    def budget(self, new_budget: int) -> None:
        """Set the optimization budget."""
        assert new_budget > 0
        self.__budget = int(new_budget)

    @property
    def n_DoE(self) -> int:
        """Get the number of initial design of experiments samples."""
        return self.__n_DoE

    @n_DoE.setter
    def n_DoE(self, new_n_DOE: int) -> None:
        """Set the number of initial design of experiments samples."""
        self.__n_DoE = int(new_n_DOE) if new_n_DOE >= 0 else None

    @property
    def lhs_sampler(self) -> LHS_sampler:
        """Get the LHS sampler object."""
        return self.__lhs_sampler

    @property
    def f_evals(self) -> List[float]:
        """Get the function evaluation values."""
        return self.__f_evals

    @property
    def x_evals(self) -> List[np.ndarray]:
        """Get the evaluated points."""
        return self.__x_evals
