"""Experiment runner module for comparing Bayesian Optimization algorithms.

This module provides tools for running experiments comparing Vanilla BO and PCA-BO
on benchmark problems from the BBOB suite.
"""

from typing import List, Optional
import os
from time import time
from tqdm.auto import tqdm
from numpy.linalg import norm

from ioh.iohcpp.suite import BBOB
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.property import RAWYBEST
from ioh.iohcpp.logger.trigger import ALWAYS

# Import BO algorithms
from Algorithms import Vanilla_BO
from Algorithms import PCA_BO
from Algorithms.BayesianOptimization.LPCA_BO.LPCA_BO import LPCA_BO
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo_interface_class import CleanPCABOInterface


class ExperimentRunner:
    """Class to run and manage experiments comparing Vanilla BO, PCA-BO and LPCA-BO algorithms."""

    def __init__(
        self,
        algorithms: List[str],
        dimensions: List[int],
        problem_ids: List[int],
        num_runs: int = 30,
        budget_factor: int = 10,
        doe_factor: float = 3.0,
        root_dir: str = os.getcwd(),
        experiment_name: str = "experiment",
        acquisition_function: str = "expected_improvement",
        pca_components: Optional[int] = None,
        var_threshold: float = 0.95,
        verbose: bool = False
    ):
        """Initialize the experiment runner with configuration parameters.

        Args:
            algorithms: List of algorithm codenames to test. (pca || vanilla)
            dimensions: List of problem dimensions to test.
            problem_ids: List of BBOB problems IDs to test.
            num_runs: Number of runs per problem and dimension combination.
            budget_factor: Factor to determine problem evaluation budget (budget = budget_factor * dim + 50).
            doe_factor: Factor to determine initial design of experiments size (n_DoE = doe_factor * dim).
            root_dir: Root directory for experiment output dir.
            experiment_name: Name of the experiment dir.
            acquisition_function: Name of the acquisition function to use.
            pca_components: Number of PCA components to use (None for automatic).
            var_threshold: Variance threshold for PCA component selection.
            verbose: Whether to print detailed progress information.
        """
        self.algorithms = algorithms
        self.dimensions = dimensions
        self.problem_ids = problem_ids
        self.num_runs = num_runs
        self.budget_factor = budget_factor
        self.doe_factor = doe_factor
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.acquisition_function = acquisition_function
        self.pca_components = pca_components
        self.var_threshold = var_threshold
        self.verbose = verbose

        # Additional logger properties
        self.triggers = [ALWAYS]  # Log on every problem evaluation
        self.logger_properties = [RAWYBEST]  # Log best-so-far value

        self.instances = range(self.num_runs)
        self.doe_params = {"criterion": "center", "iterations": 1000}

    def run_experiment(self) -> None:
        """Runs the complete experiment comparing Vanilla-BO and PCA-BO.

        This function performs the experiment across all specified dimensions,
        functions, and runs.
        """

        # Calculate total number of experiments
        total_runs = len(self.algorithms) * len(self.problem_ids) * len(self.dimensions) * self.num_runs

        print(f"\nRunning {total_runs} experiments ({len(self.algorithms)} algorithms × "
              f"{len(self.dimensions)} dimensions × {len(self.problem_ids)} problems × {self.num_runs} runs)\n")

        suite = BBOB(problem_ids=self.problem_ids, dimensions=self.dimensions, instances=self.instances)

        with tqdm(total=total_runs, position=0, desc="Total Progress") as ebar:
            for algorithm in self.algorithms:
                logger = Analyzer(
                    triggers=self.triggers,
                    root=self.root_dir,
                    folder_name=f"{algorithm}-{self.experiment_name}",
                    algorithm_name=algorithm,
                    algorithm_info=f"A {algorithm}-BO Implementation.",
                    additional_properties=self.logger_properties,
                    store_positions=True
                )

                # Add relevant shared experiment settings
                logger.set_experiment_attributes({
                    "budget_factor": f"{self.budget_factor}",
                    "doe_factor": f"{self.doe_factor}",
                    "acquisition_function": f"{self.acquisition_function}"
                })

                match algorithm:
                    case "vanilla":
                        optimizer_class = Vanilla_BO
                    case "pca":
                        logger.set_experiment_attributes({
                            "pca_components": f"{self.pca_components}",
                            "var_threshold": f"{self.var_threshold}"
                        })

                        optimizer_class = CleanPCABOInterface
                    case "clean-pca":
                        logger.set_experiment_attributes({
                            "pca_components": f"{self.pca_components}",
                            "var_threshold": f"{self.var_threshold}"
                        })

                        optimizer_class = PCA_BO
                    case "LPCA_BO":
                        logger.set_experiment_attributes({
                            "pca_components": f"{self.pca_components}",
                            "var_threshold": f"{self.var_threshold}",
                        })
                        optimizer_class = LPCA_BO  # <-- The PCA-TuRBO class
                    case _:
                        raise ValueError(f"Invalid algorithm name: '{algorithm}'")

                # Add profile timings to the run before attaching the problem
                # ioh refuses to do it DURING the run OR in a loop
                # because it is, permanently, a teapot
                # seriously, I hate ioh.iohcpp.logger.Analyzer so much I spent like two full days on this.
                # I've tried everything, trust me. It just contradicts itself in profoundly impressive ways
                # I don't even know how could it possibly be written THIS bad
                for time_profile in getattr(optimizer_class, "TIME_PROFILES", []):
                    logger.add_run_attribute(f"{time_profile}_time", 0.0)

                logger.add_run_attribute("time", 0.0)

                suite.attach_logger(logger)

                for i, problem in enumerate(suite):
                    dim = problem.meta_data.n_variables
                    problem_id = problem.meta_data.problem_id
                    instance = problem.meta_data.instance
                    maximization = bool(problem.meta_data.optimization_type.value)
                    run_num = self.instances.index(instance)

                    budget = self.budget_factor * dim + 50
                    n_doe = int(self.doe_factor * dim)
                    random_seed = 1000 * problem_id + 10 * dim + instance

                    with tqdm(total=budget, position=1, desc="", leave=False) as pbar:
                        pbar.set_description(f"{algorithm} | {dim}-dim | F-{problem_id} | run-{run_num + 1}")

                        if self.verbose:
                            pbar.write(f"\nRunning {algorithm} | {dim}-dim | F-{problem_id} | run-{run_num + 1}:\n")

                        match algorithm:
                            case "vanilla":
                                optimizer = Vanilla_BO(
                                    budget=budget,
                                    n_DoE=n_doe,
                                    acquisition_function=self.acquisition_function,
                                    random_seed=random_seed,
                                    maximization=maximization,
                                    verbose=self.verbose,
                                    DoE_parameters=self.doe_params,
                                    pbar=pbar
                                )
                            case "pca":
                                optimizer = PCA_BO(
                                    budget=budget,
                                    n_DoE=n_doe,
                                    var_threshold=self.var_threshold,
                                    acquisition_function=self.acquisition_function,
                                    random_seed=random_seed,
                                    maximization=maximization,
                                    verbose=self.verbose,
                                    DoE_parameters=self.doe_params,
                                    pbar=pbar
                                )

                            case "clean-pca":
                                optimizer = CleanPCABOInterface(
                                    budget=budget,
                                    n_DoE=n_doe,
                                    var_threshold=self.var_threshold,
                                    acquisition_function=self.acquisition_function,
                                    random_seed=random_seed,
                                    maximization=maximization,
                                    verbose=self.verbose,
                                    DoE_parameters=self.doe_params,
                                    pbar=pbar
                                )

                            case "lpca_bo":
                                optimizer = LPCA_BO(
                                    budget=budget,
                                    n_DoE=n_doe,
                                    var_threshold=self.var_threshold,
                                    acquisition_function=self.acquisition_function,  # e.g. "pei"
                                    random_seed=random_seed,
                                    maximization=maximization,
                                    verbose=self.verbose,
                                    DoE_parameters=self.doe_params,
                                    pbar=pbar,
                                )
                            case _:
                                raise ValueError(f"Invalid algorithm name: '{algorithm}'")

                        # Run the optimization
                        start_time = time()
                        optimizer(problem=problem)
                        logger.set_run_attribute("time", time() - start_time)

                        # Retrieve profiling data, Extract total function timings, Profit Operation
                        for time_profile, total_profile_time in optimizer.total_times.items():
                            logger.set_run_attribute(f"{time_profile}_time", total_profile_time)

                        if self.verbose:
                            pbar.write(f"The distance from optimum is: "
                                       f"{norm(problem.state.current_best.x - problem.optimum.x)}")
                            pbar.write(f"The regret is: {problem.state.current_best.y - problem.optimum.y}")

                        pbar.close()
                        ebar.update(1)

                # Detach logger from the suite and close logger
                suite.detach_logger()
                logger.close()
