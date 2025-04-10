"""
Experiment runner module for comparing Bayesian Optimization algorithms.
This module provides tools for running experiments comparing Vanilla BO and PCA-BO
on benchmark functions from the BBOB suite.
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
from tqdm.auto import tqdm

from ioh import get_problem
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.property import RAWYBEST, CURRENTY
from ioh.iohcpp.logger.trigger import ALWAYS

# Import the BO variants
from Algorithms import Vanilla_BO
from Algorithms import PCA_BO


class ExperimentRunner:
    """
    Class to run and manage experiments comparing Vanilla BO and PCA-BO algorithms.

    Attributes:
        dimensions (List[int]): List of problem dimensions to test.
        function_ids (List[int]): List of BBOB function IDs to test.
        num_runs (int): Number of runs per function and dimension combination.
        budget_factor (int): Factor to determine function evaluation budget (budget = budget_factor * dim + 50).
        doe_factor (float): Factor to determine initial design of experiments size (n_DoE = doe_factor * dim).
        root_dir (str): Root directory for experiment output.
        experiment_name (str): Name of the experiment folder.
        acquisition_function (str): Name of the acquisition function to use.
        pca_components (Optional[int]): Number of PCA components to use (0 for automatic).
        var_threshold (float): Variance threshold for PCA component selection.
    """

    def __init__(
        self,
        dimensions: List[int] = [10, 20, 40],
        function_ids: List[int] = [15, 16, 17],
        num_runs: int = 30,
        budget_factor: int = 10,
        doe_factor: float = 3.0,
        root_dir: str = os.getcwd(),
        experiment_name: str = "bo-experiment",
        acquisition_function: str = "expected_improvement",
        pca_components: Optional[int] = 0,
        var_threshold: float = 0.95,
        verbose: bool = False
    ):
        """
        Initialize the experiment runner with configuration parameters.

        Args:
            dimensions: List of problem dimensions to test.
            function_ids: List of BBOB function IDs to test.
            num_runs: Number of runs per function and dimension combination.
            budget_factor: Factor to determine function evaluation budget (budget = budget_factor * dim + 50).
            doe_factor: Factor to determine initial design of experiments size (n_DoE = doe_factor * dim).
            root_dir: Root directory for experiment output.
            experiment_name: Name of the experiment folder.
            acquisition_function: Name of the acquisition function to use.
            pca_components: Number of PCA components to use (0 for automatic).
            var_threshold: Variance threshold for PCA component selection.
            verbose: Whether to print detailed progress information.
        """
        self.dimensions = dimensions
        self.function_ids = function_ids
        self.num_runs = num_runs
        self.budget_factor = budget_factor
        self.doe_factor = doe_factor
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.acquisition_function = acquisition_function
        self.pca_components = pca_components
        self.var_threshold = var_threshold
        self.verbose = verbose

        # Create experiment directory if it doesn't exist
        self.experiment_dir = os.path.join(root_dir, experiment_name)
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

        # Additional tracking properties
        self.additional_properties = [
            RAWYBEST,   # Best-so-far value
            CURRENTY    # Current objective value
        ]

    def _create_logger(self, algorithm_variant: str, run_id: int) -> Analyzer:
        """
        Create a logger for a specific algorithm variant and run.

        Args:
            algorithm_variant: Name of the algorithm variant ('vanilla' or 'pca').
            run_id: Run identifier.

        Returns:
            An initialized IOH Analyzer logger.
        """
        logger = Analyzer(
            triggers=[ALWAYS],
            root=self.experiment_dir,
            folder_name=f"{algorithm_variant}_run_{run_id}",
            algorithm_name=f"{algorithm_variant.upper()} BO",
            algorithm_info=f"Run {run_id} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            additional_properties=self.additional_properties,
            store_positions=True  # Store x-variables in the logged files
        )
        return logger

    def run_experiment(self) -> None:
        """
        Run the complete experiment comparing Vanilla-BO and PCA-BO across all
        specified dimensions, functions, and runs.
        """
        # Store execution time data
        execution_times = {
            'vanilla': [],
            'pca': []
        }

        # Calculate total number of experiments
        total_experiments = len(self.dimensions) * len(self.function_ids) * self.num_runs * 2  # *2 for both algorithms

        # Set up progress tracking
        print(f"Running {total_experiments} experiments (2 algorithms × {len(self.dimensions)} dimensions × "
              f"{len(self.function_ids)} functions × {self.num_runs} runs)")

        # Create nested progress bars
        with tqdm(total=total_experiments, desc="Experiment Progress") as pbar:
            # Run all experiments
            for dim in self.dimensions:
                for func_id in self.function_ids:
                    for run_id in range(self.num_runs):
                        # Use the same random seed and problem instance for both algorithms in each run
                        random_seed = 1000 * func_id + 10 * dim + run_id
                        problem_instance = (11 * func_id + 7 * dim + run_id) % 15 + 1

                        # Calculate budget and DoE size
                        budget = self.budget_factor * dim + 50
                        n_doe = int(self.doe_factor * dim)

                        # Run both algorithm variants with the same seed and problem instance

                        # Update progress bar description
                        pbar.set_description(f"Vanilla BO [dim={dim}, func={func_id}, run={run_id}]")

                        # Run Vanilla BO
                        vanilla_time = self._run_single_experiment(
                            'vanilla', dim, func_id, run_id, random_seed,
                            budget, n_doe, problem_instance
                        )
                        execution_times['vanilla'].append({
                            'dimension': dim,
                            'function_id': func_id,
                            'run_id': run_id,
                            'time': vanilla_time
                        })
                        pbar.update(1)

                        # Update progress bar description
                        pbar.set_description(f"PCA-BO [dim={dim}, func={func_id}, run={run_id}]")

                        # Run PCA-BO
                        pca_time = self._run_single_experiment(
                            'pca', dim, func_id, run_id, random_seed,
                            budget, n_doe, problem_instance
                        )
                        execution_times['pca'].append({
                            'dimension': dim,
                            'function_id': func_id,
                            'run_id': run_id,
                            'time': pca_time
                        })
                        pbar.update(1)
                        pbar.refresh()

        # Save execution time data
        self._save_execution_times(execution_times)

    def _run_single_experiment(
        self,
        variant: str,
        dim: int,
        func_id: int,
        run_id: int,
        random_seed: int,
        budget: int,
        n_doe: int,
        problem_instance: int
    ) -> float:
        """
        Run a single experiment for a specific algorithm variant, dimension, function, and run.

        Args:
            variant: Algorithm variant ('vanilla' or 'pca').
            dim: Problem dimension.
            func_id: BBOB function ID.
            run_id: Run identifier.
            random_seed: Random seed for reproducibility.
            budget: Function evaluation budget.
            n_doe: Size of initial design of experiments.
            problem_instance: BBOB problem instance ID.

        Returns:
            Execution time in seconds.
        """
        # Create logger
        logger = self._create_logger(variant, run_id)

        # Get problem
        problem = get_problem(
            func_id,
            instance=problem_instance,
            dimension=dim,
        )

        # Attach logger to problem
        problem.attach_logger(logger)

        # Create optimizer based on variant
        if variant == 'vanilla':
            optimizer = Vanilla_BO(
                budget=budget,
                n_DoE=n_doe,
                acquisition_function=self.acquisition_function,
                random_seed=random_seed,
                maximisation=False,
                verbose=False,  # Always disable optimizer verbose output to avoid interfering with progress bars
            )
        else:  # PCA variant
            optimizer = PCA_BO(
                budget=budget,
                n_DoE=n_doe,
                n_components=self.pca_components,
                var_threshold=self.var_threshold,
                acquisition_function=self.acquisition_function,
                random_seed=random_seed,
                maximisation=False,
                verbose=False,  # Always disable optimizer verbose output to avoid interfering with progress bars
            )

        # Track algorithm parameters
        logger.watch(optimizer, ["acquisition_function_name"])
        if variant == 'pca':
            logger.watch(optimizer, ["explained_variance_ratio"])

        # Track execution time
        start_time = time.time()

        # Run optimization with progress tracking
        if self.verbose:
            # For tracking progress, we need to monitor the number of evaluations
            with tqdm(total=1, desc=f"Running optimization", leave=False) as opt_pbar:
                # Run optimization
                optimizer(problem=problem)
                opt_pbar.update(1)
        else:
            # Run optimization without progress tracking
            optimizer(problem=problem)

        # Calculate execution time
        execution_time = time.time() - start_time

        # # Calculate regret and log metrics
        # distance_from_optimum = np.linalg.norm(
        #     problem.state.current_best.x - problem.optimum.x
        # )
        # regret = problem.state.current_best.y - problem.optimum.y
        #
        # if self.verbose:
        #     print(f"{variant.upper()} BO on f{func_id} dim={dim} run={run_id}:")
        #     print(f"  Distance from optimum: {distance_from_optimum}")
        #     print(f"  Regret: {regret}")

        # Reset problem
        problem.reset()

        # Close logger
        logger.close()

        return execution_time

    def _save_execution_times(self, execution_times: Dict[str, List[Dict[str, Any]]]) -> None:
        """
        Save execution time data to CSV files.

        Args:
            execution_times: Dictionary containing execution time data for each algorithm variant.
        """
        for variant, times in execution_times.items():
            df = pd.DataFrame(times)

            # Add timestamp
            df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Save to CSV
            output_path = os.path.join(self.experiment_dir, f"{variant}_execution_times.csv")
            df.to_csv(output_path, index=False)
