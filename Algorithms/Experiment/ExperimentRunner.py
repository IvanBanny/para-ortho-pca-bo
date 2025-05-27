"""Experiment runner module for comparing Bayesian Optimization algorithms.

This module provides tools for running experiments comparing Vanilla BO, PCA-BO, and O-PCA-BO
on benchmark problems from the BBOB suite.
"""

import contextlib
from typing import List, Optional, Dict, Any
import os
from time import perf_counter
from tqdm.auto import tqdm
from numpy.linalg import norm
import joblib
from joblib import Parallel, delayed

from ioh.iohcpp.suite import BBOB
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.property import RAWYBEST
from ioh.iohcpp.logger.trigger import ALWAYS

# Import BO algorithms
from Algorithms import Vanilla_BO
from Algorithms import O_PCA_BO


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class ExperimentRunner:
    """Class to run and manage experiments comparing Vanilla BO, PCA-BO, and O-PCA-BO algorithms."""

    def __init__(
        self,
        algorithms: List[str],
        batch_sizes: List[int],
        dimensions: List[int],
        problem_ids: List[int],
        instances: Optional[List[int]] = None,
        num_runs: Optional[int] = None,
        budget_factor: int = 10,
        doe_factor: float = 3.0,
        random_seed: int = 69,
        acquisition_function: str = "EI",
        var_threshold: float = 0.95,
        root_dir: str = os.getcwd(),
        experiment_name: str = "experiment",
        torch_config: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ):
        """Initialize the experiment runner with configuration parameters.

        Args:
            algorithms: List of algorithm codenames to test. (opca || pca || vanilla)
            batch_sizes: List of batch sizes (q * ortho_samples) to test.
            dimensions: List of problem dimensions to test.
            problem_ids: List of BBOB problems IDs to test.
            instances: List of instances to use.
            num_runs: Number of runs per problem and dimension combination.
            budget_factor: Factor to determine problem evaluation budget (budget = budget_factor * dim + 50).
            doe_factor: Factor to determine initial design of experiments size (n_DoE = doe_factor * dim).
            random_seed: Randomness seed.
            acquisition_function: Acquisition function name.
            var_threshold: PCA variance threshold.
            root_dir: Root directory for experiment output dir.
            experiment_name: Name of the experiment dir.
            torch_config: gpu configuration.
            verbose: Whether to print detailed progress information.
        """
        self.algorithms = algorithms
        self.batch_sizes = batch_sizes
        self.dimensions = dimensions
        self.problem_ids = problem_ids
        self.instances = instances
        self.num_runs = num_runs
        self.budget_factor = budget_factor
        self.doe_factor = doe_factor
        self.random_seed = random_seed
        self.acquisition_function = acquisition_function
        self.var_threshold = var_threshold
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.torch_config = torch_config
        self.verbose = verbose

        # Additional logger properties
        self.triggers = [ALWAYS]  # Log on every problem evaluation
        self.logger_properties = [RAWYBEST]  # Log best-so-far value

        self.pbar_cnt = 0

        if self.instances is None:
            if self.num_runs is None:
                raise ValueError("Either instances or num_runs must be provided")
            else:
                self.instances = range(self.num_runs)

        self.doe_params = {"criterion": "center", "iterations": 1000}

    def run_experiment(self, algorithm, batch_size, dim, pid, instance):
        # Get problem info
        suite = BBOB(problem_ids=[pid], dimensions=[dim], instances=[instance])
        problem = next(iter(suite))
        maximization = bool(problem.meta_data.optimization_type.value)
        budget = self.budget_factor * dim + 50
        n_doe = int(self.doe_factor * dim)

        with tqdm(total=budget, position=self.pbar_cnt, desc="", leave=False) as pbar:
            self.pbar_cnt += 1

            pbar.set_description(f"{algorithm} | b{batch_size} | d{dim} | f{pid} | i{instance}")

            if self.verbose:
                pbar.write(f"\nRunning {algorithm} | {batch_size}-batch | {dim}-dim | F-{pid} | i-{instance}:\n")

            # Setup logger
            dump_path = os.path.join(self.root_dir, self.experiment_name)
            os.makedirs(dump_path, exist_ok=True)
            logger = Analyzer(
                triggers=self.triggers,
                root=dump_path,
                folder_name=f"{algorithm}-b{batch_size}-d{dim}-p{pid}-i{instance}",
                algorithm_name=algorithm,
                algorithm_info=f"A {algorithm}-BO Implementation.",
                additional_properties=self.logger_properties,
                store_positions=True
            )

            # Add relevant shared experiment settings
            logger.set_experiment_attributes({
                "budget": f"{budget}",
                "doe": f"{n_doe}",
                "acquisition_function": f"{self.acquisition_function}",
                "random_seed": f"{self.random_seed}",
                "torch_config": f"{self.torch_config}",
                "self.doe_params": f"{self.doe_params}",
            })

            match algorithm:
                case "vanilla":
                    optimizer = Vanilla_BO(
                        budget=budget,
                        n_DoE=n_doe,
                        q=batch_size,
                        acquisition_function=self.acquisition_function,
                        random_seed=self.random_seed,
                        torch_config=self.torch_config,
                        maximization=maximization,
                        verbose=self.verbose,
                        DoE_parameters=self.doe_params,
                        pbar=pbar
                    )
                    logger.set_experiment_attributes({
                        "q": f"{batch_size}",
                    })
                case "pca" | "opca":
                    optimizer = O_PCA_BO(
                        budget=budget,
                        n_DoE=n_doe,
                        q=(batch_size if algorithm == "pca" else 1),
                        ortho_samples=(0 if algorithm == "pca" else batch_size),
                        var_threshold=self.var_threshold,
                        acquisition_function=self.acquisition_function,
                        random_seed=self.random_seed,
                        torch_config=self.torch_config,
                        maximization=maximization,
                        verbose=self.verbose,
                        DoE_parameters=self.doe_params,
                        pbar=pbar
                    )
                    logger.set_experiment_attributes({
                        **({"q": batch_size} if algorithm == "pca" else {}),
                        **({"q": 1, "ortho_samples": batch_size} if algorithm == "opca" else {}),
                        "var_threshold": f"{self.var_threshold}"
                    })
                case _:
                    raise ValueError(f"Invalid algorithm name: '{algorithm}'")

            # Add profile timings to the run before attaching the problem
            # ioh refuses to do it DURING the run OR in a loop
            # because it is, permanently, a teapot
            # seriously, I hate ioh.iohcpp.logger.Analyzer so much I spent like two full days on this.
            # I've tried everything, trust me. It just contradicts itself in profoundly impressive ways
            # I don't even know how could it possibly be written THIS bad
            for time_profile in getattr(optimizer, "TIME_PROFILES", []):
                logger.add_run_attribute(f"{time_profile}_time", 0.0)

            logger.add_run_attribute("time", 0.0)

            suite.attach_logger(logger)

            # Run the optimization
            start_time = perf_counter()
            optimizer(problem=problem)
            logger.set_run_attribute("time", perf_counter() - start_time)

            # Retrieve profiling data, Extract total function timings, Profit Operation
            for time_profile, total_profile_time in optimizer.total_times.items():
                logger.set_run_attribute(f"{time_profile}_time", total_profile_time)

            if self.verbose:
                pbar.write(f"The distance from optimum is: "
                           f"{norm(problem.state.current_best.x - problem.optimum.x)}")
                pbar.write(f"The regret is: {problem.state.current_best.y - problem.optimum.y}")

            pbar.close()
            self.pbar_cnt -= 1

        # Detach logger from the suite and close logger
        suite.detach_logger()
        logger.close()

    def __call__(self) -> None:
        """Runs the complete experiment comparing Vanilla-BO, PCA-BO, and O-PCA-BO.

        This function performs the experiment across all specified dimensions,
        functions, and runs.
        """
        # Calculate total number of experiments
        total_runs = (len(self.algorithms) * len(self.batch_sizes) *
                      len(self.problem_ids) * len(self.dimensions) * self.num_runs)

        results = []
        if total_runs == 0:
            print("No experiments to run!")
        if total_runs == 1:
            results = self.run_experiment(self.algorithms[0], self.batch_sizes[0], self.dimensions[0],
                                          self.problem_ids[0], self.instances[0])
        else:
            print(f"\nRunning {total_runs} experiments in parallel: ({len(self.algorithms)} algorithms × "
                  f"{len(self.batch_sizes)} batch sizes × {len(self.dimensions)} dimensions × "
                  f"{len(self.problem_ids)} problems × {self.num_runs} runs)\n")

            params_list = [{"algorithm": a, "batch_size": b, "dim": d, "pid": p, "instance": i}
                           for a in self.algorithms for b in self.batch_sizes for d in self.dimensions
                           for p in self.problem_ids for i in self.instances]

            with tqdm_joblib(tqdm(desc="Total Progress", total=total_runs)) as progress_bar:
                self.pbar_cnt += 1
                results = Parallel(n_jobs=-1, verbose=10)(
                    delayed(self.run_experiment)(**params)
                    for params in params_list
                )
                self.pbar_cnt -= 1

        print(results)
