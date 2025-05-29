"""Experiment runner module for comparing Bayesian Optimization algorithms.

This module provides tools for running experiments comparing Vanilla BO, PCA-BO, and O-PCA-BO
on benchmark problems from the BBOB suite.
"""

import contextlib
from typing import List, Optional, Dict, Any, Tuple
import os
import sys
import traceback
from time import perf_counter
# from tqdm.auto import tqdm
import joblib
from joblib import Parallel, delayed

from ioh import get_problem
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.property import RAWYBEST, CURRENTY, CURRENTBESTY
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
        gpr_p: float = 0.5,
        gpr_val_factor: float = 0.5,
        onorm_factor: float = 2.0,
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
            gpr_p (float, optional): Percentage of ranked points to use in GPR fitting. Range [0.3, 1]. Defaults to 0.5.
            gpr_val_factor (float, optional): Relative influence of value rank to distance rank
                                              in GPR fitting point selection. Range [0, 1]. Defaults to 0.5.
            onorm_factor (float, optional): O-norm sampling multiplier. Range [0, +inf].
                                            0 for uniform sampling. Defaults to 2.0.
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
        self.gpr_p = gpr_p
        self.gpr_val_factor = gpr_val_factor
        self.onorm_factor = onorm_factor
        self.root_dir = root_dir
        self.experiment_name = experiment_name
        self.torch_config = torch_config
        self.verbose = verbose

        # Additional logger properties
        self.triggers = [ALWAYS]  # Log on every problem evaluation
        self.logger_properties = [RAWYBEST, CURRENTY, CURRENTBESTY]

        if self.instances is None:
            if self.num_runs is None:
                raise ValueError("Either instances or num_runs must be provided")
            else:
                self.instances = range(self.num_runs)

        self.doe_params = {"criterion": "center", "iterations": 1000}

    def run_experiment(self, algorithm, batch_size, dim, pid, instance) -> Tuple[bool, Optional[str]]:
        """Run a single experiment with specified parameters.

        Args:
            algorithm: Algorithm name to use
            batch_size: Batch size for the algorithm
            dim: Problem dimension
            pid: Problem ID
            instance: Instance number

        Returns:
            Tuple of (success: bool, error_message: Optional[str])
        """
        run_id = f"{algorithm} | b{batch_size} | d{dim} | f{pid} | i{instance}"
        logger = None

        try:
            # Get problem info
            problem = get_problem(fid=pid, instance=instance, dimension=dim)
            maximization = bool(problem.meta_data.optimization_type.value)
            budget = self.budget_factor * dim + 50
            n_doe = int(self.doe_factor * dim)

            if self.verbose:
                print(f"\nRunning {run_id}:\n")

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
                "batch_size": f"{batch_size}",
                **({"var_threshold": f"{self.var_threshold}"} if algorithm in ["pca", "opca"] else {}),
                **({"gpr_p": f"{self.gpr_p}", "gpr_val_factor": f"{self.gpr_val_factor}",
                    "onorm_factor": f"{self.onorm_factor}"} if algorithm == "opca" else {}),
                "budget": f"{budget}",
                "doe": f"{n_doe}",
                "acquisition_function": f"{self.acquisition_function}",
                "random_seed": f"{self.random_seed}",
                "torch_config": f"dict{self.torch_config}",
                "doe_params": f"dict{self.doe_params}",
            })

            # Initialize optimizer based on algorithm type
            optimizer = self._create_optimizer(
                algorithm, batch_size, budget, n_doe, maximization
            )

            # Add profiling attributes to logger
            for time_profile in getattr(optimizer, "TIME_PROFILES", []):
                logger.add_run_attribute(f"{time_profile}_time", 0.0)
            logger.add_run_attribute("time", 0.0)

            problem.attach_logger(logger)

            # Run the optimization with error handling
            start_time = perf_counter()
            optimizer(problem=problem)
            total_time = perf_counter() - start_time

            # Record timing information
            logger.set_run_attribute("time", total_time)
            for time_profile, total_profile_time in optimizer.total_times.items():
                logger.set_run_attribute(f"{time_profile}_time", total_profile_time)

            return True, None

        except Exception as e:
            error_msg = f"Run failed ({run_id}): {str(e)}"
            print(error_msg, file=sys.stderr)

            # Print traceback for debugging if verbose
            if self.verbose:
                print(f"Traceback for {run_id}:", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

            return False, error_msg

        finally:
            if logger is not None:
                try:
                    logger.close()
                except Exception as cleanup_error:
                    print(f"Warning: Error closing logger for {run_id}: {cleanup_error}",
                          file=sys.stderr)

    def _create_optimizer(self, algorithm: str, batch_size: int, budget: int,
                          n_doe: int, maximization: bool):
        """Create optimizer instance based on algorithm type.

        Args:
            algorithm: Algorithm name
            batch_size: Batch size
            budget: Evaluation budget
            n_doe: Number of initial design points
            maximization: Whether this is a maximization problem

        Returns:
            Configured optimizer instance
        """
        common_params = {
            "budget": budget,
            "n_DoE": n_doe,
            "acquisition_function": self.acquisition_function,
            "random_seed": self.random_seed,
            "torch_config": self.torch_config,
            "maximization": maximization,
            "verbose": self.verbose,
            "DoE_parameters": self.doe_params
        }

        match algorithm:
            case "vanilla":
                return Vanilla_BO(q=batch_size, **common_params)

            case "pca":
                return O_PCA_BO(
                    q=batch_size,
                    ortho_samples=0,
                    var_threshold=self.var_threshold,
                    gpr_p=1.0,
                    gpr_val_factor=0.5,
                    **common_params
                )

            case "opca":
                return O_PCA_BO(
                    q=1,
                    ortho_samples=batch_size,
                    var_threshold=self.var_threshold,
                    gpr_p=self.gpr_p,
                    gpr_val_factor=self.gpr_val_factor,
                    onorm_factor=self.onorm_factor,
                    **common_params
                )

            case _:
                raise ValueError(f"Invalid algorithm name: '{algorithm}'")

    def __call__(self) -> Dict[str, Any]:
        """Runs the complete experiment comparing Vanilla-BO, PCA-BO, and O-PCA-BO.

        This function performs the experiment across all specified dimensions,
        functions, and runs.

        Returns:
            Dictionary containing experiment summary statistics
        """
        # Calculate total number of experiments
        total_runs = (len(self.algorithms) * len(self.batch_sizes) *
                      len(self.problem_ids) * len(self.dimensions) * len(self.instances))

        if total_runs == 0:
            print("No experiments to run!")
            return {"total_runs": 0, "successful_runs": 0, "failed_runs": 0}

        print(f"\nRunning {total_runs} experiments: ({len(self.algorithms)} algorithms × "
              f"{len(self.batch_sizes)} batch sizes × {len(self.dimensions)} dimensions × "
              f"{len(self.problem_ids)} problems × {len(self.instances)} runs)\n")

        params_list = [
            {"algorithm": a, "batch_size": b, "dim": d, "pid": p, "instance": i}
            for a in self.algorithms
            for b in self.batch_sizes
            for d in self.dimensions
            for p in self.problem_ids
            for i in self.instances
        ]

        if total_runs == 1:
            results = [self.run_experiment(**params_list[0])]
        else:
            # with tqdm_joblib(tqdm(desc="Total Progress", total=total_runs, position=0)) as progress_bar:
            results = Parallel(n_jobs=-1, verbose=10)(
                delayed(self.run_experiment)(**params)
                for params in params_list
            )

        # Calculate summary statistics
        successful_runs = sum(1 for success, _ in results if success)
        failed_runs = total_runs - successful_runs

        summary = {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0.0
        }

        print(f"\nExperiment completed:")
        print(f"  Total runs: {summary['total_runs']}")
        print(f"  Successful: {summary['successful_runs']}")
        print(f"  Failed: {summary['failed_runs']}")
        print(f"  Success rate: {summary['success_rate']:.2%}")

        if failed_runs > 0:
            print(f"\nWarning: {failed_runs} runs failed. Check stderr output for details.")

        return summary
