"""A simple script for quick BO implementation testing and debugging with parallel execution."""

import os
from dataclasses import dataclass
from typing import List
from multiprocessing import Pool
import functools

from ioh import get_problem
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.property import RAWYBEST
from ioh.iohcpp.logger.trigger import ALWAYS

from Algorithms import Vanilla_BO, PCA_BO


@dataclass
class ExperimentConfig:
    algorithm_variant: str
    acquisition_function: str
    dimensions: int
    problem_ids: List[int]
    instance: int
    budget: int
    n_doe: int
    random_seed: int
    doe_params: dict
    var_threshold: float
    n_components: int


def run_single_experiment(pid: int, config: ExperimentConfig) -> None:
    """Run optimization for a single problem."""
    # Use different random seed for each process to ensure independence
    process_seed = config.random_seed + pid

    logger = Analyzer(
        triggers=[ALWAYS],
        root=os.path.join(os.getcwd(), "ioh-logs"),
        folder_name="test",
        algorithm_name=config.algorithm_variant,
        algorithm_info=f"A {config.algorithm_variant}-BO Implementation.",
        additional_properties=[RAWYBEST],
        store_positions=True
    )

    if config.algorithm_variant == 'vanilla':
        optimizer = Vanilla_BO(
            budget=config.budget,
            n_DoE=config.n_doe,
            acquisition_function=config.acquisition_function,
            random_seed=process_seed,
            maximization=False,
            verbose=True,
            DoE_parameters=config.doe_params
        )
    else:
        optimizer = PCA_BO(
            budget=config.budget,
            n_DoE=config.n_doe,
            n_components=config.n_components,
            var_threshold=config.var_threshold,
            acquisition_function=config.acquisition_function,
            random_seed=process_seed,
            maximization=False,
            verbose=True,
            visualize=True,
            save_logs=True,
            DoE_parameters=config.doe_params
        )

    problem = get_problem(
        pid,
        instance=config.instance,
        dimension=config.dimensions
    )
    problem.attach_logger(logger)

    optimizer(problem=problem)

    # Uncomment if you need these metrics
    # print(f"Problem {pid} - Distance from optimum: {norm(problem.state.current_best.x-problem.optimum.x)}")
    # print(f"Problem {pid} - Regret: {problem.state.current_best.y - problem.optimum.y}")

    logger.close()


config = ExperimentConfig(
    algorithm_variant="pca",  # vanilla / pca
    acquisition_function="expected_improvement",
    # expected_improvement, probability_of_improvement, upper_confidence_bound
    dimensions=2,
    problem_ids=list(range(15, 25)),
    instance=0,
    budget=50,
    n_doe=20,
    random_seed=69,
    doe_params={"criterion": "center", "iterations": 1000},
    n_components=1,
    var_threshold=0.95
)

# Run all problems in parallel
if __name__ == "__main__":
    # Create partial function with config bound
    worker_func = functools.partial(run_single_experiment, config=config)

    # Use multiprocessing pool to run experiments in parallel
    with Pool() as pool:
        pool.map(worker_func, config.problem_ids)

    print(f"Completed optimization for all {len(config.problem_ids)} problems.")
