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

from Algorithms import Vanilla_BO, O_PCA_BO


@dataclass
class ExperimentConfig:
    algorithm: str
    acquisition_function: str
    dimensions: int
    problem_ids: List[int]
    instance: int
    budget: int
    n_doe: int
    batch_size: int
    random_seed: int
    doe_params: dict
    var_threshold: float
    n_components: int
    vis_output_dir: str


def run_single_experiment(pid: int, config: ExperimentConfig) -> None:
    """Run optimization for a single problem."""
    dump_path = os.path.join(os.getcwd(), "vis-logs")
    os.makedirs(dump_path, exist_ok=True)
    logger = Analyzer(
        triggers=[ALWAYS],
        root=dump_path,
        folder_name=f"{config.algorithm}-b{config.batch_size}-d{config.dimensions}-p{pid}-i{config.instance}",
        algorithm_name=config.algorithm,
        algorithm_info=f"A {config.algorithm}-BO Implementation.",
        additional_properties=[RAWYBEST],
        store_positions=True
    )

    match config.algorithm:
        case "vanilla":
            optimizer = Vanilla_BO(
                budget=config.budget,
                n_DoE=config.n_doe,
                q=config.batch_size,
                acquisition_function=config.acquisition_function,
                random_seed=config.random_seed,
                maximization=False,
                verbose=True,
                visualize=True,
                vis_output_dir=config.vis_output_dir,
                DoE_parameters=config.doe_params
            )
        case "pca" | "opca":
            optimizer = O_PCA_BO(
                budget=config.budget,
                n_DoE=config.n_doe,
                q=(config.batch_size if config.algorithm == "pca" else 1),
                ortho_samples=(0 if config.algorithm == "pca" else config.batch_size),
                n_components=config.n_components,
                var_threshold=config.var_threshold,
                acquisition_function=config.acquisition_function,
                random_seed=config.random_seed,
                maximization=False,
                verbose=True,
                visualize=True,
                vis_output_dir=config.vis_output_dir,
                # save_logs=True,
                # log_dir="vis-logs",
                DoE_parameters=config.doe_params
            )
        case _:
            raise ValueError(f"Invalid algorithm name: '{config.algorithm}'")

    problem = get_problem(
        pid,
        instance=config.instance,
        dimension=config.dimensions
    )
    problem.attach_logger(logger)

    optimizer(problem=problem)

    # print(f"Problem {pid} - Distance from optimum: {norm(problem.state.current_best.x-problem.optimum.x)}")
    # print(f"Problem {pid} - Regret: {problem.state.current_best.y - problem.optimum.y}")

    logger.close()


config = ExperimentConfig(
    algorithm="opca",  # vanilla / pca / opca
    acquisition_function="expected_improvement",
    # expected_improvement, probability_of_improvement
    dimensions=2,
    problem_ids=list(range(15, 25)),
    instance=0,
    budget=70,
    n_doe=20,
    batch_size=3,
    random_seed=69,
    doe_params={"criterion": "center", "iterations": 1000},
    n_components=1,
    var_threshold=0.95,
    vis_output_dir="./visualizations/visualizations-opca-new"
)

# Run all problems in parallel
if __name__ == "__main__":
    # Create partial function with config bound
    worker_func = functools.partial(run_single_experiment, config=config)

    # Use multiprocessing pool to run experiments in parallel
    with Pool() as pool:
        pool.map(worker_func, config.problem_ids)

    print(f"Completed optimization for all {len(config.problem_ids)} problems.")
