"""A simple script for quick BO implementation testing and debugging."""

import os
from numpy.linalg import norm
from dataclasses import dataclass

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
    problem_id: int
    instance: int
    budget: int
    n_doe: int
    batch_size: int
    random_seed: int
    doe_params: dict
    var_threshold: float
    n_components: int


config = ExperimentConfig(
    algorithm="opca",  # vanilla / pca / opca
    acquisition_function="expected_improvement",
    # expected_improvement, probability_of_improvement, upper_confidence_bound
    dimensions=10,
    problem_id=20,
    instance=0,
    budget=40,
    n_doe=20,
    batch_size=3,
    random_seed=69,
    doe_params={"criterion": "center", "iterations": 1000},
    n_components=0,
    var_threshold=0.95
)

dump_path = os.path.join(os.getcwd(), "example-logs")
os.makedirs(dump_path, exist_ok=True)
logger = Analyzer(
    triggers=[ALWAYS],
    root=dump_path,
    folder_name=f"{config.algorithm}-b{config.batch_size}-d{config.dimensions}-"
                f"p{config.problem_id}-i{config.instance}",
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
            # visualize=True,
            # vis_output_dir="./test",
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
            # visualize=True,
            # vis_output_dir="./test",
            # save_logs=True,
            # log_dir="example-logs",
            DoE_parameters=config.doe_params
        )
    case _:
        raise ValueError(f"Invalid algorithm name: '{config.algorithm}'")

problem = get_problem(
    config.problem_id,
    instance=config.instance,
    dimension=config.dimensions
)
problem.attach_logger(logger)

print(f"\nRunning problem {config.problem_id} instance {config.instance} seed {config.random_seed}:\n")

optimizer(problem=problem)

print("The distance from optimum is: ", norm(problem.state.current_best.x-problem.optimum.x))
print("The regret is: ", problem.state.current_best.y - problem.optimum.y)

logger.close()
