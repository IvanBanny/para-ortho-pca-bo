"""A simple script for quick BO implementation testing and debugging."""

import os
from numpy.linalg import norm
from dataclasses import dataclass

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
    problem_id: int
    instance: int
    budget: int
    n_doe: int
    ortho_samples: int
    random_seed: int
    doe_params: dict
    var_threshold: float
    n_components: int


config = ExperimentConfig(
    algorithm_variant="pca",  # vanilla / pca
    acquisition_function="expected_improvement",
    # expected_improvement, probability_of_improvement, upper_confidence_bound
    dimensions=2,
    problem_id=20,
    instance=0,
    budget=17,
    n_doe=15,
    ortho_samples=2,
    random_seed=69,
    doe_params={"criterion": "center", "iterations": 1000},
    n_components=1,
    var_threshold=0.95
)

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
        random_seed=config.random_seed,
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
        ortho_samples=config.ortho_samples,
        acquisition_function=config.acquisition_function,
        random_seed=config.random_seed,
        maximization=False,
        verbose=True,
        visualize=True,
        save_logs=True,
        DoE_parameters=config.doe_params
    )

problem = get_problem(
    config.problem_id,
    instance=config.instance,
    dimension=config.dimensions
)
problem.attach_logger(logger)

print(f"\nRunning problem {config.problem_id} instance {config.instance} seed {config.random_seed}:\n")

optimizer(problem=problem)

# print("The distance from optimum is: ", norm(problem.state.current_best.x-problem.optimum.x))
# print("The regret is: ", problem.state.current_best.y - problem.optimum.y)

logger.close()
