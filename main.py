import os
from pathlib import Path
import numpy as np
from numpy.linalg import norm

from ioh import get_problem  # Function to set up problems from BBOB
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.property import RAWYBEST
from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT, ALWAYS  # Triggers defining how often to log results

# Choose your BO variant here: "vanilla" or "pca"
bo_variant = "pca"

# import the BO variants
from Algorithms import Vanilla_BO
from Algorithms import PCA_BO


# Logger setup

# These are the triggers to set a form
# how to log your data
triggers = [
    Each(10),  # Log after (10) evaluations
    ON_IMPROVEMENT  # Log when there's an improvement
]


# this automatically creates a folder 'my-experiment' in the current working directory
# if the folder already exists, it will given an additional number to make the name unique.


for i in range(10):
    logger = Analyzer(
        triggers=triggers,
        root=os.getcwd(),  # Store data in the current working directory
        folder_name="my-experiment",  # in a folder named: 'my-experiment'
        #algorithm_name="Vanilla BO" + str(i),  # meta-data for the algorithm used to generate these results
        algorithm_name=f"{bo_variant.upper()} BO {i}",
        algorithm_info="Bo-Torch Implementation",  # Some meta-data about the algorithm used (for reference)
        additional_properties=[RAWYBEST],  # Use this to log the best-so-far
        store_positions=True  # store x-variables in the logged files
    )
    for seed in range(3):
        for dimensions in [5, 10, 15]:
            # In order to log data for a problem, we only have to attach it to a logger
            problem = get_problem(
                21,  # An integer denoting one of the 24 BBOB problem
                instance=42-i,  # An instance, meaning the optimum of the problem is changed via some transformations
                dimension=dimensions,  # The problem's dimension
            )

problem.attach_logger(logger)


            # Set up the Vanilla BO or PCA_BO
            budget = min(200, 50 * problem.meta_data.n_variables)
            n_DoE = 3 * problem.meta_data.n_variables

            if bo_variant == "vanilla":
                optimizer = Vanilla_BO(
                    budget=budget,
                    n_DoE=n_DoE,
                    acquisition_function="expected_improvement",
                    random_seed=seed,
                    maximisation=False,
                    verbose=True,
                    DoE_parameters={'criterion': "center", 'iterations': 1000}
                )
            elif bo_variant == "pca":
                optimizer = PCA_BO(
                    budget=budget,
                    n_DoE=n_DoE,
                    n_components=5,  # You can tune this as needed
                    acquisition_function="expected_improvement",
                    random_seed=seed,
                    verbose=True
                )
            else:
                raise ValueError(f"Unknown BO variant: {bo_variant}")

            logger.watch(optimizer, "acquistion_function_name")

            # Run the optimization loop
            optimizer(problem=problem)

            # Compare the distance from optimum and regret of the optimizer at the end
            print("The distance from optimum is: ", norm(problem.state.current_best.x-problem.optimum.x))
            print("The regret is: ", problem.state.current_best.y - problem.optimum.y )

    # Close the logger
    logger.close()
