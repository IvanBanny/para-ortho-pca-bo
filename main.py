import os
from pathlib import Path
import numpy as np
from numpy.linalg import norm

from ioh import get_problem  # Function to set up problems from BBOB
from ioh.iohcpp.logger import Analyzer
from ioh.iohcpp.logger.property import RAWYBEST
from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT, ALWAYS  # Triggers defining how often to log results

from Algorithms import Vanilla_BO


# Logger setup

# These are the triggers to set a form
# how to log your data
triggers = [
    Each(10),  # Log after (10) evaluations
    ON_IMPROVEMENT  # Log when there's an improvement
]

logger = Analyzer(
    triggers=triggers,
    root=os.getcwd(),  # Store data in the current working directory
    folder_name="my-experiment",  # in a folder named: 'my-experiment'
    algorithm_name="Vanilla BO",  # meta-data for the algorithm used to generate these results
    algorithm_info="Bo-Torch Implementation",  # Some meta-data about the algorithm used (for reference)
    additional_properties=[RAWYBEST],  # Use this to log the best-so-far
    store_positions=True  # store x-variables in the logged files
)

# this automatically creates a folder 'my-experiment' in the current working directory
# if the folder already exists, it will given an additional number to make the name unique.

# In order to log data for a problem, we only have to attach it to a logger
problem = get_problem(
    7,  # An integer denoting one of the 24 BBOB problem
    instance=1,  # An instance, meaning the optimum of the problem is changed via some transformations
    dimension=5,  # The problem's dimension
)

problem.attach_logger(logger)


acquisition_function = "expected_improvement"

# Set up the Vanilla BO
optimizer = Vanilla_BO(
    budget=min(200, 50*problem.meta_data.n_variables),
    n_DoE=3*problem.meta_data.n_variables,
    acquisition_function=acquisition_function,  # "probability_of_improvement", "upper_confidence_bound"
    random_seed=45,
    maximisation=False,
    verbose=True,  # Print the best result-so-far
    DoE_parameters={'criterion': "center", 'iterations': 1000}
)

logger.watch(optimizer, acquisition_function)

# Run the optimization loop
optimizer(problem=problem, bounds=np.ones(problem.meta_data.n_variables))

# Compare the distance from optimum and regret of the optimizer at the end
print("The distance from optimum is: ", norm(problem.state.current_best.x-problem.optimum.x))
print("The regret is: ", problem.state.current_best.y - problem.optimum.y )

# Close the logger
logger.close()
