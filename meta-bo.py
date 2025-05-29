import os
import torch
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args

from Algorithms import ExperimentRunner
from Algorithms.utils.experiment_loss import get_loss

param_space = [
    Real(0.0, 1.0, name='gpr_p'),
    Real(0.0, 1.0, name='gpr_val_factor'),
    Categorical([False, True], name='use_onorm'),
    Real(1.0, 3.0, name='onorm_value')
]

eval_cnt = 0

@use_named_args(param_space)
def objective(gpr_p, gpr_val_factor, use_onorm, onorm_value):
    global eval_cnt
    eval_cnt += 1

    onorm_factor = 0.0 if not use_onorm else onorm_value

    print(f"Evaluating: gpr_p={gpr_p:.4f}, gpr_val_factor={gpr_val_factor:.4f}, "
          f"onorm_factor={onorm_factor:.4f}:\n")

    # Initialize experiment object
    experiment = ExperimentRunner(
        algorithms=["opca"],
        batch_sizes=[5],
        dimensions=[10, 20, 40],
        problem_ids=[15, 16, 17, 18, 20, 21, 22, 24],
        instances=None,
        num_runs=2,
        budget_factor=10,
        doe_factor=3,
        random_seed=69,
        acquisition_function="EI",
        var_threshold=0.95,
        gpr_p=gpr_p,
        gpr_val_factor=gpr_val_factor,
        onorm_factor=onorm_factor,
        root_dir=os.getcwd(),
        experiment_name=f"meta-bo/{eval_cnt}",
        torch_config={
            "device": torch.device("cpu"),
            "dtype": torch.double,
            "NUM_RESTARTS": 20,
            "RAW_SAMPLES": 4096,
            "OPTIMIZE_ACQF_OPTIONS": {"maxiter": 100, "method": "L-BFGS-B"}
        },
        verbose=False
    )

    # Run the optimization iteration (48 runs ~= 16 min with my 6 cores?)
    experiment()

    loss = get_loss(os.path.join(os.getcwd(), f"meta-bo/{eval_cnt}"))
    print(f"Loss: {loss}")
    return loss

result = gp_minimize(
    func=objective,
    dimensions=param_space,
    n_calls=40,
    n_initial_points=10,
    acq_func="EI",
    acq_optimizer="lbfgs",
    n_restarts_optimizer=20,
    noise="gaussian",
    random_state=69,
    verbose=True,
    n_jobs=1,
)

# Extract best parameters
best_gpr_p = result.x[0]
best_gpr_val_factor = result.x[1]
best_onorm_factor = 0.0 if not result.x[2] else result.x[3]

print(f"\nBest parameters found:")
print(f"gpr_p: {best_gpr_p:.4f}")
print(f"gpr_val_factor: {best_gpr_val_factor:.4f}")
print(f"onorm_factor: {best_onorm_factor:.4f}")
print(f"Best objective value: {result.fun:.6f}\n")
