from argparse import ArgumentParser
from tqdm import tqdm
from time import perf_counter
import json
import os

import numpy as np
import torch

from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.generation import get_best_candidates, gen_candidates_torch
from botorch.optim import gen_batch_initial_conditions

from ioh import get_problem

import warnings
from botorch.exceptions import InputDataWarning, OptimizationWarning, BadInitialCandidatesWarning
warnings.filterwarnings("ignore", category=InputDataWarning)
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def main(budget=200, doe_num=30, dim=6, fid=19, instance=0, seed=69, if_gpu=True, if_32=True, method=0):
    device = torch.device(f"cuda" if torch.cuda.is_available() and if_gpu else "cpu")
    dtype = (torch.float if if_32 else torch.double)
    torch.set_default_dtype(dtype)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    num_samples = doe_num
    problem = get_problem(fid=fid, instance=instance, dimension=dim)
    bounds = torch.tensor(np.stack([problem.bounds.lb, problem.bounds.ub]), device=device, dtype=dtype)
    train_x = torch.rand(num_samples, dim, device=device, dtype=dtype)
    train_x = train_x * (bounds[1] - bounds[0]) + bounds[0]
    train_obj = torch.tensor(problem(np.random.rand(num_samples, dim)), device=device, dtype=dtype).unsqueeze(-1)

    pbar = tqdm(total=budget, initial=num_samples)
    pbar.set_description(f"Best: {train_obj.min().item():.6f}")

    stats = {"gpr": [], "acqf": []}
    time_start = perf_counter()

    while num_samples < budget:
        # Init and fit GPR
        t0 = perf_counter()
        model = SingleTaskGP(train_x, train_obj).to(train_x)
        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_x)
        fit_gpytorch_mll(mll)
        stats["gpr"].append(perf_counter() - t0)

        # Init and optimize acqf
        t0 = perf_counter()
        acqf = LogExpectedImprovement(model, train_obj.min(), maximize=False).to(train_x)

        if method == 0:
            candidates, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=1024,
                options={
                    "batch_limit": 10,
                    "maxiter": 300,
                    "method": "L-BFGS-B",
                    "ftol": 1e-8,
                    "sequential": True
                }
            )
        else:
            batch_initial_conditions = gen_batch_initial_conditions(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=1024,
            )
            batch_candidates, batch_acq_values = gen_candidates_torch(
                initial_conditions=batch_initial_conditions,
                acquisition_function=acqf,
                lower_bounds=bounds[0],
                upper_bounds=bounds[1],
                optimizer=torch.optim.Adam,
                options={"maxiter": 300},
            )
            candidates = get_best_candidates(
                batch_candidates=batch_candidates, batch_values=batch_acq_values
            ).detach()

        candidates = candidates[: budget - num_samples]
        stats["acqf"].append(perf_counter() - t0)

        # Observe
        observations = torch.tensor(problem(candidates.cpu().detach().numpy()), device=device, dtype=dtype).unsqueeze(-1)
        train_x = torch.cat([train_x, candidates])
        train_obj = torch.cat([train_obj, observations])

        new_samples = candidates.shape[0]
        num_samples += new_samples
        pbar.update(new_samples)
        pbar.set_description(f"Best: {train_obj.min().item():.6f}")

    pbar.close()

    time_all = perf_counter() - time_start
    print(f"Time: {time_all:.2f}s | fit_gpytorch_mll: {sum(stats['gpr']):.2f}s | optimize_acqf: {sum(stats['acqf']):.2f}s")
    stats = {k: sum(v) for k, v in stats.items()}
    stats.update({
        "total": time_all,
        "dim": dim,
        "gpu": str(device),
        "dtype": str(dtype),
        "method": method,
        "budget": budget,
        "fid": fid,
        "instance": instance,
        "seed": seed,
        "best": train_obj.min().item()
    })
    stats_list = []
    if os.path.exists("stats.json"):
        with open("stats.json", 'r') as f:
            stats_list = json.load(f)
    stats_list.append(stats)
    with open("stats.json", 'w') as f:
        json.dump(stats_list, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--budget', type=int, default=200)
    parser.add_argument('--doe_num', type=int, default=30)
    parser.add_argument('--dim', type=int, default=5)
    parser.add_argument('--fid', type=int, default=19)
    parser.add_argument('--instance', type=int, default=0)
    parser.add_argument('--seed', type=float, default=69)
    parser.add_argument('--if_gpu', action='store_true')
    parser.add_argument('--if_32', action='store_true')
    parser.add_argument('--method', type=int, default=0)
    args = parser.parse_args()

    main(
        budget=args.budget,
        doe_num=args.doe_num,
        dim=args.dim,
        fid=args.fid,
        instance=args.instance,
        seed=args.seed,
        if_gpu=args.if_gpu,
        if_32=args.if_32,
        method=args.method
    )
