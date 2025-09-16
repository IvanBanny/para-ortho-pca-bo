import dataclasses
import pickle
from abc import ABC
from typing import Union, Callable, Optional, Dict, Any, List, Tuple

import numpy as np
import torch
from ioh.iohcpp.problem import RealSingleObjective, BBOB

from Algorithms.AbstractAlgorithm import AbstractAlgorithm
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import CleanPCABO, DOE, AcquisitionFunctionEnum, \
    PCBANumComponents, MyPCA, calculate_reduced_space_bounds
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.vanilla_bo import CleanVanillaBO

examplePath = r"example.pkl"

class CleanVanillaBOInterface(AbstractAlgorithm):
    def __init__(
            self,
            budget: int,
            n_DoE: int = 0,
            n_components: Optional[int] = None,
            var_threshold: float = 0.95,
            acquisition_function: str = "expected_improvement",
            random_seed: int = 43,
            torch_config: Optional[Dict[str, Any]] = None,
            visualize: bool = False,
            vis_output_dir: str = "./visualizations",
            save_logs: bool = False,
            log_dir: str = "./logs",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.budget = budget
        self.n_DoE = n_DoE
        self.n_components = n_components
        self.var_threshold = var_threshold
        self.acquisition_function = AcquisitionFunctionEnum.from_name(acquisition_function)
        self.random_seed = random_seed
        self.torch_config = torch_config
        self.visualize = visualize
        self.vis_output_dir = vis_output_dir
        self.save_logs = save_logs


    def __call__(
            self,
            problem: Union[RealSingleObjective, BBOB, Callable],
            dim: Optional[int] = -1,
            bounds: Optional[np.ndarray] = None,
            **kwargs
    ):
        super().__call__(problem, dim, bounds, **kwargs)

        assert isinstance(problem, Callable)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        clean_pcabo = CleanVanillaBO(
            problem=problem,
            budget=self.budget,
            bounds=self.bounds,
            doe=DOE(n=self.n_DoE),
            maximization=self.maximization,
            acquisition_function_class=self.acquisition_function.class_type,
        )

    def __str__(self):
        return "CleanVanillaBO_Interface"

    def reset(self):
        super().reset()

