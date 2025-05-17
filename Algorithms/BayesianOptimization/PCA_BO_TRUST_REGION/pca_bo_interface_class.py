from abc import ABC
from typing import Union, Callable, Optional, Dict, Any

import numpy as np
from ioh.iohcpp.problem import RealSingleObjective, BBOB

from Algorithms.AbstractAlgorithm import AbstractAlgorithm
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import CleanPCABO, DOE, AcquisitionFunctionEnum, \
    PCBANumComponents


class CleanPCABOInterface(AbstractAlgorithm):
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

        self.pca_num_components = PCBANumComponents(
                num_components=self.n_components,
                var_threshold=self.var_threshold
            )



    def __call__(
            self,
            problem: Union[RealSingleObjective, BBOB, Callable],
            dim: Optional[int] = -1,
            bounds: Optional[np.ndarray] = None,
            **kwargs
         ):
        super().__call__(problem, dim, bounds, **kwargs)


        assert isinstance(problem, Callable)

        clean_pcabo = CleanPCABO(
            problem=problem,
            budget=self.budget,
            bounds=self.bounds,
            doe=DOE(n=self.n_DoE),
            maximization=self.maximization,
            acquisition_function_class=self.acquisition_function.class_type,
            pca_num_components=self.pca_num_components,
        )





    def __str__(self):
        return "CleanPCABO_Interface"

    def reset(self):
        super().reset()

    