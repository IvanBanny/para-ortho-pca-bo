import dataclasses
import pickle
from abc import ABC
from typing import Union, Callable, Optional, Dict, Any, List



import numpy as np
import torch
from ioh.iohcpp.problem import RealSingleObjective, BBOB

from Algorithms.AbstractAlgorithm import AbstractAlgorithm
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.lpca_bo import CleanLPCABO
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import CleanPCABO, DOE, AcquisitionFunctionEnum, \
    PCBANumComponents, MyPCA
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo_interface_class import plot2d, IterationData

examplePath = r"examplelpcabo.pkl"

class CleanLPCABOInterface(AbstractAlgorithm):
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

        clean_pcabo = CleanLPCABOWithLogging(
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



class CleanLPCABOWithLogging(CleanLPCABO):
    # overwrite every function which result we want to save

    iterations: List[IterationData]

    plot_X_grid: Optional[np.ndarray] = None
    plot_Y_grid: Optional[np.ndarray] = None
    plot_Z: Optional[np.ndarray] = None

    global_optimum_x: Optional[np.ndarray] = None

    def optimize(self):
        self.iterations = []
        # in this function y and z do not referr to the usual meanings of y and z in this class
        if self.bounds.shape[0] == 2:
            x_min = self.bounds[0, 0]
            x_max = self.bounds[0, 1]
            y_min = self.bounds[1, 0]
            y_max = self.bounds[1, 1]

            padding_x = 0.1 * (x_max - x_min)
            padding_y = 0.1 * (y_max - y_min)

            x = np.linspace(x_min - padding_x, x_max + padding_x, 100)
            y = np.linspace(y_min - padding_y, y_max + padding_y, 100)
            X_grid, Y_grid = np.meshgrid(x, y)
            XY = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
            Z = np.array([self.problem(point) for point in XY])
            self.plot_Z = Z.reshape(X_grid.shape)
            self.plot_X_grid = X_grid
            self.plot_Y_grid = Y_grid

            self.global_optimum_x = self.problem.optimum.x

        super().optimize()

    def create_gpr_model(self, points_z, z_bounds):
        gpr = super().create_gpr_model(points_z, z_bounds)

        assert z_bounds.shape == (2, 1), z_bounds.shape

        plot_z_lb = np.min(points_z)
        plot_z_ub = np.max(points_z)

        iteration = self.iterations[-1]
        iteration.gpr_x = np.linspace(plot_z_lb, plot_z_ub, 100).reshape(-1, 1)  # 100 evenly spaced points in the 1 dimensional reduced space

        # Predict the mean and variance of the gpr model at the 100 evenly spaced points
        with torch.no_grad():
            posterior = gpr.posterior(torch.from_numpy(iteration.gpr_x))
            mean = posterior.mean.numpy().flatten()
            # Extract the standard deviation (square root of variance)
            std = posterior.variance.sqrt().numpy().flatten()

        # Store mean and std for visualization
        iteration.gpr_y = mean
        iteration.gpr_std = std  # Add this to store standard deviation

        return gpr

    def create_penalized_acquisition(self, acquisition_function, gpr_model, pca):
        penalized_acqf = super().create_penalized_acquisition(acquisition_function, gpr_model, pca)

        iteration : IterationData = self.iterations[-1]

        # Create a grid of points in the reduced space for visualization
        if iteration.gpr_x is not None:
            # We already have a grid of points from the GPR visualization
            # Use the same grid for the acquisition function
            z_points = iteration.gpr_x

            # Get the acquisition function values at these points
            with torch.no_grad():
                # Convert to tensor
                z_tensor = torch.from_numpy(z_points.reshape(-1, 1, 1))
                # Get acqf values
                acqf_values = penalized_acqf(z_tensor).detach().numpy().flatten()

            # Store for visualization
            iteration.acqf_x = z_points
            iteration.acqf_y = acqf_values

        return penalized_acqf

    def iteration(self):
        self.iterations.append(IterationData())
        super().iteration()

        self.iterations[-1].points_x = self.X
        self.iterations[-1].points_y = self.fX
        self.iterations[-1].bounds = self.return_tr_bounds()

        with open(examplePath, 'wb') as f:
            pickle.dump(self, f)

        print(f'saved after iteration', len(self.iterations))

    def fit_pca(self):
        pca = super().fit_pca()

        self.iterations[-1].pca = pca

        return pca

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["problem"]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)

if __name__ == "__main__":
    with open(examplePath, 'rb') as f:
        loaded_data : CleanLPCABOWithLogging = pickle.load(f)
        print(len(loaded_data.iterations))
        print(loaded_data.X)
        plot2d(loaded_data)