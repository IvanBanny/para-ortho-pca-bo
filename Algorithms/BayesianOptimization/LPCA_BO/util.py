import numpy as np

from Algorithms import PCA_BO


def transform_point_to_original_space(self: PCA_BO, z: np.ndarray) -> np.ndarray:
    """Transform a point from reduced space back to the original space.

    Args:
        z (np.ndarray): Point in the reduced space.

    Returns:
        np.ndarray: Corresponding point in the original space.
    """
    # Handle the case before PCA is fitted
    if self.pca is None:
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

    # Reshape to 2D for sklearn
    z_2d = z.reshape(1, -1)

    # Transform back to original space
    x = self.pca.inverse_transform(z_2d) + self.data_mean

    return x.ravel()


def calculate_weights(self: PCA_BO) -> np.ndarray:
    """Calculate rank-based weights for PCA transformation.

    This implements the rank-based weighting scheme described in the original PCA-BO paper,
    where better points (with lower function values for minimization) are assigned higher weights.

    Returns:
        np.ndarray: Weights for each data point.
    """
    n = len(self.f_evals)

    # Get the ranking of points (1 = best, n = worst)
    if self.maximization:
        # For maximization, higher values are better
        ranks = np.argsort(np.argsort(-np.array(self.f_evals))) + 1
    else:
        # For minimization, lower values are better
        ranks = np.argsort(np.argsort(np.array(self.f_evals))) + 1

    # Calculate pre-weights
    pre_weights = np.log(n) - np.log(ranks)

    # Normalize weights
    weights = pre_weights / pre_weights.sum()

    return weights














def latin_hypercube(n_points, dim):
    """Generate points using Latin hypercube sampling.

    Parameters
    ----------
    n_points : int
        Number of points to generate.
    dim : int
        Dimension of the space.

    Returns
    -------
    numpy.ndarray
        Generated points, shape (n_points, dim).
    """
    points = np.zeros((n_points, dim))
    centers = (1.0 + 2.0 * np.arange(n_points)) / (2.0 * n_points)

    for i in range(dim):
        points[:, i] = centers[np.random.permutation(n_points)]

    return points


def from_unit_cube(points, lb, ub):
    """Scale points from [0, 1]^d to [lb, ub].

    Parameters
    ----------
    points : numpy.ndarray
        Points in [0, 1]^d, shape (n_points, dim).
    lb : numpy.ndarray
        Lower bounds, shape (dim,).
    ub : numpy.ndarray
        Upper bounds, shape (dim,).

    Returns
    -------
    numpy.ndarray
        Scaled points, shape (n_points, dim).
    """
    return lb + (ub - lb) * points


def to_unit_cube(points, lb, ub):
    """Scale points from [lb, ub] to [0, 1]^d.

    Parameters
    ----------
    points : numpy.ndarray
        Points in [lb, ub], shape (n_points, dim).
    lb : numpy.ndarray
        Lower bounds, shape (dim,).
    ub : numpy.ndarray
        Upper bounds, shape (dim,).

    Returns
    -------
    numpy.ndarray
        Scaled points, shape (n_points, dim).
    """
    return (points - lb) / (ub - lb)

