from typing import Tuple

import numpy as np

from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import MyPCA


def calculate_pc1_bounds_intersection(bounds: np.ndarray, pca: MyPCA) -> Tuple[float, float]:
    """
    Calculate the intersection points of the PC1 line with the rectangular bounds.

    Args:
        bounds: 2D array of shape (2, 2) where bounds[0] = [x_min, x_max] and bounds[1] = [y_min, y_max]
        pca: PCA object with transform_to_reduced and transform_to_original methods

    Returns:
        Tuple of (z_min, z_max) representing the intersection points in PC1 space
    """

    # Extract bounds
    x_min, x_max = bounds[0, 0], bounds[0, 1]
    y_min, y_max = bounds[1, 0], bounds[1, 1]

    assert x_min < x_max
    assert y_min < y_max

    # Get the PC1 direction vector and center point
    # We'll use the PCA's mean as a reference point on the line
    center_original = pca.transform_to_original(np.array([[0]]))  # Transform z=0 back to original space
    center_x, center_y = center_original[0, 0], center_original[0, 1]

    # Get another point on the PC1 line to determine direction
    unit_point_original = pca.transform_to_original(np.array([[1]]))  # Transform z=1 back to original space
    direction_x = unit_point_original[0, 0] - center_x
    direction_y = unit_point_original[0, 1] - center_y

    # Normalize direction vector
    direction_norm = np.sqrt(direction_x ** 2 + direction_y ** 2)
    if direction_norm == 0:
        raise ValueError("PC1 direction vector has zero length")

    direction_x /= direction_norm
    direction_y /= direction_norm

    # Find intersections with each boundary line
    intersection_params = []

    # Left boundary (x = x_min)
    if True:  # Avoid division by zero
        t = (x_min - center_x) / direction_x
        y_intersect = center_y + t * direction_y
        intersection_params.append(t)

    # Right boundary (x = x_max)
    if abs(direction_x) > 1e-10:
        t = (x_max - center_x) / direction_x
        y_intersect = center_y + t * direction_y
        intersection_params.append(t)

    # Bottom boundary (y = y_min)
    if abs(direction_y) > 1e-10:  # Avoid division by zero
        t = (y_min - center_y) / direction_y
        x_intersect = center_x + t * direction_x
        intersection_params.append(t)

    # Top boundary (y = y_max)
    if abs(direction_y) > 1e-10:
        t = (y_max - center_y) / direction_y
        x_intersect = center_x + t * direction_x
        intersection_params.append(t)

    if len(intersection_params) < 2:
        raise ValueError(f"Found only {len(intersection_params)} intersection points, expected 2")

    # Remove duplicates and sort
    intersection_params = sorted(list(intersection_params))

    if len(intersection_params) < 2:
        raise ValueError("After removing duplicates, found less than 2 intersection points")

    # Take the two extreme intersection points
    t_min, t_max = intersection_params[1], intersection_params[2]

    # Convert parametric distances to PC1 coordinates
    # Since we normalized the direction vector, t represents the actual distance
    # We need to convert this back to PC1 space
    point_min_original = np.array([[center_x + t_min * direction_x, center_y + t_min * direction_y]])
    point_max_original = np.array([[center_x + t_max * direction_x, center_y + t_max * direction_y]])

    z_min = pca.transform_to_reduced(point_min_original)[0, 0]
    z_max = pca.transform_to_reduced(point_max_original)[0, 0]

    return float(z_min), float(z_max)