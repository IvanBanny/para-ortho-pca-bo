import torch


def estimate_f00_variance(b, d, f_values, k=1, polynomial_degree=2):
    """
    Estimate variance of f(0,0) using all data points with distance-scaled uncertainty.
    Variance smoothly approaches 0 as we get closer to origin.

    Args:
        b, d, f_values: 1D tensors of data points
        k: scale coefficient
        polynomial_degree: degree of polynomial fit (1=linear, 2=quadratic)

    Returns:
        variance: estimated variance f(0,0)
    """
    n = len(b)

    # Compute distance to origin for scaling
    distances = torch.sqrt(b ** 2 + d ** 2)
    min_distance = torch.min(distances)

    # Create polynomial features using ALL points
    def make_poly_features(b_vals, d_vals, degree):
        features = [torch.ones_like(b_vals)]  # constant term

        for deg in range(1, degree + 1):
            for i in range(deg + 1):
                j = deg - i
                if i == 0 and j == 0:
                    continue  # skip constant (already added)
                features.append(b_vals ** i * d_vals ** j)

        return torch.stack(features, dim=1)

    # Fit polynomial to ALL data points
    X = make_poly_features(b, d, polynomial_degree)

    try:
        # Solve polynomial fit
        coef = torch.linalg.lstsq(X, f_values).solution

        # Predict f(0,0) using the polynomial
        X_origin = make_poly_features(torch.tensor([0.0]), torch.tensor([0.0]), polynomial_degree)
        f_00_pred = (X_origin @ coef).item()

        # Calculate residual variance from fit
        f_pred = X @ coef
        residuals = f_values - f_pred
        n_params = len(coef)

        if n > n_params:
            residual_var = torch.var(residuals, unbiased=True)
        else:
            residual_var = torch.var(residuals, unbiased=False)

        # Ensure minimum baseline uncertainty
        data_var = torch.var(f_values, unbiased=False)
        base_uncertainty = torch.max(residual_var, data_var * 0.01)

    except:
        # Fallback: simple mean
        f_00_pred = torch.mean(f_values).item()
        base_uncertainty = torch.var(f_values, unbiased=False)

    # Distance-based scaling: variance approaches 0 as min_distance approaches 0
    # Using sigmoid-like scaling: small distances give small multipliers
    distance_scale = min_distance ** 2 / (min_distance ** 2 + 0.5)  # ranges from 0 to 1

    # Add extrapolation uncertainty that grows with distance
    extrapolation_factor = 1.0 + min_distance ** 2
    # extrapolation_factor = 1.0 + (min_distance / 5).log()

    # Final variance: base uncertainty scaled by distance and extrapolation
    final_variance = base_uncertainty * k * distance_scale * extrapolation_factor

    return final_variance  # mean: torch.tensor(f_00_pred)
