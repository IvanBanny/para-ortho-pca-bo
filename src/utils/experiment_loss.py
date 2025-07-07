from typing import List, Optional, Union, Dict
import polars as pl
import src.utils.iohreader as iohreader


def get_loss(
        experiment_dir: str,
        algorithms: Optional[List[str]] = None,
        batch_sizes: Optional[List[int]] = None,
        dimensions: Optional[List[int]] = None,
        functions: Optional[List[int]] = None,
        average: bool = False,
        verbose: bool = False
) -> Union[pl.DataFrame, Dict[str, pl.DataFrame]]:
    """Calculate experiment loss (less = better) based on ioh data in experiment_dir.

    Args:
        experiment_dir: The directory containing the experiment.
        algorithms: src to consider (None for all).
        batch_sizes: Batch sizes to consider (None for all).
        dimensions: Dimensions to consider (None for all).
        functions: Functions to consider (None for all).
        average: Output one entry only - the average loss across all runs.
        verbose: Output detailed metrics (auc, final, loss).

    Returns:
        DataFrame with varying columns and loss metrics, or dict of DataFrames if multiple algorithms.
    """
    manager = iohreader.DataManager()

    try:
        manager.add_folder(experiment_dir)
    except FileNotFoundError:
        return pl.DataFrame()

    # Load and process data by algorithm
    algo_data = _load_algorithm_data(manager, dimensions)
    if not algo_data:
        return pl.DataFrame()

    # Apply filters and process
    filtered_algo_data = _apply_filters(algo_data, algorithms, batch_sizes, dimensions, functions)
    if not filtered_algo_data:
        return pl.DataFrame()

    # Determine return format and process results
    return_dict = len(filtered_algo_data) > 1 and (algorithms is None or len(algorithms) != 1)

    if return_dict:
        return {algo: _process_single_algorithm(df, average, verbose)
                for algo, df in filtered_algo_data.items()}
    else:
        df = next(iter(filtered_algo_data.values()))
        return _process_single_algorithm(df, average, verbose)


def _load_algorithm_data(manager, dimensions: Optional[List[int]]) -> Dict[str, pl.DataFrame]:
    """Load data by algorithm, handling dimension alignment efficiently."""
    available_algorithms = manager.overview["algorithm_name"].unique().to_list()
    dimensions_to_use = dimensions or manager.overview["dimension"].unique().to_list()

    algo_data = {}

    for algo in available_algorithms:
        df_list = []

        for dim in dimensions_to_use:
            try:
                data = manager.select(algorithms=[algo], dimensions=[dim]).load(False, True)
                if len(data) > 0:
                    # More efficient column filtering
                    data = _filter_columns(data)
                    df_list.append(data)
            except (FileNotFoundError, ValueError, KeyError):
                # More specific exception handling
                continue

        if df_list:
            algo_data[algo] = _align_and_concat_dataframes(df_list)

    return algo_data


def _filter_columns(data: pl.DataFrame) -> pl.DataFrame:
    """Remove x-coordinate and time columns efficiently."""
    cols_to_keep = [
        col for col in data.columns
        if not (_is_x_coordinate_column(col) or col.endswith("time"))
    ]
    return data.select(cols_to_keep)


def _is_x_coordinate_column(col: str) -> bool:
    """Check if column is an x-coordinate column (x0, x1, x2, etc.)."""
    return col.startswith('x') and len(col) > 1 and col[1:].isdigit()


def _align_and_concat_dataframes(df_list: List[pl.DataFrame]) -> pl.DataFrame:
    """Align DataFrames by common columns and concatenate."""
    if len(df_list) == 1:
        return df_list[0]

    # Find common columns more efficiently
    common_cols = set(df_list[0].columns)
    for df in df_list[1:]:
        common_cols.intersection_update(df.columns)

    common_cols = list(common_cols)
    aligned_dfs = [df.select(common_cols) for df in df_list]
    return pl.concat(aligned_dfs)


def _apply_filters(
        algo_data: Dict[str, pl.DataFrame],
        algorithms: Optional[List[str]],
        batch_sizes: Optional[List[int]],
        dimensions: Optional[List[int]],
        functions: Optional[List[int]]
) -> Dict[str, pl.DataFrame]:
    """Apply filters to algorithm data."""
    filtered_algo_data = {}

    # Build filter conditions once
    filter_conditions = []
    if algorithms is not None:
        filter_conditions.append(pl.col("algorithm_name").is_in(algorithms))
    if batch_sizes is not None:
        filter_conditions.append(pl.col("batch_size").is_in(batch_sizes))
    if dimensions is not None:
        filter_conditions.append(pl.col("dimension").is_in(dimensions))
    if functions is not None:
        filter_conditions.append(pl.col("function_id").is_in(functions))

    # Combine all conditions
    combined_filter = pl.fold(pl.lit(True), lambda acc, x: acc & x, filter_conditions) if filter_conditions else pl.lit(
        True)

    for algo, df in algo_data.items():
        filtered_df = df.filter(combined_filter).drop_nulls()
        if len(filtered_df) > 0:
            filtered_algo_data[algo] = filtered_df

    return filtered_algo_data


def _process_single_algorithm(df: pl.DataFrame, average: bool, verbose: bool) -> pl.DataFrame:
    """Process data for a single algorithm."""
    # Get column classifications
    varying_columns = _get_varying_columns(df)
    algorithm_parameters = _get_algorithm_parameters(df, varying_columns)

    # Select only varying columns
    df = df.select(varying_columns)

    # Process the data pipeline
    df_with_metrics = _calculate_run_metrics(df)

    # Return aggregated results
    return _aggregate_results(df_with_metrics, algorithm_parameters, average, verbose)


def _get_varying_columns(df: pl.DataFrame) -> List[str]:
    """Get columns that have more than one unique value."""
    ignore_columns = {
        "algorithm_info", "suite", "function_name", "instance", "run_id", "evals", "budget"
    }

    return [
        col for col in df.columns
        if col not in ignore_columns and df[col].n_unique() > 1
    ]


def _get_algorithm_parameters(df: pl.DataFrame, varying_columns: List[str]) -> List[str]:
    """Extract algorithm parameters from varying columns."""
    experimental_conditions = {"dimension", "batch_size", "function_id", "algorithm_name"}
    data_columns = {
        "data_id", "iteration", "best_y", "raw_y_best", "evaluations",
        "evals", "time", "instance", "random_seed", "doe"
    }

    # Sort to ensure consistent column ordering
    return sorted([
        col for col in varying_columns
        if col not in experimental_conditions and col not in data_columns
    ])


def _calculate_run_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate metrics for each run."""
    # Add iteration numbers and get f_init
    df_with_iteration = (
        df.sort("data_id")
        .with_columns(pl.int_range(pl.len()).over("data_id").alias("iteration"))
    )

    # Get f_init more efficiently
    f_init = (
        df_with_iteration
        .filter(pl.col("iteration") == pl.col("doe") - 1)
        .select(["data_id", pl.col("raw_y_best").alias("f_init")])
    )

    # Join and calculate normalized gaps
    df_normalized = (
        df_with_iteration
        .join(f_init, on="data_id")
        .with_columns((pl.col("raw_y_best") / pl.col("f_init")).alias("normalized_gap"))
    )

    # Calculate metrics per run with constants
    WEIGHT_AUC = 0.3
    WEIGHT_FINAL = 0.7

    # Get algorithm parameter columns (excluding fixed experimental conditions) and sort them
    varying_columns = _get_varying_columns(df)
    algorithm_params = _get_algorithm_parameters(df, varying_columns)

    return (
        df_normalized
        .filter(pl.col("iteration") >= pl.col("doe"))  # Only BO iterations
        .group_by("data_id")
        .agg([
            # Keep algorithm parameters in sorted order
            *[pl.col(col).first() for col in algorithm_params],
            # Calculate metrics in consistent order
            pl.col("normalized_gap").mean().alias("auc"),
            pl.col("normalized_gap").last().alias("final"),
            (WEIGHT_AUC * pl.col("normalized_gap").mean() +
             WEIGHT_FINAL * pl.col("normalized_gap").last()).alias("loss")
        ])
    )


def _aggregate_results(
        run_metrics: pl.DataFrame,
        algorithm_parameters: List[str],
        average: bool,
        verbose: bool
) -> pl.DataFrame:
    """Aggregate results based on parameters."""
    # Define output columns in consistent order
    metric_cols = ["auc", "final", "loss"] if verbose else ["loss"]

    if average or not algorithm_parameters:
        # Average across all runs - select columns in consistent order
        return run_metrics.select([
            pl.col(col).mean().alias(col) for col in metric_cols
        ])
    else:
        # Group by algorithm parameters
        result = run_metrics.group_by(algorithm_parameters).agg([
            pl.col(col).mean().alias(col) for col in metric_cols
        ])

        # Sort rows and ensure consistent column order: algorithm params first, then metrics
        result = result.sort(algorithm_parameters)
        final_column_order = algorithm_parameters + metric_cols
        return result.select(final_column_order)


def get_opca_loss(experiment_dir) -> pl.DataFrame:
    """Calculate experiment loss (less = better) based on ioh data in experiment_dir.

    Args:
        experiment_dir: The directory containing the experiment.

    Returns:
        loss (float): Loss value calculated based on ioh data in experiment_dir.
    """
    manager = iohreader.DataManager()

    try:
        manager.add_folder(experiment_dir)
    except FileNotFoundError:
        return pl.DataFrame({
            'gpr_p': [],
            'gpr_val_factor': [],
            'onorm_factor': [],
            'loss': []
        })

    cols = ['data_id', 'gpr_p', 'gpr_val_factor', 'onorm_factor', 'doe', 'raw_y_best']

    df = pl.concat([manager.select(dimensions=[d]).load(False, True)
                   .select(cols).drop_nulls() for d in manager.overview["dimension"].unique().to_list()])

    df_with_row_num = (
        df.sort(["data_id"])  # Ensure consistent ordering
        .with_columns([
            pl.int_range(pl.len()).over("data_id").alias("iteration")
        ])
    )

    # Get f_init (best value at the end of DOE phase)
    experiment_info = (
        df_with_row_num
        .filter(pl.col("iteration") == pl.col("doe") - 1)  # Get row at doe position
        .select(["data_id", "raw_y_best"])
        .rename({"raw_y_best": "f_init"})
    )

    # Join back to get f_init and f_opt for all rows
    df_enriched = df_with_row_num.join(experiment_info, on="data_id")

    # Calculate normalized gaps
    df_normalized = df_enriched.with_columns([
        (pl.col("raw_y_best") / pl.col("f_init")).alias("normalized_gap")
    ])

    weight_auc = 0.7
    weight_final = 0.3

    return (
        df_normalized
        .filter(pl.col("iteration") >= pl.col("doe"))  # Only BO iterations
        .group_by("data_id")
        .agg([
            pl.col("gpr_p").first(),
            pl.col("gpr_val_factor").first(),
            pl.col("onorm_factor").first(),
            # Loss
            (weight_auc * pl.col("normalized_gap").mean() +
             weight_final * pl.col("normalized_gap").last()).alias("loss")
        ])
        .group_by("gpr_p", "gpr_val_factor", "onorm_factor")
        .agg([
            pl.col("loss").mean()
        ])
    )
