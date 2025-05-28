import os
import polars as pl
import iohinspector

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="iohinspector")


def get_loss(experiment_dir) -> float:
    """Calculate experiment loss (less = better) based on ioh data in experiment_dir.

    Args:
        experiment_dir: The directory containing the experiment.

    Returns:
        loss (float): Loss value calculated based on ioh data in experiment_dir.
    """
    manager = iohinspector.DataManager()
    manager.add_folder(experiment_dir)

    df = manager.select(dimensions=[10]).load(False, True)
    df = df.select(["data_id", "doe", "current_y", "current_y_best", "optimum"]).drop_nulls()

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
        .select(["data_id", "current_y_best", "optimum"])
        .rename({"current_y_best": "f_init", "optimum": "f_opt"})
    )

    # Join back to get f_init and f_opt for all rows
    df_enriched = df_with_row_num.join(experiment_info, on="data_id")

    # Calculate normalized gaps
    df_normalized = df_enriched.with_columns([
        ((pl.col("current_y_best") - pl.col("f_opt")) /
         (pl.col("f_init") - pl.col("f_opt"))).alias("normalized_gap")
    ])

    # Calculate metrics per experiment
    metrics = (
        df_normalized
        .filter(pl.col("iteration") >= pl.col("doe"))  # Only BO iterations
        .group_by("data_id")
        .agg([
            # AUC approximation (mean of normalized gaps during BO phase)
            pl.col("normalized_gap").mean().alias("normalized_auc"),
            # Final gap
            pl.col("normalized_gap").last().alias("normalized_final_gap")
        ])
    )

    # Combine metrics
    weight_auc = 0.7
    weight_final = 0.3

    final_metrics = metrics.with_columns(
        (weight_auc * pl.col("normalized_auc") +
         weight_final * pl.col("normalized_final_gap")).alias("combined_loss")
    )

    return final_metrics["combined_loss"].mean()
