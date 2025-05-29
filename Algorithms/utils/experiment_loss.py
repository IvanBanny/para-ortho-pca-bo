import polars as pl
import iohinspector


def get_loss(experiment_dir) -> pl.DataFrame:
    """Calculate experiment loss (less = better) based on ioh data in experiment_dir.

    Args:
        experiment_dir: The directory containing the experiment.

    Returns:
        loss (float): Loss value calculated based on ioh data in experiment_dir.
    """
    manager = iohinspector.DataManager()

    try:
        manager.add_folder(experiment_dir)
    except FileNotFoundError:
        return pl.DataFrame({
            'gpr_p': [],
            'gpr_val_factor': [],
            'onorm_factor': [],
            'loss': []
        })

    cols = ['data_id', 'gpr_p', 'gpr_val_factor', 'onorm_factor', 'doe', 'raw_y_best', 'current_y_best']

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
        .select(["data_id", "current_y_best"])
        .rename({"current_y_best": "f_init"})
    )

    # Join back to get f_init and f_opt for all rows
    df_enriched = df_with_row_num.join(experiment_info, on="data_id")

    # Calculate normalized gaps
    df_normalized = df_enriched.with_columns([
        (pl.col("current_y_best") / pl.col("f_init")).alias("normalized_gap")
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
