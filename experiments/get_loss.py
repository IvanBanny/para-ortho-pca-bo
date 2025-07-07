import polars as pl

from src import get_loss

with pl.Config(tbl_cols=50, tbl_rows=200):
    print(get_loss("data", verbose=True))
