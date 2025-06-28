from Algorithms import get_loss
import polars as pl

with pl.Config(tbl_cols=50, tbl_rows=200):
    print(get_loss("meta-bo/meta-bo-data", verbose=True))
