import os
import polars as pl
import iohinspector

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="iohinspector")

manager = iohinspector.DataManager()
manager.add_folder("experiment")

df = manager.select(dimensions=[10]).load(False, True)

print(df)
print(df.columns)
