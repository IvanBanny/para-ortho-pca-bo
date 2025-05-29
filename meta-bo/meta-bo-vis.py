import matplotlib.pyplot as plt
import polars as pl

from Algorithms.utils.experiment_loss import get_loss

df = get_loss("meta-bo")

with pl.Config(tbl_rows=200, tbl_cols=20):
    print(df)

# Extract data from polars dataframe
x = df['gpr_p'].to_numpy()
y = df['gpr_val_factor'].to_numpy()
z = df['onorm_factor'].to_numpy()
colors = df['loss'].to_numpy()

# Create 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create scatter plot with color mapping
scatter = ax.scatter(x, y, z, c=colors, cmap='viridis', s=50, alpha=0.7)

# Add labels and title
ax.set_xlabel('gpr_p')
ax.set_ylabel('gpr_val_factor')
ax.set_zlabel('onorm_factor')
ax.set_title('3D Visualization of DataFrame')

# Add colorbar
colorbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
colorbar.set_label('loss')

# Show plot
plt.show()
