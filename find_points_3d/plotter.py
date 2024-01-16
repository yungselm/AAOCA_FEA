import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "C:/WorkingData/Documents/2_Coding/Python/FEManalysis/1_all_results/all_results.csv"
df = pd.read_csv(path, sep=";")

# Find the index of the row with the lowest total_mse
min_index = df['total_mse'].idxmin()

# Create a 3D plot with a surface plot and scatter points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot the surface
ax.plot_trisurf(df["Poisson_ratio"], df["E_module"], df["total_mse"], edgecolor='none', alpha=0.8)
# Scatter points
ax.scatter(df["Poisson_ratio"], df["E_module"], df["total_mse"], c='r', marker='o', label='Simulations')
# Mark the lowest total_mse point
ax.scatter(
    df["Poisson_ratio"].iloc[min_index],
    df["E_module"].iloc[min_index],
    df["total_mse"].iloc[min_index],
    c='g',
    marker='X',
    s=100,
    label='Lowest mean squared error',
)

ax.set_xlabel('Poisson ratio')
ax.set_ylabel('Youngs modulus [MPa]')
ax.set_zlabel('Mean squared error')
ax.legend()
plt.show()
