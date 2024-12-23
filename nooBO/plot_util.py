import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a helper function to plot the surface
def plot_surface(function, x_range, y_range, title, ax):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Convert to PyTorch tensor
            point = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float)
            Z[i, j] = function(point).item()
    
    surf = ax.plot_surface(X, Y, Z, cmap="jet", edgecolor='k', linewidth=0.1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    return surf