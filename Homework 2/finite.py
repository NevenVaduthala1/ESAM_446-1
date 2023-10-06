import numpy as np
from scipy.sparse import diags
import pytest
import matplotlib.pyplot as plt

# Class Definitions
class UniformPeriodicGrid:
    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N

class Difference:
    def __matmul__(self, other):
        return self.matrix @ other

class DifferenceUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid):
        super().__init__()
        N = grid.N
        dx = grid.dx
class DifferenceUniformGrid(Difference):
    def __init__(self, derivative_order, convergence_order, grid):
        super().__init__()

        N = grid.N
        dx = grid.dx

        if derivative_order == 1 and convergence_order == 2:
            coeffs = np.array([-0.5, 0, 0.5])  # Directly using coefficients for first derivative, second-order accuracy
            offsets = np.array([-1, 0, 1])  # Corresponding offsets

        # You can add more conditions here for other orders of derivative and accuracy as needed

        else:
            raise ValueError(f"Combination of derivative order {derivative_order} and convergence order {convergence_order} not implemented yet.")

        self.matrix = diags(coeffs, offsets, (N, N)).toarray() / dx

        # Apply periodic boundary conditions
        if offsets[0] < 0:
            self.matrix += diags([coeffs[0]], [offsets[0] + N], (N, N)).toarray() / dx
        if offsets[-1] > 0:
            self.matrix += diags([coeffs[-1]], [offsets[-1] - N], (N, N)).toarray() / dx
