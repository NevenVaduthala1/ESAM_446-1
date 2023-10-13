import numpy as np
from scipy.special import factorial
from scipy import sparse
import math

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N
        
class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)
        
class Difference:
    def __matmul__(self, other):
        return self.matrix@other
    
class DifferenceUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis

        h = grid.dx
        N = grid.N

        # Recalculate the range of offsets based on derivative and convergence orders
        offsets = range(-convergence_order, convergence_order + 1)

        # Compute the Vandermonde matrix
        S = np.vander(offsets, increasing=True).T

        # Correct the powers of the nodes to account for the step size 'h'
        for i in range(1, 2 * convergence_order + 1):
            S[i] /= math.pow(h, i)

        # Calculate the RHS vector for the targeted derivative order
        b = np.zeros(2 * convergence_order + 1)
        b[derivative_order] = factorial(derivative_order)

        # Solve for the finite difference coefficients
        a = np.linalg.solve(S, b)

        # Populate the difference matrix considering periodic boundary conditions
        rows, cols, data = [], [], []
        for i in range(N):
            for j, coef in enumerate(a):
                col = (i + j - convergence_order) % N
                rows.append(i)
                cols.append(col)
                data.append(coef)

        self.matrix = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        

class DifferenceNonUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        
        N = grid.N
        x = grid.values

        offset_range = self._calculate_offset_range()

        D = np.zeros([N, N])

        for i in range(N):
            coefficients = self._calculate_coefficients(offset_range, x, i)
            D[i, [(i + offset) % N for offset in offset_range]] = coefficients

        self.matrix = D

    def _calculate_offset_range(self):
        offset = (2 * math.floor((self.derivative_order + 1) / 2) - 1 + self.convergence_order) // 2
        return range(-offset, offset + 1)

    def _calculate_coefficients(self, offset_range, x, i):
        b = np.zeros(len(offset_range))
        b[self.derivative_order] = 1

        h_values = np.array([x[(i + offset) % len(x)] - x[i] for offset in offset_range])
        S = np.column_stack([h_values**k / factorial(k) for k in range(len(offset_range))])

        return np.linalg.solve(S, b)