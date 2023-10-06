import numpy as np
from scipy.sparse import diags

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
        N = grid.N
        dx = grid.dx
        
        coeffs = None
        offsets = None
        
        if derivative_order == 1 and convergence_order == 2:
            coeffs = np.array([-0.5, 0, 0.5]) / dx
            offsets = np.array([-1, 0, 1])
        elif derivative_order == 2 and convergence_order == 2:
            coeffs = np.array([1, -2, 1]) / dx**2
            offsets = np.array([-1, 0, 1])
            
        if coeffs is not None and offsets is not None:
            self.matrix = diags(coeffs, offsets, (N, N)).toarray()

            # Apply periodic boundary conditions
            if offsets[0] < 0:
                self.matrix[0, -1] = coeffs[0]
            if offsets[-1] > 0:
                self.matrix[-1, 0] = coeffs[-1]
        else:
            raise ValueError("Unsupported derivative or convergence order")

# Test Cases
if __name__ == "__main__":
    grid = UniformPeriodicGrid(100, 2*np.pi)  # Increased grid points for better accuracy
    x = grid.values
    f = np.sin(x)
    
    # First Derivative Test
    first_derivative_operator = DifferenceUniformGrid(1, 2, grid)
    numerical_first_derivative = first_derivative_operator @ f
    analytical_first_derivative = np.cos(x)
    max_error_first = np.max(np.abs(numerical_first_derivative - analytical_first_derivative))
    
    print("First Derivative Test")
    print("Numerical Derivative:", numerical_first_derivative)
    print("Analytical Derivative:", analytical_first_derivative)
    print("Max Error:", max_error_first)
    
    # Second Derivative Test
    second_derivative_operator = DifferenceUniformGrid(2, 2, grid)
    numerical_second_derivative = second_derivative_operator @ f
    analytical_second_derivative = -np.sin(x)
    max_error_second = np.max(np.abs(numerical_second_derivative - analytical_second_derivative))
    
    print("\nSecond Derivative Test")
    print("Numerical Derivative:", numerical_second_derivative)
    print("Analytical Derivative:", analytical_second_derivative)
    print("Max Error:", max_error_second)
