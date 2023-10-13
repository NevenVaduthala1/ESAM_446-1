import numpy as np
from scipy import sparse
import sympy as sp


class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, u, f):
        super().__init__()
        self.u = u
        self.f = f


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.u + dt*self.f(self.u)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.f(self.u)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.f(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.f(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, f1, f2):
        super().__init__()
        self.u = u
        self.f1 = f1
        self.f2 = f2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(ExplicitTimestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a  # coefficients for stages
        self.b = b  # coefficients for final sum

    def _step(self, dt):
        u_shape = self.u.shape
        ks = np.zeros((u_shape[0], self.stages), dtype=self.u.dtype)  # Initialized with zeros, and considering the dtype of u.

        # First stage, k0
        ks[:, 0] = self.f(self.u)

        # Computing other stages
        for i in range(1, self.stages):
            weighted_sum = np.dot(ks[:, :i], self.a[i, :i])  # Vectorized sum, considering dtype and efficiency
            ks[:, i] = self.f(self.u + dt * weighted_sum)

        # Final summation to compute the next step
        final_sum = np.dot(ks, self.b)  # Vectorized sum to get the final result

        # Updating the solution
        return self.u + dt * final_sum


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt
        self.fPast = np.zeros((len(u), steps))  # Store past values of f(u)
        self.ab_coefs = self.compute_ab_coefs(steps)  # Pre-compute AB coefficients

    def compute_ab_coefs(self, steps):
        t = sp.symbols('t')
        a = []
        for i in range(steps):
            lagrange_polynomial = 1
            for j in range(steps):
                if j != i:
                    lagrange_polynomial *= (t - j) / (i - j)
            a_i = sp.integrate(lagrange_polynomial, (t, 0, steps - 1))
            a.append(float(a_i))  # Convert to float for numerical calculations
        return np.array(a)

    def _step(self, dt):
        if self.iter < self.steps - 1:
            self.fPast[:, self.iter] = self.f(self.u)
            return self.u + dt * self.f(self.u)
        else:
            if self.iter == self.steps - 1:
                # When enough steps are available, shift to AB
                self.fPast = np.roll(self.fPast, shift=-1, axis=1)
            self.fPast[:, -1] = self.f(self.u)
            delta_u = dt * np.dot(self.ab_coefs, self.fPast)
            return self.u + delta_u





