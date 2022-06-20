import numpy as np
import dolfin as dl

#### WORK IN PROGRESS DO NOT USE

raise RuntimeError('Do not use!')

class StokesForwardProblem:
    def __init__(me, rheology_n, rheology_A):
        # Rheology
        me.rheology_n = rheology_n
        me.rheology_A = rheology_A

        # Basal Boundary
        me.normal = normal
        me.ds_base = ds_base

        # Forcing term
        self.f = f

        # Smooth strain
        self.eps = dl.Constant(1e-6)

        # penalty parameter for Dirichlet condition
        self.lam = dl.Constant(0.5 * lam)

    def _epsilon(self, velocity):
        return dl.sym(dl.grad(velocity))

    def _tang(self, velocity):
        return (velocity - dl.outer(self.normal, self.normal) * velocity)

    def energy_fun(self, u, m):
        velocity, _ = dl.split(u)
        normEu12 = 0.5 * dl.inner(self._epsilon(velocity), self._epsilon(velocity)) + self.eps

        return self.A ** (-1. / self.n) * ((2. * self.n) / (1. + self.n)) * (
                    normEu12 ** ((1. + self.n) / (2. * self.n))) * dl.dx \
               - dl.inner(self.f, velocity) * dl.dx \
               + dl.Constant(.5) * dl.inner(dl.exp(m) * self._tang(velocity), self._tang(velocity)) * self.ds_base \
               + self.lam * dl.inner(velocity, self.normal) ** 2 * self.ds_base

    def constraint(self, u):
        vel, pressure = dl.split(u)
        return dl.inner(-dl.div(vel), pressure) * dl.dx

        # why is the constraint added to the variational form
        # this should not be of no consequence if the constraint
        # is satisfied exactly, but is it needed at all?

    def varf_handler(self, u, m, p):
        return dl.derivative(self.energy_fun(u, m) + self.constraint(u), u, p)  # + self.constraint(u)