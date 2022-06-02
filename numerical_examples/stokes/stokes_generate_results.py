#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix
from localpsf import localpsf_root


from stokes_inverse_problem_cylinder import *
# from .stokes_inverse_problem_cylinder import *
import dolfin as dl

import scipy.sparse.linalg as spla
from NewtonCGPCH import *
from hippylib import nb
import sys

noise_level = 0.01 # float(sys.argv[1])
gamma_min = 1.e3
gamma_max = 1.e5
initial_GN_iter = 3
GN_iter_after_initial = 4
max_newton_iter = 10
mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_coarse"
# mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_medium"



mesh = dl.Mesh(mfile_name+".xml")
boundary_markers = dl.MeshFunction("size_t", mesh, mfile_name+"_facet_region.xml")

nondefault_StokesIP_options = {'mesh': mesh, 'boundary_markers': boundary_markers,
                               'load_fwd': False,
                               'lam': 1.e10,
                               'noise_level': noise_level,
                               'm0': 1.5 * 7.,
                               'mtrue_string': 'm0 - (m0 / 7.)*std::cos(2.*x[0]*pi/Radius)'}

parameters = ReducedSpaceNewtonCG_ParameterList()
parameters["rel_tolerance"] = 1.e-8
parameters["abs_tolerance"] = 1.e-12
parameters["max_iter"] = max_newton_iter
parameters["globalization"] = "LS"
parameters["cg_coarse_tolerance"] = 0.5
parameters["initial_iter"] = initial_GN_iter
parameters["GN_iter"] = GN_iter_after_initial
parameters["print_level"] = 2

def solve_inverse_problem(gamma, use_PCH_precond=True):
    nondefault_StokesIP_options['gamma'] = gamma

    StokesIP = StokesInverseProblemCylinder(**nondefault_StokesIP_options)
    m0 = dl.interpolate(dl.Constant(1.5 * 7.), StokesIP.Vh[1]).vector()
    StokesIP.set_parameter(m0)

    K = csr_fenics2scipy(StokesIP.priorVsub.A)
    M = csr_fenics2scipy(StokesIP.priorVsub.M)
    Minv = sps.diags(1. / M.diagonal(), format="csr")
    R = K.dot(Minv).dot(K)

    parameters["projector"] = StokesIP.prior.P
    parameters["IP"] = StokesIP
    parameters["Rscipy"] = R
    parameters["Pscipy"] = csr_fenics2scipy(StokesIP.prior.P)
    parameters["PCH_precond"] = use_PCH_precond

    solver = ReducedSpaceNewtonCG(StokesIP.model, parameters)
    m_star = solver.solve(StokesIP.x)

    if solver.converged:
        print("\n Converged in ", solver.it, " iterations.")
    else:
        print("\n Not Converged")

    print("\n {0:d} cumulative preconditioned Krylov iterations".format(solver.total_cg_iter))

    return m_star, StokesIP, solver

gamma0 = 1e5 # 1e5 is pretty good

x, StokesIP, solver = solve_inverse_problem(gamma0, use_PCH_precond=True)

noise_datanorm = StokesIP.noise_datanorm
morozov_discrepancy = np.sqrt(2.0*StokesIP.model.misfit.cost(x)) # ||d - G(m_star)||_datanorm
print('noise_datanorm=', noise_datanorm, ', morozov_discrepancy=', morozov_discrepancy)


mReconstruction = dl.Function(StokesIP.V)
mTrue = dl.Function(StokesIP.V)
StokesIP.prior.P.mult(x[1], mReconstruction.vector())
StokesIP.prior.P.mult(StokesIP.mtrue, mTrue.vector())
dl.File("data/mReconstruction_noise" + str(noise_level) + ".pvd") << mReconstruction
dl.File("data/mTrue.pvd") << mTrue

stateObserved = dl.Function(StokesIP.Vh[0], StokesIP.model.misfit.d)
stateReconstruction = dl.Function(StokesIP.Vh[0], x[0])
dl.File("data/uObs_noise" + str(noise_level) + ".pvd") << stateObserved.sub(0)
dl.File("data/uReconstruction_noise" + str(noise_level) + ".pvd") << stateReconstruction.sub(0)
dl.File("data/pReconstruction_noise" + str(noise_level) + ".pvd") << stateReconstruction.sub(1)




# d = G(m_true) + eta
#   d = data
#   m_true = true model parameter
#   G = parameter to observable map
#   eta = noise
#
# cost(m) = 1/2 ||d - G(m)||^2
#   ||eta|| <= c, c is known
#
# set of valid m:
#   S = {m: ||d - G(m)|| <= c}
# 0 = ||d - G(m_true) - eta|| <= ||d - G(m_true)||
#
# min_m Reg(m)
# s.t. m \in S
#
# min_m Reg(m)
# s.t. ||d - G(m)||^2 = c^2
#
# min_m max_\lambda L(m,lambda)
# L = Reg(m) + lambda/2 (||d - G(m)||^2 - c^2)
#
# Choose lambda such that:
#    ||d - G(m)|| = c #Morozov discrepancy principle
#
# m(\alpha) :=  argmin_m 1/2 ||d - G(m)||_X^2 + \alpha/2 ||Rm||_Y^2
# evaluating m(\alpha) means solving optimization problem for given alpha
# solve for \alpha in following equation:
#   ||d - G(m(\alpha))|| = c
#
# \alpha = 100
# while True:
#     m_alpha = argmin_m 1/2 ||d - G(m)||_X^2 + \alpha/2 ||Rm||_Y^2
#     morozov_discrepancy = ||d - G(m(\alpha))||
#     if morozov_discrepancy < c:
#         break
#     alpha = alpha/2
#
# ||eta||_X = c = noise_level * ||d||_X
# norm_data = ||d||_X
# noise0 = np.random.randn(num_param)        # has X-norm not equal 1
# noise1 = noise0 / ||noise0||_X             # has X-norm=1
# noise = (noise_level / norm_data) * noise1 # has ||noise||_X = noise_level * ||d||_X









