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

from scipy.optimize import root_scalar

noise_level = 0.01 # float(sys.argv[1])
gamma0 = 1.e5
gamma1 = 1.e6
initial_GN_iter = 3
GN_iter_after_initial = 5
max_newton_iter = 10
# mesh_type = 'coarse'
mesh_type = 'medium'
morozov_rtol = 5.e-2
rel_correlation_Length = 0.05

if mesh_type == 'coarse':
    mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_coarse"
    lam = 1e10
elif mesh_type == 'medium':
    mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_medium"
    lam = 1e11
elif mesh_type == 'fine':
    mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_medium"
    lam = 1e12
else:
    raise RuntimeError('invalid mesh type '+mesh_type+', valid types are coarse, medium, fine')



mesh = dl.Mesh(mfile_name+".xml")
boundary_markers = dl.MeshFunction("size_t", mesh, mfile_name+"_facet_region.xml")

nondefault_StokesIP_options = {'mesh': mesh, 'boundary_markers': boundary_markers,
                               'load_fwd': False,
                               'lam': lam, #1.e10,
                               'rel_correlation_Length': rel_correlation_Length,
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

nondefault_StokesIP_options['gamma'] = gamma0

StokesIP = StokesInverseProblemCylinder(**nondefault_StokesIP_options)

m0_original = dl.interpolate(dl.Constant(1.5 * 7.), StokesIP.Vh[1]).vector()
m0_numpy = m0_original[:]
m0 = dl.Function(StokesIP.Vh[1]).vector()

def solve_inverse_problem(gamma, use_PCH_precond=True):
    StokesIP.set_gamma(gamma)

    m0[:] = m0_numpy
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
    x = solver.solve(StokesIP.x)

    if solver.converged:
        print("\n Converged in ", solver.it, " iterations.")
    else:
        print("\n Not Converged")

    print("\n {0:d} cumulative preconditioned Krylov iterations".format(solver.total_cg_iter))

    return x, StokesIP, solver


def log_morozov_function(log_gamma):
    gamma = np.exp(log_gamma)
    print('gamma=', gamma)
    x, StokesIP, solver = solve_inverse_problem(gamma, use_PCH_precond=True)

    noise_datanorm = StokesIP.noise_datanorm
    morozov_discrepancy = np.sqrt(2.0 * StokesIP.model.misfit.cost(x))  # ||d - G(m_star)||_datanorm
    print('gamma=', gamma, ', noise_datanorm=', noise_datanorm, ', morozov_discrepancy=', morozov_discrepancy)
    return np.log(morozov_discrepancy) - np.log(noise_datanorm)


sol = root_scalar(log_morozov_function, x0=np.log(gamma0), x1=np.log(gamma1), rtol=morozov_rtol)

gamma_morozov = np.exp(sol.root)

print('gamma_morozov=', gamma_morozov)

mReconstruction = dl.Function(StokesIP.V)
mTrue = dl.Function(StokesIP.V)
StokesIP.prior.P.mult(StokesIP.x[1], mReconstruction.vector())
StokesIP.prior.P.mult(StokesIP.mtrue, mTrue.vector())
dl.File("data/mReconstruction_noise" + str(noise_level) + ".pvd") << mReconstruction
dl.File("data/mTrue.pvd") << mTrue

stateObserved = dl.Function(StokesIP.Vh[0], StokesIP.model.misfit.d)
stateReconstruction = dl.Function(StokesIP.Vh[0], StokesIP.x[0])
dl.File("data/uObs_noise" + str(noise_level) + ".pvd") << stateObserved.sub(0)
dl.File("data/uReconstruction_noise" + str(noise_level) + ".pvd") << stateReconstruction.sub(0)
dl.File("data/pReconstruction_noise" + str(noise_level) + ".pvd") << stateReconstruction.sub(1)


# PCH 3,4,3
# Newton CG convergence information:
# It  cg_it cost            misfit          reg             (g,dm)          ||g||L2        alpha          tolcg
#   1   1    1.430241e+09    1.412300e+09    1.794024e+07   -6.030297e+09   5.135519e+08   5.000000e-01   5.000000e-01
#   2   2    8.255355e+08    7.867917e+08    3.874385e+07   -1.278585e+09   3.840691e+08   1.000000e+00   5.000000e-01
#   3   1    7.169226e+08    6.815608e+08    3.536182e+07   -2.168616e+08   3.214223e+08   1.000000e+00   5.000000e-01
#   4   2    4.361433e+08    3.041152e+08    1.320281e+08   -7.152277e+08   1.426738e+08   1.000000e+00   5.000000e-01
#   5   1    3.451799e+08    2.203775e+08    1.248024e+08   -1.824239e+08   2.097923e+08   1.000000e+00   5.000000e-01
#   6   1    3.183553e+08    2.046801e+08    1.136751e+08   -5.228334e+07   7.969363e+07   1.000000e+00   3.939305e-01
#   7   3    3.044090e+08    1.935796e+08    1.108294e+08   -2.789136e+07   3.685330e+07   1.000000e+00   2.678835e-01
#   8   5    3.031538e+08    1.928697e+08    1.102841e+08   -2.425432e+06   1.280161e+07   1.000000e+00   1.578847e-01
#   9   6    3.031344e+08    1.930294e+08    1.101051e+08   -4.744807e+04   1.604268e+06   1.000000e+00   5.589157e-02
#  10   8    3.031308e+08    1.930296e+08    1.101012e+08   -4.942153e+01   3.965288e+04   1.000000e+00   8.787092e-03

# Solve with regularization preconditioning
x2, StokesIP2, solver2 = solve_inverse_problem(gamma_morozov, use_PCH_precond=True)

# PCH 3,5,2
# It  cg_it cost            misfit          reg             (g,dm)          ||g||L2        alpha          tolcg
#   1   1    1.433183e+09    1.414280e+09    1.890333e+07   -6.022196e+09   5.135519e+08   5.000000e-01   5.000000e-01
#   2   2    8.105208e+08    7.692901e+08    4.123073e+07   -1.294907e+09   3.844680e+08   1.000000e+00   5.000000e-01
#   3   3    5.792979e+08    5.277201e+08    5.157777e+07   -4.485286e+08   3.112426e+08   1.000000e+00   5.000000e-01
#   4   2    3.693364e+08    2.415136e+08    1.278228e+08   -4.475060e+08   1.338754e+08   1.000000e+00   5.000000e-01
#   5   1    3.317419e+08    2.077879e+08    1.239541e+08   -7.453887e+07   1.059505e+08   1.000000e+00   4.542127e-01
#   6   2    3.119247e+08    1.971395e+08    1.147852e+08   -3.884419e+07   4.504598e+07   1.000000e+00   2.961664e-01
#   7   4    3.093800e+08    1.943120e+08    1.150681e+08   -5.035769e+06   1.387791e+07   1.000000e+00   1.643879e-01
#   8   5    3.093204e+08    1.944013e+08    1.149191e+08   -1.316938e+05   2.089865e+06   1.000000e+00   6.379211e-02
#   9   9    3.093162e+08    1.943774e+08    1.149388e+08   -9.515372e+02   1.291263e+05   1.000000e+00   1.585678e-02
#  10   9    3.093161e+08    1.943774e+08    1.149387e+08   -1.517595e+00   3.645396e+03   1.000000e+00   2.664282e-03

# Solve with regularization preconditioning
x3, StokesIP3, solver3 = solve_inverse_problem(gamma_morozov, use_PCH_precond=False)

# Reg 3,4,3
# Newton CG convergence information:
# It  cg_it cost            misfit          reg             (g,dm)          ||g||L2        alpha          tolcg
#   1   1    1.433183e+09    1.414280e+09    1.890333e+07   -6.022196e+09   5.135519e+08   5.000000e-01   5.000000e-01
#   2   2    8.105208e+08    7.692901e+08    4.123073e+07   -1.294907e+09   3.844680e+08   1.000000e+00   5.000000e-01
#   3   3    5.792979e+08    5.277201e+08    5.157777e+07   -4.485286e+08   3.112426e+08   1.000000e+00   5.000000e-01
#   4   6    3.640090e+08    2.527819e+08    1.112271e+08   -4.397427e+08   1.338754e+08   1.000000e+00   5.000000e-01
#   5   3    3.225688e+08    2.198049e+08    1.027639e+08   -8.155774e+07   9.858503e+07   1.000000e+00   4.381404e-01
#   6  13    3.096641e+08    1.944937e+08    1.151704e+08   -2.573158e+07   2.531926e+07   1.000000e+00   2.220411e-01
#   7  16    3.093235e+08    1.948493e+08    1.144743e+08   -5.996989e+05   5.023445e+06   1.000000e+00   9.890282e-02
#   8  25    3.093208e+08    1.943830e+08    1.149379e+08   -2.292072e+04   6.027039e+05   1.000000e+00   3.425783e-02
#   9  30    3.093149e+08    1.943758e+08    1.149391e+08   -1.573540e+01   1.851706e+04   1.000000e+00   6.004734e-03
#  10  37    3.093143e+08    1.943758e+08    1.149385e+08   -1.878599e+00   3.330189e+03   1.000000e+00   2.546492e-03


# gamma0 = 1e5 # 1e5 is pretty good
#
# x, StokesIP, solver = solve_inverse_problem(gamma0, use_PCH_precond=True)
#
# noise_datanorm = StokesIP.noise_datanorm
# morozov_discrepancy = np.sqrt(2.0*StokesIP.model.misfit.cost(x)) # ||d - G(m_star)||_datanorm
# print('noise_datanorm=', noise_datanorm, ', morozov_discrepancy=', morozov_discrepancy)







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









