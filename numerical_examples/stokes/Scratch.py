#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
def solveInv(noise_level):
    # gamma = 1.e5 #* np.sqrt(noise_level / 5.e-2)
    gamma = 1.e4
    # --------- set up the problem
    # mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_coarse"
    mfile_name = str(localpsf_root) + "/numerical_examples/stokes/meshes/cylinder_medium"
    mesh = dl.Mesh(mfile_name+".xml")
    boundary_markers = dl.MeshFunction("size_t", mesh, mfile_name+"_facet_region.xml")
    nondefault_StokesIP_options = {'mesh' : mesh,'boundary_markers' : boundary_markers,
        'load_fwd': False,
        'lam': 1.e10,
        'gamma': gamma,
        'noise_level': noise_level,
        'm0': 1.5*7.,
        'mtrue_string': 'm0 - (m0 / 7.)*std::cos(2.*x[0]*pi/Radius)'}
    StokesIP = StokesInverseProblemCylinder(**nondefault_StokesIP_options)
    m0 = dl.interpolate(dl.Constant(1.5*7.), StokesIP.Vh[1]).vector()
    StokesIP.set_parameter(m0)
    rtol = 1.e-8
    atol = 1.e-12
    Newton_iterations = 12 #2 # 50
    initial_iterations = 3
    GN_iterations = 5
    cg_coarse_tolerance = 0.5
    parameters = ReducedSpaceNewtonCG_ParameterList()
    parameters["rel_tolerance"] = rtol
    parameters["abs_tolerance"] = atol
    parameters["max_iter"]      = Newton_iterations
    parameters["globalization"] = "LS"
    parameters["cg_coarse_tolerance"] = cg_coarse_tolerance
    parameters["initial_iter"]  = initial_iterations
    parameters["GN_iter"]       = GN_iterations
    parameters["projector"]     = StokesIP.prior.P
    parameters["IP"]            = StokesIP
    # ===== to do build R scipy to feed to parameters of NewtonCG solver
    # ---- build the regularization operator for future compression
    K = csr_fenics2scipy(StokesIP.priorVsub.A)
    M = csr_fenics2scipy(StokesIP.priorVsub.M)
    Minv = sps.diags(1. / M.diagonal(), format="csr")
    R = K.dot(Minv).dot(K)
    # ----
    parameters["Rscipy"] = R
    parameters["Pscipy"] = csr_fenics2scipy(StokesIP.prior.P)
    # parameters["PCH_precond"] = True
    parameters["PCH_precond"] = False
    parameters["print_level"] = 2
    solver = ReducedSpaceNewtonCG(StokesIP.model, parameters)
    x   = solver.solve(StokesIP.x)
    StokesIP.misfit.cost(x)
    if solver.converged:
        print("\n Converged in ", solver.it, " iterations.")
    else:
        print("\n Not Converged")
    print("\n {0:d} cumulative preconditioned Krylov iterations".format(solver.total_cg_iter))
    mReconstruction = dl.Function(StokesIP.V)
    mTrue           = dl.Function(StokesIP.V)
    StokesIP.prior.P.mult(x[1], mReconstruction.vector())
    StokesIP.prior.P.mult(StokesIP.mtrue, mTrue.vector())
    dl.File("data/mReconstruction_noise"+str(noise_level)+".pvd") << mReconstruction
    dl.File("data/mTrue.pvd") << mTrue

    stateObserved = dl.Function(StokesIP.Vh[0], StokesIP.model.misfit.d)
    stateReconstruction = dl.Function(StokesIP.Vh[0], x[0])
    dl.File("data/uObs_noise"+str(noise_level)+".pvd") << stateObserved.sub(0)
    dl.File("data/uReconstruction_noise"+str(noise_level)+".pvd") << stateReconstruction.sub(0)
    dl.File("data/pReconstruction_noise"+str(noise_level)+".pvd") << stateReconstruction.sub(1)






noise_level = 0.01 # float(sys.argv[1])
solveInv(noise_level)


