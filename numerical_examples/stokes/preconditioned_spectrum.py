import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix
from localpsf.positive_definite_modifications import get_negative_eig_correction_factors, WoodburyObject

from localpsf.stokes_inverse_problem_cylinder import *
import dolfin as dl

import scipy.sparse.linalg as spla
import scipy.sparse as sps

save_data = True
save_figures = True

stokes_base_dir = get_project_root() / 'numerical_examples' / 'stokes'

save_dir = stokes_base_dir / 'preconditioned_spectrum'
save_dir.mkdir(parents=True, exist_ok=True)


# --------- set up the problem
# mfile_name = "meshes/cylinder_medium"
mfile_name = stokes_base_dir / 'meshes' / 'cylinder_medium'
mesh = dl.Mesh(str(mfile_name)+".xml")
boundary_markers = dl.MeshFunction("size_t", mesh, str(mfile_name)+"_facet_region.xml")


Newton_iterations = 2
nondefault_StokesIP_options = {'mesh' : mesh,'boundary_markers' : boundary_markers,
        'Newton_iterations': Newton_iterations,
        'misfit_only': True,
        'gauss_newton_approx': True,
        'load_fwd': False,
        'lam': 1.e10,
        'solve_inv': False,
        'gamma': 1.e4,
        'm0': 1.5*7.,
        'mtrue_string': 'm0 - (m0 / 7.)*std::cos(2.*x[0]*pi/Radius)'}

StokesIP = StokesInverseProblemCylinder(**nondefault_StokesIP_options)


########    OPTIONS    ########


num_neighbors = 10
tau          = 3.0 
gamma        = 1e-5
hmatrix_tol = 1e-4
all_batch_sizes = [3, 6, 9]

num_eigs = 500#10#min(1000, StokesIP.N)-10

########    BUILD PC-HMATRIX APPROXIMATIONS FOR A VARIETY OF BATCH SIZES    ########

all_Hd_hmatrix = list()
all_extras = list()
for num_batches in all_batch_sizes:
    PCK = ProductConvolutionKernel(StokesIP.V, StokesIP.V, StokesIP.apply_Hd_petsc, StokesIP.apply_Hd_petsc,
                                   num_batches, num_batches,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma,
                                   cols_only=True)
                                   # cols_only=False)

    # Hd_hmatrix, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=hmatrix_tol)
    Hd_hmatrix, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=False, hmatrix_tol=hmatrix_tol)
    Hd_hmatrix.sym(overwrite=True)

    all_Hd_hmatrix.append(Hd_hmatrix)
    all_extras.append(extras)


best_Hd_hmatrix = all_Hd_hmatrix[-1]

#

z = np.random.randn(all_Hd_hmatrix[0].shape[1])
y0 = all_Hd_hmatrix[0] * z
y1 = all_Hd_hmatrix[1] * z
y2 = all_Hd_hmatrix[2] * z
y = StokesIP.apply_H_Vsub_numpy(z)
e0 = np.linalg.norm(y-y0)/np.linalg.norm(y)
e1 = np.linalg.norm(y-y1)/np.linalg.norm(y)
e2 = np.linalg.norm(y-y2)/np.linalg.norm(y)
print('e0=', e0)
print('e1=', e1)
print('e2=', e2)

#

# ---- build the regularization operator for future compression
K = csr_fenics2scipy(StokesIP.priorVsub.A) # K is not just the stiffness matrix, R = K M^-1 K, K an elliptic PDE operator
# K coming from discretization of + \gamma \nabla u \cdot \nabla p + \delta u * p
M = csr_fenics2scipy(StokesIP.priorVsub.M)
Minv = sps.diags(1. / M.diagonal(), format="csr")
R = K.dot(Minv).dot(K)
# ----

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(R, best_Hd_hmatrix.bct)

#

ee_R, _ = spla.eigsh(R, which='SM')
min_eig_R = np.min(ee_R)

UU = []
EE = []
all_Hd_plus_linops = []
for Hd_hmatrix in all_Hd_hmatrix:
    Hd_hmatrix_linop = Hd_hmatrix.as_linear_operator()
    U, E, Hd_plus_linop = get_negative_eig_correction_factors(Hd_hmatrix_linop.shape, Hd_hmatrix_linop.matvec, min_eig_R/2.)
    UU.append(U)
    EE.append(E)
    all_Hd_plus_linops.append(Hd_plus_linop)


########    COMPUTE PRECONDITIONED SPECTRUM    ########

print('PCH preconditioning')
all_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
all_abs_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
for kk_batch in range(len(all_batch_sizes)):
    print('batch size=', all_batch_sizes[kk_batch])
    Hd_hmatrix = all_Hd_hmatrix[kk_batch]
    H_hmatrix = Hd_hmatrix + R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()
    U = UU[kk_batch]
    E = EE[kk_batch]
    H_hmatrix_linop = H_hmatrix.as_linear_operator()
    iH_hmatrix_linop = iH_hmatrix.as_linear_operator()

    WO = WoodburyObject(H_hmatrix_linop.matvec, iH_hmatrix_linop.matvec, U, E, U.T)
    delta_hmatrix_linop = spla.LinearOperator((StokesIP.N, StokesIP.N),
                                              matvec=lambda x: StokesIP.apply_H_Vsub_numpy(x) - WO.apply_modified_A(x))
    ee_hmatrix, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=WO.apply_modified_A_linop,
                               Minv=WO.solve_modified_A_linop, which='LM')
    # delta_hmatrix_linop = spla.LinearOperator((StokesIP.N, StokesIP.N), matvec=lambda x: StokesIP.apply_H_Vsub_numpy(x) - H_hmatrix * x)
    # ee_hmatrix, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=H_hmatrix.as_linear_operator(),
    #                            Minv=iH_hmatrix.as_linear_operator(), which='LM')
    # preconditioned_linop = spla.LinearOperator((StokesIP.N, StokesIP.N),
    #                                            matvec=lambda x: iH_hmatrix * StokesIP.apply_H_Vsub_numpy(x))
    # ee_hmatrix, _ = spla.eigs(preconditioned_linop, k=num_eigs, which='LM')
    abs_ee_hmatrix = np.sort(np.abs(ee_hmatrix))[::-1]

    all_ee_hmatrix[kk_batch, :] = ee_hmatrix
    all_abs_ee_hmatrix[kk_batch, :] = abs_ee_hmatrix

print('reg preconditioning')

Rinv_linop = spla.LinearOperator((StokesIP.N, StokesIP.N), matvec = lambda x: StokesIP.apply_Rinv_numpy(x))

delta_reg_linop = spla.LinearOperator((StokesIP.N, StokesIP.N), matvec=lambda x: StokesIP.apply_H_Vsub_numpy(x) - R.dot(x))
ee_reg, _ = spla.eigsh(delta_reg_linop, k=num_eigs, M=R, Minv=Rinv_linop, which='LM')
abs_ee_reg = np.sort(np.abs(ee_reg))[::-1]


########    SAVE DATA    ########

all_batch_sizes = np.array(all_batch_sizes)
abs_ee_hmatrix = np.array(abs_ee_hmatrix)
abs_ee_reg = np.array(abs_ee_reg)

if save_data:
    np.savetxt(str(save_dir) + '/all_batch_sizes.txt', all_batch_sizes)
    np.savetxt(str(save_dir) + '/abs_ee_hmatrix.txt', abs_ee_hmatrix)
    np.savetxt(str(save_dir) + '/abs_ee_reg.txt', abs_ee_reg)


########    MAKE FIGURE    ########

plt.figure()
plt.semilogy(abs_ee_reg)
for abs_ee_hmatrix in all_abs_ee_hmatrix:
    plt.semilogy(abs_ee_hmatrix)

plt.title(r'Absolute values of eigenvalues of $P^{-1}H-I$')
plt.xlabel(r'$i$')
plt.ylabel(r'$|\lambda_i|$')
plt.legend(['Reg'] + ['PCH' + str(nB) for nB in all_batch_sizes])


if save_figures:
    plt.savefig(str(save_dir) + '/preconditioned_spectrum.pdf', bbox_inches='tight', dpi=100)
plt.show()
plt.close()
