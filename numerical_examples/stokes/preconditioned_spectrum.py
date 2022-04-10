import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix

from localpsf.stokes_inverse_problem_cylinder import *
import dolfin as dl

import scipy.sparse.linalg as spla
import scipy.sparse as sps

save_data = True
save_figures = True


# import os
# root_dir = os.path.abspath(os.curdir)
# rel_save_dir = './preconditioned_spectrum/'
# save_dir = os.path.join(root_dir, rel_save_dir)
# os.makedirs(save_dir, exist_ok=True)

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

# nondefault_StokesIP_options = {'mesh' : mesh,'boundary_markers' : boundary_markers,
#         'misfit_only': True,
#         'gauss_newton_approx': True,
#         # 'load_fwd': True,
#         'load_fwd': False,
#         'lam': 1.e10,
#         'solve_inv': False,
#         'gamma': 1.e4,
#         'm0': 1.5*7.,
#         'mtrue_string': 'm0 - (m0 / 7.)*std::cos((x[0]*x[0]+x[1]*x[1])*pi/(Radius*Radius))'}
#
# StokesIP = StokesInverseProblemCylinder(**nondefault_StokesIP_options)


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

PCH = all_Hd_hmatrix[0]
max_eig = spla.eigsh(PCH.as_linear_operator(), 1, which='LM')[0][0]
shifted_PCH_linop = spla.LinearOperator(PCH.shape, matvec=lambda x: max_eig * x - PCH * x)
min_eig = max_eig - spla.eigsh(shifted_PCH_linop, 1, which='LM')[0][0]
print('A.sym(): lambda_min=', min_eig, ', lambda_max=', max_eig)

U = np.zeros((PCH.shape[0], 0))
negative_eigs = np.array([])
E = np.diag(-2*negative_eigs)

PCH_plus_linop = spla.LinearOperator(PCH.shape, matvec=lambda x: PCH * x + np.dot(U, np.dot(E, np.dot(U.T,x))))

block_size=10
min_eigs, min_evecs = spla.eigsh(PCH_plus_linop, block_size, which='SA')

negative_inds = min_eigs<0
new_negative_eigs = min_eigs[negative_inds]
new_negative_evecs = min_evecs[:,negative_inds]

U = np.hstack([U, new_negative_evecs])
negative_eigs = np.concatenate([negative_eigs, new_negative_eigs])
E = np.diag(-2*negative_eigs)

PCH_plus_linop = spla.LinearOperator(PCH.shape, matvec=lambda x: PCH * x + np.dot(U, np.dot(E, np.dot(U.T,x))))

#

min_eigs, min_evecs = spla.eigsh(PCH_plus_linop, block_size, which='SA')

negative_inds = min_eigs<0
new_negative_eigs = min_eigs[negative_inds]
new_negative_evecs = min_evecs[:,negative_inds]

U = np.hstack([U, new_negative_evecs])
negative_eigs = np.concatenate([negative_eigs, new_negative_eigs])
E = np.diag(-2*negative_eigs)

PCH_plus_linop = spla.LinearOperator(PCH.shape, matvec=lambda x: PCH * x + np.dot(U, np.dot(E, np.dot(U.T,x))))

#

min_eigs, min_evecs = spla.eigsh(PCH_plus_linop, block_size, which='SA')

negative_inds = min_eigs<0
new_negative_eigs = min_eigs[negative_inds]
new_negative_evecs = min_evecs[:,negative_inds]

U = np.hstack([U, new_negative_evecs])
negative_eigs = np.concatenate([negative_eigs, new_negative_eigs])
E = np.diag(-2*negative_eigs)

PCH_plus_linop = spla.LinearOperator(PCH.shape, matvec=lambda x: PCH * x + np.dot(U, np.dot(E, np.dot(U.T,x))))

#

min_eigs, min_evecs = spla.eigsh(PCH_plus_linop, block_size, which='SA')

negative_inds = min_eigs<0
new_negative_eigs = min_eigs[negative_inds]
new_negative_evecs = min_evecs[:,negative_inds]

U = np.hstack([U, new_negative_evecs])
negative_eigs = np.concatenate([negative_eigs, new_negative_eigs])
E = np.diag(-2*negative_eigs)

PCH_plus_linop = spla.LinearOperator(PCH.shape, matvec=lambda x: PCH * x + np.dot(U, np.dot(E, np.dot(U.T,x))))

#

PCH_plus_matvec = lambda x: PCH * x + np.dot(U, np.dot(E, np.dot(U.T,x)))

shifted_PCH_linop = spla.LinearOperator(PCH.shape, matvec=lambda x: max_eig * x - PCH_plus_matvec(x))

min_eigs_shifted, min_evecs = spla.eigsh(shifted_PCH_linop, block_size, which='LM')
min_eigs = max_eig - min_eigs_shifted

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

def get_negative_eig_correction_factors(A_linop, cutoff, block_size=20, maxiter=50, display=True):
    cutoff = -np.abs(cutoff)

    U = np.zeros((A_linop.shape[0], 0))
    negative_eigs = np.array([])
    E = np.diag(-2 * negative_eigs)

    for k in range(maxiter):
        A2_linop = spla.LinearOperator(A_linop.shape, matvec=lambda x: A_linop.matvec(x) + np.dot(U, np.dot(E, np.dot(U.T, x))))

        min_eigs, min_evecs = spla.eigsh(A2_linop, block_size, which='SA')

        negative_inds = min_eigs < 0
        new_negative_eigs = min_eigs[negative_inds]
        new_negative_evecs = min_evecs[:, negative_inds]

        U = np.hstack([U, new_negative_evecs])
        negative_eigs = np.concatenate([negative_eigs, new_negative_eigs])
        E = np.diag(-2 * negative_eigs)

        if display:
            print('k=', k, 'min_eigs=', min_eigs, 'cutoff=', cutoff)
        if (np.max(min_eigs) > cutoff):
            print('negative eigs smaller than cutoff. Good.')
            break

    return U, E, A2_linop

ee_R, _ = spla.eigsh(R, which='SM')
min_eig_R = np.min(ee_R)

UU = []
EE = []
all_Hd_plus_linops = []
for Hd_hmatrix in all_Hd_hmatrix:
    U, E, Hd_plus_linop = get_negative_eig_correction_factors(Hd_hmatrix.as_linear_operator(), min_eig_R/2.)
    UU.append(U)
    EE.append(E)
    all_Hd_plus_linops.append(Hd_plus_linop)

# def woodbury_solve(b, apply_A, solve_A, U, C, V): # (A+UCV)x = b
#     y = solve_A(b)
#     return y - solve_A(np.dot(U,))




# Hd_array = np.zeros((StokesIP.N, StokesIP.N))
# for ii in tqdm(range(Hd_array.shape[1])):
#     ei = np.zeros(Hd_array.shape[1])
#     ei[ii] = 1.0
#     Hd_array[:,ii] = StokesIP.apply_H_Vsub_numpy(ei)

PCH3 = np.zeros((StokesIP.N, StokesIP.N))
for ii in tqdm(range(PCH3.shape[1])):
    ei = np.zeros(PCH3.shape[1])
    ei[ii] = 1.0
    PCH3[:,ii] = all_Hd_hmatrix[2] * ei

ee, P = np.linalg.eigh(PCH3)
# ee_plus = ee.copy()
# ee_plus[ee_plus < 0.] = 0.
ee_plus = np.abs(ee) # Best
PCH3_plus = np.dot(P, np.dot(np.diag(ee_plus), P.T))

R_array = R.toarray()

M = PCH3_plus + R_array
iM = np.linalg.inv(M)

M_linop = spla.LinearOperator((StokesIP.N, StokesIP.N), matvec=lambda x: np.dot(M, x))
iM_linop = spla.LinearOperator((StokesIP.N, StokesIP.N), matvec=lambda x: np.dot(iM, x))

delta_hmatrix_linop = spla.LinearOperator((StokesIP.N, StokesIP.N),
                                          matvec=lambda x: StokesIP.apply_H_Vsub_numpy(x) - M_linop(x))
ee_plus3, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=M_linop, Minv=iM_linop, which='LM')

########    COMPUTE PRECONDITIONED SPECTRUM    ########

print('PCH preconditioning')
all_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
all_abs_ee_hmatrix = np.zeros((len(all_batch_sizes), num_eigs))
for kk_batch in range(len(all_batch_sizes)):
    print('batch size=', all_batch_sizes[kk_batch])
    Hd_hmatrix = all_Hd_hmatrix[kk_batch]
    H_hmatrix = Hd_hmatrix + R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()
    delta_hmatrix_linop = spla.LinearOperator((StokesIP.N, StokesIP.N), matvec=lambda x: StokesIP.apply_H_Vsub_numpy(x) - H_hmatrix * x)
    # ee_hmatrix, _ = spla.eigsh(delta_hmatrix_linop, k=num_eigs, M=H_hmatrix.as_linear_operator(),
    #                            Minv=iH_hmatrix.as_linear_operator(), which='LM')
    preconditioned_linop = spla.LinearOperator((StokesIP.N, StokesIP.N),
                                               matvec=lambda x: iH_hmatrix * StokesIP.apply_H_Vsub_numpy(x))
    ee_hmatrix, _ = spla.eigs(preconditioned_linop, k=num_eigs, which='LM')
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
