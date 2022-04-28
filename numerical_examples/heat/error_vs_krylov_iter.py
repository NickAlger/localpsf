import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from time import time

import hlibpro_python_wrapper as hpro
from nalger_helper_functions import *
from localpsf.heat_inverse_problem import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix
from localpsf.visualization import column_error_plot
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter

import scipy.sparse.linalg as spla
import scipy.linalg as sla


save_data = True
save_figures = True

save_dir = get_project_root() / 'numerical_examples' / 'heat' / 'error_vs_krylov_iter'
save_dir.mkdir(parents=True, exist_ok=True)


########    OPTIONS    ########

nondefault_HIP_options = {'mesh_h': 2.5e-2} # {'mesh_h': 2e-2}

tau = 2.5
all_num_batches = [1,3,6,9]
num_neighbors = 10
gamma = 1e-5
sigma_min = 1e-6 # 1e-1 # minimum width of support ellipsoid
hmatrix_tol=1e-4


########    SET UP HEAT INVERSE PROBLEM    ########

HIP = HeatInverseProblem(**nondefault_HIP_options)

########    CONSTRUCT KERNEL    ########

PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                               10, 10,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma)

########    CREATE HMATRICES    ########

A_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)

# _, eeA_max, _ = spla.svds(A_pch_nonsym.as_linear_operator(), k=3, which='LM')
# eA_max = np.max(eeA_max)

# cutoff = -1e-2*eA_max

#########################################

A_hmatrix = A_pch_nonsym

a_factor=2.0
b_factor=0.0
k=5
rtol = hpro.default_rtol
atol = hpro.default_atol
# rtol = 1e-5 # 1e-14
# atol = 1e-8 # 1e-14
display_progress=True

if display_progress:
    print('making hmatrix spd')
    print('symmetrizing')
A_hmatrix = A_hmatrix.sym()

if display_progress:
    print('getting largest eigenvalue with Lanczos')
ee_LM, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='LM')
lambda_max = np.max(ee_LM)
if display_progress:
    print('lambda_max=', lambda_max)

if display_progress:
    print('getting smallest eigenvalue with Lanczos')
ee_SA, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='SA')
lambda_min = np.min(ee_SA)
if display_progress:
    print('lambda_min=', lambda_min)

B = (A_hmatrix * (0.5/np.abs(lambda_min))).add_identity().inv(rtol=rtol, atol=atol)
# B = (A_hmatrix * (0.5/np.abs(lambda_min)))
# B.visualize('BBB')

ee_B, _ = spla.eigsh(B.as_linear_operator(), k=1, which='LA')
ee_B

if display_progress:
    print('Setting up operator T = (2*A - (b+a) I) / (b-a)')
a = 1.0
b = 4.0
# a = lambda_min * a_factor
# b = lambda_min * b_factor

if display_progress:
    scaling_at_one = 1. / (1. + ((2 - (b + a)) / (b - a)) ** (2 ** k))
    print('scaling_at_one=', scaling_at_one)
    # scaling_at_zero = 1. / (1. + ((b + a) / (b - a)) ** (2 ** k))
    # print('scaling_at_zero=', scaling_at_zero)

# T = A_hmatrix.copy()
T = B.copy()
T = (T * 2.0).add_identity(s=-(b + a)) * (1.0 / (b - a))

ee_T, _ = spla.eigsh(T.as_linear_operator(), k=1, which='LM')
T_scaling_factor = np.max(np.abs(ee_T))
if display_progress:
    print('T_scaling_factor=', T_scaling_factor)

T_hat = T * (1. / T_scaling_factor)

if display_progress:
    print('computing T^(2^k)')
for ii in range(k):
    if display_progress:
        print('computing T^(2^' + str(ii + 1) + ') = T^(2^' + str(ii) + ') * T^(2^' + str(ii) + ')')
    T_hat = hpro.h_mul(T_hat, T_hat,
              rtol=rtol, atol=atol,
              display_progress=display_progress)
    T_scaling_factor = T_scaling_factor*T_scaling_factor

T = T_hat * T_scaling_factor

if display_progress:
    print('computing spectral projector Pi = I / (I + T^(2^k))')
Pi = T.add_identity().inv(rtol=rtol, atol=atol, display_progress=display_progress)

if display_progress:
    print('computing A_minus = Pi * A')
A_minus = hpro.h_mul(Pi, A_hmatrix,
                     rtol=rtol, atol=atol,
                     display_progress=display_progress).sym()

A_plus = A_hmatrix + (A_minus * -1.0)
# A_plus = A_hmatrix + (A_minus * -2.0)

T_dense = build_dense_matrix_from_matvecs(T.matvec, T.shape[1])

A_plus_dense = build_dense_matrix_from_matvecs(A_plus.matvec, A_plus.shape[1])
ee_plus, uu_plus = sla.eigh(A_plus_dense)
print('np.min(ee_plus)=', np.min(ee_plus))

plt.figure()
plt.plot(ee_plus)

#

A_pch0 = A_pch_nonsym.sym()
A0_dense = build_dense_matrix_from_matvecs(A_pch0.matvec, A_pch0.shape[1])
ee0, uu0 = sla.eigh(A0_dense)
plt.plot(ee0)

#

ee_T, _ = sla.eigh(T_dense)

#

class LinopWithCounter:
    def __init__(me, linop):
        me.counter = 0
        me.linop = linop
        me.shape = linop.shape
        me.dtype = linop.dtype

    def matvec(me, x):
        me.counter+=1
        print('counter=', me.counter)
        return me.linop.matvec(x)

#

A2 = hpro.h_mul(A_hmatrix, A_hmatrix,
                rtol=rtol, atol=atol,
                display_progress=display_progress)

A4 = hpro.h_mul(A2, A2,
                rtol=rtol, atol=atol,
                display_progress=display_progress)

A8 = hpro.h_mul(A4, A4,
                rtol=rtol, atol=atol,
                display_progress=display_progress)

ee2, uu2 = spla.eigsh(LinopWithCounter(A2.as_linear_operator()), k=20, which='SA')
ee4, uu4 = spla.eigsh(LinopWithCounter(A4.as_linear_operator()), k=20, which='SA')
ee8, uu8 = spla.eigsh(LinopWithCounter(A8.as_linear_operator()), k=20, which='SA')

#

rtol=1e-4
atol=1e-6

if display_progress:
    print('getting smallest eigenvalue with Lanczos')
ee_SA, _ = spla.eigsh(A_hmatrix.as_linear_operator(), k=1, which='SA')
lambda_min = np.min(ee_SA)
if display_progress:
    print('lambda_min=', lambda_min)

B = (A_hmatrix * (0.5/np.abs(lambda_min))).add_identity().inv(rtol=rtol, atol=atol)

ee_B, _ = spla.eigsh(B.as_linear_operator(), k=1, which='LA')
print('ee_B=', ee_B)

B2 = hpro.h_mul(B, B,
                rtol=rtol, atol=atol,
                display_progress=display_progress)

B4 = hpro.h_mul(B2, B2,
                rtol=rtol, atol=atol,
                display_progress=display_progress)

B8 = hpro.h_mul(B4, B4,
                rtol=rtol, atol=atol,
                display_progress=display_progress)

ee1, uu1 = spla.eigsh(LinopWithCounter(B.as_linear_operator()), k=100, which='LM')
ee2, uu2 = spla.eigsh(LinopWithCounter(B2.as_linear_operator()), k=100, which='LM')
ee4, uu4 = spla.eigsh(LinopWithCounter(B4.as_linear_operator()), k=100, which='LM')
ee8, uu8 = spla.eigsh(LinopWithCounter(B8.as_linear_operator()), k=100, which='LM')

#########################################

# A_pch = A_pch_nonsym.spd(rtol=1e-3, atol=1e-5)
A_pch0 = A_pch_nonsym.sym()
A_pch3 = A_pch_nonsym.spd(k=3, a_factor=0.0)
A_pch5 = A_pch_nonsym.spd(k=5, a_factor=0.0)
A_pch7 = A_pch_nonsym.spd(k=7, a_factor=0.0)
A_pch7 = A_pch_nonsym.spd(k=7, a_factor=0.0, rtol=1e-3, atol=1e-3)

A_pch = A_pch7

A0_dense = build_dense_matrix_from_matvecs(A_pch0.matvec, A_pch0.shape[1])
A3_dense = build_dense_matrix_from_matvecs(A_pch3.matvec, A_pch3.shape[1])
A5_dense = build_dense_matrix_from_matvecs(A_pch5.matvec, A_pch5.shape[1])
A7_dense = build_dense_matrix_from_matvecs(A_pch7.matvec, A_pch7.shape[1])


ee0, uu0 = sla.eigh(A0_dense)
ee3, uu3 = sla.eigh(A3_dense)
ee5, uu5 = sla.eigh(A5_dense)
ee7, uu7 = sla.eigh(A7_dense)

print('np.min(ee0)=', np.min(ee0))
print('np.min(ee3)=', np.min(ee3))
print('np.min(ee5)=', np.min(ee5))
print('np.min(ee7)=', np.min(ee7))

plt.figure()
plt.plot(ee0)
plt.plot(ee3)
plt.plot(ee5)
plt.plot(ee7)
plt.legend(['ee0', 'ee3', 'ee5', 'ee7'])

# min_reg_param = 1e-3
# eeR_min, _ = spla.eigsh(min_reg_param * HIP.R0_scipy, k=3, which='SM')
# eR_min = np.min(eeR_min)
#
# A_pch = A_pch_nonsym.spd(cutoff=0.8*eR_min)


########    SET UP REGULARIZATION OPERATOR AND INVERSE PROBLEM SOLVER    ########

R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(HIP.R0_scipy, A_pch.bct)

def solve_inverse_problem(a_reg):
    HIP.regularization_parameter = a_reg

    g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
    H_hmatrix = A_pch + a_reg * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                          M=iH_hmatrix.as_linear_operator(),
                                          tol=1e-10, maxiter=500)

    u0 = dl.Function(HIP.V)
    u0.vector()[:] = u0_numpy

    return u0.vector()  # dolfin Vector


########    FIND REGULARIZATION PARAMETER VIA MOROZOV DISCREPANCY PRINCIPLE    ########

a_reg_morozov = compute_morozov_regularization_parameter(solve_inverse_problem,
                                                         HIP.morozov_discrepancy,
                                                         HIP.noise_Mnorm)



########    SOLVE INVERSE PROBLEM VIA CG WITH DIFFERENT PRECONDITIONERS    ########

HIP.regularization_parameter = a_reg_morozov

g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
H_hmatrix = A_pch + a_reg_morozov * R0_hmatrix
iH_hmatrix = H_hmatrix.inv()

u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                      M=iH_hmatrix.as_linear_operator(),
                                      tol=1e-11, maxiter=500)

_, _, residuals_None, errors_None = custom_cg(HIP.H_linop, -g0_numpy,
                                              x_true=u0_numpy,
                                              tol=1e-10, maxiter=500)

_, _, residuals_Reg, errors_Reg = custom_cg(HIP.H_linop, -g0_numpy,
                                            x_true=u0_numpy,
                                            M=HIP.solve_R_linop,
                                            tol=1e-10, maxiter=1000)


all_residuals_PCH = list()
all_errors_PCH = list()
for num_batches in all_num_batches:
    print('num_batches=', num_batches)
    PCK = ProductConvolutionKernel(HIP.V, HIP.V, HIP.apply_Hd_petsc, HIP.apply_Hd_petsc,
                                   num_batches, num_batches,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma)

    A_pch_nonsym, extras = make_hmatrix_from_kernel(PCK, hmatrix_tol=hmatrix_tol)

    A_pch = A_pch_nonsym.spd()

    H_hmatrix = A_pch + a_reg_morozov * R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()

    _, _, residuals_PCH, errors_PCH = custom_cg(HIP.H_linop, -g0_numpy,
                                                x_true=u0_numpy,
                                                M=iH_hmatrix.as_linear_operator(),
                                                tol=1e-10, maxiter=500)

    all_residuals_PCH.append(residuals_PCH)
    all_errors_PCH.append(errors_PCH)


########    SAVE DATA    ########

all_num_batches = np.array(all_num_batches)
residuals_None = np.array(residuals_None)
errors_None = np.array(errors_None)
residuals_Reg = np.array(residuals_Reg)
errors_Reg = np.array(errors_Reg)
all_residuals_PCH = [np.array(r) for r in all_residuals_PCH]
all_errors_PCH = [np.array(e) for e in all_errors_PCH]

if save_data:
    np.savetxt(save_dir / 'all_num_batches.txt', all_num_batches)
    np.savetxt(save_dir / 'residuals_None.txt', residuals_None)
    np.savetxt(save_dir / 'errors_None.txt', errors_None)
    np.savetxt(save_dir / 'residuals_Reg.txt', residuals_Reg)
    np.savetxt(save_dir / 'errors_Reg.txt', errors_Reg)
    for k in range(len(all_num_batches)):
        np.savetxt(save_dir / ('all_residuals_PCH' + str(all_num_batches[k]) + '.txt'), all_residuals_PCH[k])
        np.savetxt(save_dir / ('all_errors_PCH' + str(all_num_batches[k]) + '.txt'), all_errors_PCH[k])


########    MAKE FIGURES    ########

plt.figure()
plt.semilogy(errors_Reg)
plt.semilogy(errors_None)
legend = ['Reg', 'None']
for k in range(len(all_num_batches)):
    plt.semilogy(all_errors_PCH[k])
    legend.append('PCH ' + str(all_num_batches[k]))

plt.xlabel('Iteration')
plt.ylabel('relative l2 error')
plt.title('Relative error vs. Krylov iteration')
plt.legend(legend)

if save_figures:
    plt.savefig(save_dir / 'error_vs_krylov_iter.pdf', bbox_inches='tight', dpi=100)


plt.figure()
plt.semilogy(residuals_Reg)
plt.semilogy(residuals_None)
legend = ['Reg', 'None']
for k in range(len(all_num_batches)):
    plt.semilogy(all_residuals_PCH[k])
    legend.append('PCH ' + str(all_num_batches[k]))

plt.xlabel('Iteration')
plt.ylabel('relative residual')
plt.title('Relative residual vs. Krylov iteration')
plt.legend(legend)

if save_figures:
    plt.savefig(save_dir / 'relative_residual_vs_krylov_iter.pdf', bbox_inches='tight', dpi=100)

