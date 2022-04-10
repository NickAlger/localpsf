from localpsf import ellipsoid
from localpsf import impulse_response_batches
from localpsf import impulse_response_moments
from localpsf import product_convolution_kernel
from localpsf import product_convolution_hmatrix
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel
from localpsf import sample_point_batches
from localpsf import visualization
import localpsf.stokes_inverse_problem_cylinder
import localpsf.op_operations


import sys
import hlibpro_python_wrapper as hpro
from localpsf.stokes_inverse_problem_cylinder import *
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel, product_convolution_hmatrix
import dolfin as dl


# In[3]:


from nalger_helper_functions import plot_ellipse
from nalger_helper_functions.custom_cg import custom_gmres

save_data = True
save_figures = True
symmetric_PCH = True
import os
root_dir = os.path.abspath(os.curdir)
rel_save_dir = './error_vs_krylov_iter/'
save_dir = os.path.join(root_dir, rel_save_dir)
os.makedirs(save_dir, exist_ok=True)


# --------- set up the problem
mfile_name = "meshes/cylinder_coarse"
mesh = dl.Mesh(mfile_name+".xml")
boundary_markers = dl.MeshFunction("size_t", mesh, mfile_name+"_facet_region.xml")


Newton_iterations = 2

nondefault_StokesIP_options = {'mesh' : mesh,'boundary_markers' : boundary_markers,
        'Newton_iterations': Newton_iterations, 
        'misfit_only': True,
        'gauss_newton_approx': True,
        'load_fwd': True,
        'Newton_iterations': Newton_iterations,
        'lam': 1.e10,
        'solve_inv': False,
        'gamma': 1.e4,
        'm0': 1.5*7.,
        'mtrue_string': 'm0 - (m0 / 7.)*std::cos((x[0]*x[0]+x[1]*x[1])*pi/(Radius*Radius))'}

StokesIP = StokesInverseProblemCylinder(**nondefault_StokesIP_options)
g0_numpy = StokesIP.g_numpy(StokesIP.x)

H_linop       = spla.LinearOperator((StokesIP.N, StokesIP.N), matvec=StokesIP.apply_H_Vsub_numpy)
solve_R_linop = spla.LinearOperator((StokesIP.N, StokesIP.N), matvec=StokesIP.apply_Rinv_numpy)

# ---- build the regularization operator for future compression
K = csr_fenics2scipy(StokesIP.priorVsub.A)
M = csr_fenics2scipy(StokesIP.priorVsub.M)
Minv = sps.diags(1. / M.diagonal(), format="csr")
R = K.dot(Minv).dot(K)
# ----

hmatrix_tol = 1e-4
tau   = 3
gamma = 1.e-5
sigma_min = 1.
num_neighbors = 10

all_num_batches = [5, 10, 15]
num_batches     = all_num_batches[-1]
PCK = ProductConvolutionKernel(StokesIP.V, StokesIP.V, StokesIP.apply_Hd_petsc, StokesIP.apply_Hd_petsc,
                               num_batches, num_batches,
                               tau_rows=tau, tau_cols=tau,
                               num_neighbors_rows=num_neighbors,
                               num_neighbors_cols=num_neighbors,
                               symmetric=True,
                               gamma=gamma,
                               sigma_min=sigma_min,
                               max_scale_discrepancy=1e6)

A_pch_sym, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=hmatrix_tol, use_lumped_mass_matrix=True)
A_pch_nonsym = extras['A_hmatrix_nonsym']

maxiter = 400
lintol  = 1.e-12

if symmetric_PCH:
    A_pch = A_pch_sym
    linear_solver = custom_cg
    kwargs = {'maxiter': maxiter}
else:
    A_pch = A_pch_nonsym
    linear_solver = custom_gmres
    inner_iterations = maxiter
    outer_iterations = 1
    kwargs = {'maxiter': outer_iterations, 'restrt': inner_iterations}

# ------- ERROR ESTIMATION -----

x = np.random.randn(A_pch.shape[1])
y1_fct = dl.Function(StokesIP.V)
y1_fct.vector()[:] = A_pch * x

x_petsc = dl.Function(StokesIP.V).vector()
x_petsc[:] = x
y2_fct = dl.Function(StokesIP.V)
y2_fct.vector()[:] = StokesIP.apply_Hd_petsc(x_petsc)[:]

y3_fct = dl.Function(StokesIP.V)
y3_fct.vector()[:] = A_pch_nonsym * x

err_sym = np.linalg.norm(y1_fct.vector()[:] - y2_fct.vector()[:]) / np.linalg.norm(y2_fct.vector()[:])
print('err_sym=', err_sym)

err_nonsym = np.linalg.norm(y3_fct.vector()[:] - y2_fct.vector()[:]) / np.linalg.norm(y2_fct.vector()[:])
print('err_nonsym=', err_nonsym)


# In[13]:


def apply_Hd_numpy(x_numpy):
    x_petsc = dl.Function(StokesIP.V).vector()
    x_petsc[:] = x_numpy
    return StokesIP.apply_Hd_petsc(x_petsc)[:]


num_random_error_matvecs = 50

_, _, _, Anonsym_relative_err_fro     = estimate_column_errors_randomized(apply_Hd_numpy,
                                        lambda x: A_pch_nonsym * x,
                                        A_pch_nonsym.shape[1],
                                        num_random_error_matvecs)

print('Anonsym_relative_err_fro=', Anonsym_relative_err_fro)

_, _, _, A_relative_err_fro     = estimate_column_errors_randomized(apply_Hd_numpy,
                                        lambda x: A_pch * x,
                                        A_pch.shape[1],
                                        num_random_error_matvecs)
if symmetric_PCH:
    print('Asym_relative_err_fro=', A_relative_err_fro)
else:
    print('Anonsym_relative_err_fro=', A_relative_err_fro)

# In[13]:


R0_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(R, A_pch.bct)
H_hmatrix = A_pch + R0_hmatrix
iH_hmatrix = H_hmatrix.inv()



# ----- increased fidelity linear system solution


u0_numpy, info, residuals = linear_solver(H_linop, -g0_numpy,
                                      M=iH_hmatrix.as_linear_operator(),
                                      tol=lintol/10., **kwargs)


if info == 0:
    print("SUCCESSFUL LINEAR SOLVE!")
else:
    print("LINEAR SOLVE FAILURE!")


_, info, residuals_None, errors_None = linear_solver(H_linop, -g0_numpy,
                                              x_true = u0_numpy,          
                                              tol=1e-10, **kwargs)
if info == 0:
    print("SUCCESSFUL LINEAR SOLVE!")
else:
    print("LINEAR SOLVE FAILURE!")



_, info, residuals_Reg, errors_Reg = linear_solver(H_linop, -g0_numpy,
                                            M=solve_R_linop,
                                            x_true = u0_numpy,
                                            tol=1e-10, **kwargs)

if info == 0:
    print("SUCCESSFUL LINEAR SOLVE!")
else:
    print("LINEAR SOLVE FAILURE!")

# ---- generate data -----
all_residuals_PCH = list()
all_errors_PCH    = list()
for num_batches in all_num_batches:
    PCK = ProductConvolutionKernel(StokesIP.V, StokesIP.V, StokesIP.apply_Hd_petsc, StokesIP.apply_Hd_petsc,
                                   num_batches, num_batches,
                                   tau_rows=tau, tau_cols=tau,
                                   num_neighbors_rows=num_neighbors,
                                   num_neighbors_cols=num_neighbors,
                                   symmetric=True,
                                   gamma=gamma,
                                   sigma_min=sigma_min,
                                   max_scale_discrepancy=1e6)
    
    A_pch_sym, extras = make_hmatrix_from_kernel(PCK, make_positive_definite=True, hmatrix_tol=hmatrix_tol, use_lumped_mass_matrix=True)
    A_pch_nonsym = extras['A_hmatrix_nonsym']
    if symmetric_PCH:
        A_pch = A_pch_sym
    else:
        A_pch = A_pch_nonsym
    H_hmatrix = A_pch + R0_hmatrix
    iH_hmatrix = H_hmatrix.inv()
    _, info, residuals_PCH, errors_PCH = linear_solver(H_linop, -g0_numpy,
                                      x_true = u0_numpy,
                                      M=iH_hmatrix.as_linear_operator(),
                                      tol=lintol, **kwargs)
    if info == 0:
        print("SUCCESSFUL LINEAR SOLVE!")
    else:
        print("LINEAR SOLVE FAILURE!")
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
    np.savetxt(save_dir + 'all_num_batches.txt', all_num_batches)
    np.savetxt(save_dir + 'residuals_None.txt', residuals_None)
    np.savetxt(save_dir + 'errors_None.txt', errors_None)
    np.savetxt(save_dir + 'residuals_Reg.txt', residuals_Reg)
    np.savetxt(save_dir + 'errors_Reg.txt', errors_Reg)
    for k in range(len(all_num_batches)):
        np.savetxt(save_dir + ('all_residuals_PCH' + str(all_num_batches[k]) + '.txt'), all_residuals_PCH[k])
        np.savetxt(save_dir + ('all_errors_PCH' + str(all_num_batches[k]) + '.txt'), all_errors_PCH[k])


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
    if symmetric_PCH:
        plt.savefig(save_dir + 'error_vs_krylov_iter_sym.pdf', bbox_inches='tight', dpi=100)
    else:
        plt.savefig(save_dir + 'error_vs_krylov_iter_nonsym.pdf', bbox_inches='tight', dpi=100)


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
    if symmetric_PCH:
        plt.savefig(save_dir + 'relative_residual_vs_krylov_iter_sym.pdf', bbox_inches='tight', dpi=100)
    else:
        plt.savefig(save_dir + 'relative_residual_vs_krylov_iter_nonsym.pdf', bbox_inches='tight', dpi=100)



