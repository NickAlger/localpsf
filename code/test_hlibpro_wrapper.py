import numpy as np
import scipy.sparse as sps
import fenics
import hlibpro_wrapper as hpro


########    SETUP    ########

mesh = fenics.UnitSquareMesh(75, 75)
V = fenics.FunctionSpace(mesh, 'CG', 1)
dof_coords = V.tabulate_dof_coordinates()

u_trial = fenics.TrialFunction(V)
v_test = fenics.TestFunction(V)

stiffness_form = fenics.inner(fenics.grad(u_trial), fenics.grad(v_test))*fenics.dx
mass_form = u_trial*v_test*fenics.dx

K = fenics.assemble(stiffness_form)
M = fenics.assemble(mass_form)

def convert_fenics_csr_matrix_to_scipy_csr_matrix(A_fenics):
    ai, aj, av = fenics.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy

K_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(K).tocsc()
M_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(M).tocsc()

A_csc = K_csc + M_csc


########    CLUSTER TREE / BLOCK CLUSTER TREE    ########

ct = hpro.build_cluster_tree_from_dof_coords(dof_coords, 60)
bct = hpro.build_block_cluster_tree(ct, ct, 2.0)
hpro.visualize_cluster_tree(ct, "sparsemat_cluster_tree_from_python")
hpro.visualize_block_cluster_tree(bct, "sparsemat_block_cluster_tree_from_python")


########    BUILD HMATRIX FROM SPARSE    ########

K_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(K_csc, bct)
M_hmatrix = hpro.build_hmatrix_from_scipy_sparse_matrix(M_csc, bct)

hpro.visualize_hmatrix(K_hmatrix, "stiffness_hmatrix_from_python")

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(K_hmatrix, x)

y2 = K_csc * x
err_hmat_stiffness = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_hmat_stiffness=', err_hmat_stiffness)


########    SCALE HMATRIX   ########

three_K_hmatrix = hpro.h_scale(K_hmatrix, 3.0)
three_y = hpro.h_matvec(three_K_hmatrix, x)

err_h_scale = np.linalg.norm(three_y - 3.0*y)
print('err_h_scale=', err_h_scale)


########    VISUALIZE HMATRIX    ########

hpro.visualize_hmatrix(M_hmatrix, "mass_hmatrix_from_python")

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(M_hmatrix, x)

y2 = M_csc * x
err_hmat_mass = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_hmat_mass=', err_hmat_mass)


########    ADD HMATRICES   ########

A_hmatrix = hpro.h_add(K_hmatrix, M_hmatrix)

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(A_hmatrix, x)

y2 = A_csc * x
err_h_add = np.linalg.norm(y - y2)/np.linalg.norm(y2)
print('err_h_add=', err_h_add)


########    MULTIPLY HMATRICES   ########

KM_hmatrix = hpro.h_mul(K_hmatrix, M_hmatrix, rtol=1e-6)

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(KM_hmatrix, x)

y2 = hpro.h_matvec(K_hmatrix, hpro.h_matvec(M_hmatrix, x))
# y4 = 0.5 * (K_csc * (M_csc * x) + M_csc * (K_csc * x))
err_H_mul = np.linalg.norm(y2 - y)/np.linalg.norm(y2)
print('err_H_mul=', err_H_mul)

y3 = K_csc * (M_csc * x)
err_H_mul_exact = np.linalg.norm(y3 - y)/np.linalg.norm(y2)
print('err_H_mul_exact=', err_H_mul_exact)


########    FACTORIZE HMATRIX   ########

iA_factorized = hpro.h_factorized_inverse(A_hmatrix, rtol=1e-10)

x = np.random.randn(dof_coords.shape[0])
y = hpro.h_matvec(A_hmatrix, x)

x2 = hpro.h_factorized_solve(iA_factorized, y)

err_hfac = np.linalg.norm(x - x2)/np.linalg.norm(x2)
print('err_hfac=', err_hfac)

hpro.visualize_inverse_factors(iA_factorized, 'inv_A_factors')