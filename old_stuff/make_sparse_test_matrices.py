import numpy as np
import fenics
import scipy.io as sio
import scipy.sparse as sps

mesh = fenics.UnitSquareMesh(85, 79)
V = fenics.FunctionSpace(mesh, 'CG', 1)
dof_coords = V.tabulate_dof_coordinates()
x_coords = dof_coords[:,0]
y_coords = dof_coords[:,1]

u_trial = fenics.TrialFunction(V)
v_test = fenics.TestFunction(V)

# Test matrix: discretized Laplacian on regular grid
A = fenics.assemble(fenics.inner(fenics.grad(u_trial), fenics.grad(v_test)) * fenics.dx)
B = fenics.assemble(u_trial * v_test * fenics.dx)

def convert_fenics_csr_matrix_to_scipy_csr_matrix(A_fenics):
    ai, aj, av = fenics.as_backend_type(A_fenics).mat().getValuesCSR()
    A_scipy = sps.csr_matrix((av, aj, ai))
    return A_scipy

A_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(A).tocsc()
B_csc = convert_fenics_csr_matrix_to_scipy_csr_matrix(B).tocsc()
AB_csc = (A_csc * B_csc).tocsc()
x = np.random.randn(V.dim())


sio.savemat("test_matrix_1.mat", {'A': A_csc})
sio.savemat("test_matrix_2.mat", {'B': B_csc})
sio.savemat("test_matrix_12.mat", {'AB': AB_csc})
sio.savemat("test_vector.mat", {'x': x})

A_csc[0,1] = A_csc[0,1] + 1e-14
B_csc[0,1] = B_csc[0,1] + 1e-14
AB_csc[0,1] = AB_csc[0,1] + 1e-14
sio.savemat("test_matrix_1_perturbed.mat", {'A': A_csc})
sio.savemat("test_matrix_2_perturbed.mat", {'B': B_csc})
sio.savemat("test_matrix_12_perturbed.mat", {'AB': AB_csc})

sio.savemat("test_x_coords.mat", {'x_coords': x_coords})
sio.savemat("test_y_coords.mat", {'y_coords': y_coords})