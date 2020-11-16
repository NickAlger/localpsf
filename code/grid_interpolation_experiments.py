import numpy as np
import fenics
import matplotlib.pyplot as plt
from fenics_function_extend_by_zero_evaluator import FenicsFunctionExtendByZeroEvaluator
from smooth_basis_maker import SmoothBasisMakerNeumann
from time import time
from heat_inverse_problem import HeatInverseProblem
from scipy.interpolate import interpn


final_time_T = 7e-5 # 5e-5 is good
num_timesteps = 35 # 35 is good
random_seed = 1 # 1 is good
mesh_h = 5e-3 # 1e-2 is good
finite_element_order = 1
lumped_mass_matrix=False
boundary_epsilon = 1e-1
num_standard_deviations_tau = 3
max_smooth_vectors = 25
max_batches = 4
error_epsilon = 1e-4

np.random.seed(random_seed)
HIP = HeatInverseProblem(mesh_h=mesh_h, finite_element_order=finite_element_order, final_time_T=final_time_T,
                         num_timesteps=num_timesteps, lumped_mass_matrix=lumped_mass_matrix,
                         perform_checks=True, uniform_kappa=False)
HIP.interactive_hessian_impulse_response_plot()

###

dof_coords_X = HIP.V.tabulate_dof_coordinates()

print('building smooth basis')
sbm = SmoothBasisMakerNeumann(HIP.V, max_smooth_vectors=max_smooth_vectors)



oversampling_parameter = 2
V = HIP.V
d = V.mesh().geometric_dimension()
mesh = V.mesh()
hmin = mesh.hmin()
X = V.tabulate_dof_coordinates()

f = fenics.Function(V)
f.vector()[:] = sbm.U_smooth[:,1].copy()
f.set_allow_extrapolation(True)

min_point = np.min(X, axis=0)
max_point = np.max(X, axis=0)
nn = (oversampling_parameter * (max_point - min_point) / hmin).astype(int)
xx = tuple([np.linspace(min_point[i], max_point[i], nn[i]) for i in range(d)])
XX = np.meshgrid(*xx, indexing='ij')
pp = np.array([X.reshape(-1) for X in XX]).T

t = time()
ff = np.zeros(nn).reshape(-1)
for k in range(len(ff)):
    ff[k] = f(pp[k,:])
F = ff.reshape(nn)
dt_grid_eval = time() - t
print('dt_grid_eval=', dt_grid_eval)

zz = interpn(xx,F,X, bounds_error=False, fill_value=0.0)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=zz)

f_vec = f.vector()[:]
err_grid_interp = np.linalg.norm(f_vec - zz)/np.linalg.norm(f_vec)
print('err_grid_interp=', err_grid_interp)

n_test = int(1e4)
qq = np.random.randn(n_test,d)
t = time()
zz = interpn(xx,F,qq, bounds_error=False, fill_value=0.0)
dt_interpn = time() - t
print('n_test=', n_test, ', dt_interpn=', dt_interpn)

n_test = int(1e4)
qq = np.random.randn(n_test,d)
t = time()
for k in range(n_test):
    zz = interpn(xx,F,qq[k,:], bounds_error=False, fill_value=0.0)
dt_interpn_loop = time() - t
print('n_test=', n_test, ', dt_interpn_loop=', dt_interpn_loop)