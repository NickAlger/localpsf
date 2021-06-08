import numpy as np
import fenics
import matplotlib.pyplot as plt
from build_dense_matrix_from_matvecs import build_dense_matrix_from_matvecs
from fenics_to_scipy_sparse_csr_conversion import vec2fct
from localpsf import LocalPSF
from fenics_function_extend_by_zero_evaluator import FenicsFunctionExtendByZeroEvaluator
from smooth_basis_maker import SmoothBasisMakerNeumann
from time import time
from heat_inverse_problem import HeatInverseProblem
from fenics_function_fast_grid_evaluator import FenicsFunctionFastGridEvaluator


final_time_T = 7e-5 # 5e-5 is good
num_timesteps = 35 # 35 is good
random_seed = 1 # 1 is good
# mesh_h = 5e-3 # 1e-2 is good
mesh_h = 2e-2
finite_element_order = 1
lumped_mass_matrix=False
boundary_epsilon = 1e-1
num_standard_deviations_tau = 3
max_smooth_vectors = 100
max_batches = 10
error_epsilon = 1e-4
use_neumann_modes = False

np.random.seed(random_seed)
HIP = HeatInverseProblem(mesh_h=mesh_h, finite_element_order=finite_element_order, final_time_T=final_time_T,
                         num_timesteps=num_timesteps, lumped_mass_matrix=lumped_mass_matrix,
                         perform_checks=True, uniform_kappa=False)
HIP.interactive_hessian_impulse_response_plot()

###


boundary_source_dual_vector = fenics.assemble(fenics.TestFunction(HIP.V) * fenics.ds)[:]

# x = HIP.solve_M(HIP.apply_hessian(HIP.solve_M(boundary_source_dual_vector)))

def evaluation_function_factory(u_vec):
    # u_evaluator = FenicsFunctionExtendByZeroEvaluator(u_vec, HIP.V)
    u_evaluator = FenicsFunctionFastGridEvaluator(u_vec, HIP.V, oversampling_parameter=1.5)
    return u_evaluator


u_vec = HIP.solve_M(HIP.apply_hessian(HIP.solve_M(np.random.randn(HIP.N))))
# u = vec2fct(u_vec, HIP.V)
# plt.figure()
# fenics.plot(u)
# plt.title('hessian evaluation on random function')

t = time()
u_evaluator = evaluation_function_factory(u_vec)
time_evaluation_function_factory = time() - t
print('time_evaluation_function_factory=', time_evaluation_function_factory)
dof_coords_X = HIP.V.tabulate_dof_coordinates()

u2_vec = u_evaluator(dof_coords_X)
err_u_evaluator = np.linalg.norm(u2_vec - u_vec) / np.linalg.norm(u_vec)
print('err_u_evaluator=', err_u_evaluator)

ntest = int(HIP.N)
pp = np.random.rand(ntest,2)
t = time()
z = u_evaluator(pp)
interpolate_time = time() - t
print('ntest=', ntest, ', u_evaluator_time=', interpolate_time)

def multiquadric(r, epsilon):
    return np.sqrt(1. + (epsilon * r)**2)

def gaussian(r, sigma):
    return np.exp(-0.5 * r**2 / sigma**2)

def make_radial_basis_function(p):
    rr = np.linalg.norm(dof_coords_X - p, axis=1)
    # return multiquadric(rr, 1.0e-3)
    return gaussian(rr, 1e-2)

if use_neumann_modes:
    print('building smooth basis')
    sbm = SmoothBasisMakerNeumann(HIP.V, max_smooth_vectors=max_smooth_vectors)
    sbm.k = 0
    get_smooth_vector = lambda p: sbm.get_smooth_vector()
else: # use radial basis functions
    get_smooth_vector = make_radial_basis_function


print('constructing localpsf')
lpsf = LocalPSF(HIP.apply_hessian, HIP.apply_hessian, dof_coords_X, boundary_source_dual_vector, get_smooth_vector, evaluation_function_factory,
                HIP.M, HIP.solve_M, error_epsilon, boundary_epsilon=boundary_epsilon, num_standard_deviations_tau=num_standard_deviations_tau,
                max_smooth_vectors=max_smooth_vectors, max_batches=max_batches)


####
U = np.array(lpsf.uu).T

ww = lpsf.evaluate_spatially_varying_weighting_functions_at_points_xk(lpsf.X)
# ww2 = np.dot(np.linalg.pinv(lpsf.U_S), U)
k=6
# wk_fct= vec2fct(lpsf.uu[k], HIP.V)
wk_fct= vec2fct(ww[k,:], HIP.V)
# wk_fct= vec2fct(ww2[k,:], HIP.V)
# wk_fct.vector()[:] = ww2[k,:] * (np.abs(ww2[k,:]) < 0.3)
plt.figure()
fenics.plot(wk_fct)

# ntest = int(HIP.N)
# pp = np.random.rand(ntest,2)


# for k in range(lpsf.num_batches):
#     plt.figure()
#     fenics.plot(vec2fct(lpsf.all_eta[k], HIP.V))
#     plt.title('eta'+str(k))
#
# k=15
# plt.figure()
# fenics.plot(vec2fct(lpsf.uu[k], HIP.V))
# plt.title('uk')

####

if False:
    p = np.array([0.5,0.5])

    yy = lpsf.impulse_responses
    xx = np.dot(np.ones((yy.shape[0],1)), p.reshape((1,-1)))

    t = time()
    # zz = lpsf.evaluate_Htilde_at_points_yy_xx(yy,xx)
    zz = lpsf.evaluate_Htilde_at_points_yy_xx(yy,xx)
    dt_evalcol = time() - t
    print('dt_evalcol=', dt_evalcol)

    z_fct = vec2fct(zz, HIP.V)
    plt.figure()
    fenics.plot(z_fct)

