import dolfin as dl
import ufl
import math
import numpy as np
import typing as typ
import matplotlib.pyplot as plt
from dataclasses import dataclass
from functools import cached_property
import sys
import os
sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "../") )
from hippylib import *
sys.path.append( "/home/nick/repos/hippylib/applications/ad_diff" )

# sys.path.append( os.environ.get('HIPPYLIB_BASE_DIR', "..") + "/applications/ad_diff/" )
from model_ad_diff import TimeDependentAD, SpaceTimePointwiseStateObservation

import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)

from scipy.spatial import KDTree
import scipy.sparse.linalg as spla
import scipy.linalg as sla
# from localpsf.bilaplacian_regularization import BiLaplacianRegularization
from localpsf.bilaplacian_regularization_lumped import BilaplacianRegularization, BiLaplacianCovariance, make_bilaplacian_covariance
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
from localpsf.inverse_problem_objective import PSFHessianPreconditioner
import nalger_helper_functions as nhf

import hlibpro_python_wrapper as hpro
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel


###############################    MESH AND FINITE ELEMENT SPACES    ###############################

@dataclass(frozen=True)
class AdvMeshAndFunctionSpace:
    mesh: dl.Mesh
    Vh: dl.FunctionSpace

    @cached_property
    def dof_coords(me) -> np.ndarray:
        return me.Vh.tabulate_dof_coordinates()

    @cached_property
    def kdtree(me) -> KDTree:
        return KDTree(me.dof_coords)

    @cached_property
    def mass_lumps(me) -> np.ndarray:
        return dl.assemble(dl.TestFunction(me.Vh) * dl.dx)[:]

    def find_nearest_dof(me, p: np.ndarray) -> int:
        return me.kdtree.query(p)[1]


def make_adv_mesh_and_function_space(num_refinements: int=2):
    mesh = dl.Mesh("ad_20.xml")
    for _ in range(num_refinements):
        mesh = dl.refine(mesh)

    Vh = dl.FunctionSpace(mesh, "Lagrange", 1)
    print("Number of dofs: {0}".format(Vh.dim()))

    return AdvMeshAndFunctionSpace(mesh, Vh)


###############################    VELOCITY FIELD    ###############################

@dataclass(frozen=True)
class AdvWind:
    reynolds_number: float
    velocity: dl.Function
    pressure: dl.Function

    @cached_property
    def v_plot(me) -> dl.Function:
        return nb.coarsen_v(me.velocity)

    @cached_property
    def plot_mesh(me) -> dl.Mesh:
        return me.v_plot.function_space().mesh()

    def plot_velocity(me,
                      figsize:typ.Tuple[int, int]=(5,5),
                      quiver_options: typ.Dict[str, typ.Any]=None):
        quiver_options2 = {'units' : 'x',
                           'headaxislength' : 7,
                           'headwidth' : 7,
                           'headlength' : 7,
                           'scale' : 4,
                           'pivot' : 'tail' # 'middle'
                           }
        if quiver_options is not None:
            quiver_options2.update(quiver_options)

        plt.figure(figsize=figsize)
        w0 = me.v_plot.compute_vertex_values(me.plot_mesh)

        X = me.plot_mesh.coordinates()[:, 0]
        Y = me.plot_mesh.coordinates()[:, 1]
        U = w0[:me.plot_mesh.num_vertices()]
        V = w0[me.plot_mesh.num_vertices():]
        # C = np.sqrt(U*U+V*V)
        C = np.ones(U.shape)

        plt.quiver(X, Y, U, V, C, **quiver_options2)

        plt.axis('off')
        plt.set_cmap('gray')
        plt.gca().set_aspect('equal')


def make_adv_wind(mesh: dl.Mesh,
                  reynolds_number: float=1e2) -> AdvWind:
    def v_boundary(x, on_boundary):
        return on_boundary

    def q_boundary(x, on_boundary):
        return x[0] < dl.DOLFIN_EPS and x[1] < dl.DOLFIN_EPS  # original

    Xh = dl.VectorFunctionSpace(mesh, 'Lagrange', 2)
    Wh = dl.FunctionSpace(mesh, 'Lagrange', 1)
    mixed_element = ufl.MixedElement([Xh.ufl_element(), Wh.ufl_element()])
    XW = dl.FunctionSpace(mesh, mixed_element)

    Re = dl.Constant(reynolds_number)

    g = dl.Expression(('0.0', '(x[0] < 1e-14) - (x[0] > 1 - 1e-14)'), degree=1)  # original
    bc1 = dl.DirichletBC(XW.sub(0), g, v_boundary)
    bc2 = dl.DirichletBC(XW.sub(1), dl.Constant(0), q_boundary, 'pointwise')
    bcs = [bc1, bc2]

    vq = dl.Function(XW)
    (v, q) = ufl.split(vq)
    (v_test, q_test) = dl.TestFunctions(XW)

    def strain(v):
        return ufl.sym(ufl.grad(v))

    F = ((2. / Re) * ufl.inner(strain(v), strain(v_test)) + ufl.inner(ufl.nabla_grad(v) * v, v_test)
         - (q * ufl.div(v_test)) + (ufl.div(v) * q_test)) * ufl.dx

    dl.solve(F == 0, vq, bcs, solver_parameters={"newton_solver":
                                                     {"relative_tolerance": 1e-4, "maximum_iterations": 100}})

    velocity = dl.project(v, Xh)
    pressure = dl.project(q, Wh)

    return AdvWind(reynolds_number, velocity, pressure)


###############################    INITIAL CONDITION    ###############################

class FenicsFunctionSmoother:
    def __init__(me, function_space_V, smoothing_time=1e-2, num_timesteps=10):
        me.function_space_V = function_space_V
        me.num_timesteps = num_timesteps
        me.dt = smoothing_time / num_timesteps

        u = dl.TrialFunction(function_space_V)
        v = dl.TestFunction(function_space_V)

        mass_form = u * v * dl.dx
        stiffness_form = dl.inner(dl.grad(u), dl.grad(v)) * dl.dx

        me.M = dl.assemble(mass_form)
        Z = dl.assemble(mass_form + me.dt * stiffness_form)

        me.Z_solver = dl.LUSolver(Z)

    def smooth(me, function_f):
        for k in range(me.num_timesteps):
            me.Z_solver.solve(function_f.vector(), me.M * function_f.vector())

    def random_smooth_function(me):
        f = dl.Function(me.function_space_V)
        f.vector()[:] = np.random.randn(me.function_space_V.dim())
        me.smooth(f)
        return f


def checkerboard_function(nx, ny, Vh, smoothing_time=2e-4):
    xx = list(np.linspace(0.0, 1.0, nx+1))
    yy = list(np.linspace(0.0, 1.0, ny+1))
    sx = ''
    for x in xx[1:-1]:
        sx += '(2*(x[0] > ' + str(x) + ')-1)*'
    sy = ''
    for y in yy[1:-1]:
        sy += '(2*(x[1] > ' + str(y) + ')-1)*'
    s = sx + sy + '0.5 + 0.5'
    checker_expr = dl.Expression(s, element=Vh.ufl_element())

    checker_func = dl.interpolate(checker_expr, Vh)
    Vh_smoother = FenicsFunctionSmoother(Vh, smoothing_time=smoothing_time)
    Vh_smoother.smooth(checker_func)
    return checker_func


####

def adv_parameter_to_triple(
        m_vec: np.ndarray,
        Vh: dl.FunctionSpace,
        problem: TimeDependentAD
) -> typ.List[dl.Function]:
    assert (m_vec.shape == (Vh.dim(),))
    m_petsc = dl.Function(Vh).vector()
    m_petsc[:] = m_vec
    u_star = problem.generate_vector(0)
    x = [u_star, m_petsc, None]
    problem.solveFwd(x[0], x)
    return x

def adv_parameter_to_observable_map(
        m_vec: np.ndarray,
        Vh: dl.FunctionSpace,
        problem: TimeDependentAD,
        misfit: SpaceTimePointwiseStateObservation,
) -> np.ndarray:
    assert (m_vec.shape == (Vh.dim(),))
    x = adv_parameter_to_triple(m_vec, Vh, problem)
    problem.solveFwd(x[0], x)
    new_obs_hippy = misfit.d.copy()
    new_obs_hippy.zero()
    misfit.observe(x, new_obs_hippy)
    new_obs = np.array([di[:].copy() for di in new_obs_hippy.data])
    return new_obs


@dataclass(frozen=True)
class AdvUniverse:
    mesh_and_function_space: AdvMeshAndFunctionSpace
    wind_object: AdvWind
    problem: TimeDependentAD
    misfit: SpaceTimePointwiseStateObservation
    regularization: BilaplacianRegularization
    true_initial_condition: np.ndarray
    obs: np.ndarray
    true_obs: np.ndarray

    def __post_init__(me):
        assert(me.obs.shape == (me.num_obs_times, me.num_obs_locations))
        assert(me.true_obs.shape == (me.num_obs_times, me.num_obs_locations))
        assert(me.true_initial_condition.shape == (me.N,))
        assert(me.regularization.N == me.N)

    @cached_property
    def H(me) -> ReducedHessian:
        return ReducedHessian(me.problem, misfit_only=True)

    @cached_property
    def Vh(me) -> dl.FunctionSpace:
        return me.mesh_and_function_space.Vh

    @cached_property
    def N(me) -> int:
        return me.Vh.dim()

    @cached_property
    def num_obs_locations(me) -> int:
        return me.true_obs.shape[1]

    @cached_property
    def num_obs_times(me) -> int:
        return me.true_obs.shape[0]

    @cached_property
    def noise(me) -> np.ndarray:
        return me.obs - me.true_obs

    def misfit_cost(me, m_vec: np.ndarray) -> float:
        assert (m_vec.shape == (me.N,))
        [u, m, p] = me.problem.generate_vector()
        m[:] = m_vec.copy()
        me.problem.solveFwd(u, [u, m, p])
        me.problem.solveAdj(p, [u, m, p])
        total_cost, reg_cost, misfit_cost = me.problem.cost([u, m, p])
        return misfit_cost

    def misfit_gradient(me, m_vec: np.ndarray) -> np.ndarray:
        assert(m_vec.shape == (me.N,))
        [u, m, p] = me.problem.generate_vector()
        m[:] = m_vec.copy()
        me.problem.solveFwd(u, [u, m, p])
        me.problem.solveAdj(p, [u, m, p])
        mg = me.problem.generate_vector(PARAMETER)
        grad_norm = me.problem.evalGradientParameter([u, m, p], mg, misfit_only=True)
        return mg[:].copy()

    def apply_misfit_hessian(me, x: np.ndarray) -> np.ndarray:
        assert(x.shape == (me.N,))
        x_petsc = dl.Function(me.mesh_and_function_space.Vh).vector()
        x_petsc[:] = x.copy()
        y_petsc = dl.Function(me.mesh_and_function_space.Vh).vector()
        me.H.mult(x_petsc, y_petsc)
        return y_petsc[:].copy()

    def get_misfit_hessian_impulse_response(me, location: typ.Union[int, np.ndarray]) -> np.ndarray:  # impulse response at kth dof location
        if isinstance(location, int):
            k = location
        else:
            assert(location.shape == (me.mesh_and_function_space.dof_coords.shape[1],))
            k = me.mesh_and_function_space.find_nearest_dof(location)
        assert(k < me.N)
        ek = np.zeros(me.N)
        ek[k] = 1.0
        return me.apply_misfit_hessian(ek / me.mesh_and_function_space.mass_lumps) / me.mesh_and_function_space.mass_lumps

    def cost(me, m_vec: np.ndarray, areg: float) -> float:
        assert(m_vec.shape == (me.N,))
        assert(areg >= 0.0)
        Jr = me.regularization.cost(m_vec, areg)
        Jd = me.misfit_cost(m_vec)
        J = Jd + Jr
        return J

    def gradient(me, m_vec: np.ndarray, areg: float) -> np.ndarray:
        assert(m_vec.shape == (me.N,))
        assert(areg >= 0.0)
        gr = me.regularization.gradient(m_vec, areg)
        gd = me.misfit_gradient(m_vec)
        g = gd + gr
        assert(g.shape == (me.N,))
        return g

    def apply_hessian(me, x: np.ndarray, areg: float) -> np.ndarray:
        assert(x.shape == (me.N,))
        assert(areg >= 0.0)
        HR_x = me.regularization.apply_hessian(x, me.regularization.mu, areg)
        Hd_x = me.apply_misfit_hessian(x)
        H_x = Hd_x + HR_x
        assert(H_x.shape == (me.N,))
        return H_x

    def noise_norm(me) -> float:
        return np.linalg.norm(me.noise)

    def parameter_to_observable_map(me, m_vec: np.ndarray) -> np.ndarray:
        assert(m_vec.shape == (me.N,))
        new_obs = adv_parameter_to_observable_map(
            m_vec,
            me.mesh_and_function_space.Vh,
            me.problem,
            me.misfit)
        assert(new_obs.shape == (me.num_obs_times, me.num_obs_locations))
        return new_obs

    def compute_discrepancy_norm(me, m_vec: np.ndarray) -> float:
        new_obs = me.parameter_to_observable_map(m_vec)
        discrepancy = new_obs - me.obs
        return np.linalg.norm(discrepancy)


def make_adv_universe(
        noise_level: float, # 0.05, i.e., 5% noise, is typical
        kappa: float, # 1e-3 is default for hippylib
        reynolds_number: float=1e2,
        num_mesh_refinements=2,
        t_init = 0.0,
        t_final: float=0.5,
        dt: float=0.1,
        prior_correlation_length: float=0.25,
        num_checkers_x: int=6,
        num_checkers_y: int=6,
) -> AdvUniverse:
    mesh_and_function_space = make_adv_mesh_and_function_space(num_refinements=num_mesh_refinements)

    mesh = mesh_and_function_space.mesh
    Vh = mesh_and_function_space.Vh

    Cov = make_bilaplacian_covariance(
        1.0, prior_correlation_length, mesh_and_function_space.Vh,
        mesh_and_function_space.mass_lumps)

    mu = np.zeros(mesh_and_function_space.Vh.dim())

    regularization = BilaplacianRegularization(Cov, mu)

    simulation_times = np.arange(t_init, t_final + .5 * dt, dt)
    observation_times = np.array([t_final])

    obs_coords = mesh_and_function_space.dof_coords

    misfit = SpaceTimePointwiseStateObservation(
        mesh_and_function_space.Vh, observation_times, obs_coords)

    misfit.noise_variance = 1.0 # Use our own noise, not hippylib's

    wind_object = make_adv_wind(mesh, reynolds_number=reynolds_number)

    fake_gamma = 1.0
    fake_delta = 1.0
    fake_prior = BiLaplacianPrior(mesh_and_function_space.Vh, fake_gamma, fake_delta, robin_bc=True)

    problem = TimeDependentAD(
        mesh, [Vh, Vh, Vh], fake_prior, misfit,
        simulation_times, wind_object.velocity, True, kappa=kappa)

    ic_func = checkerboard_function(num_checkers_x, num_checkers_y, Vh)
    true_initial_condition = ic_func.vector()[:].copy()

    x_true = adv_parameter_to_triple(true_initial_condition, Vh, problem)
    problem.solveFwd(x_true[0], x_true)
    true_obs_hippy = misfit.d.copy()
    true_obs_hippy.zero()
    misfit.observe(x_true, true_obs_hippy)
    true_obs = np.array([di[:].copy() for di in true_obs_hippy.data])

    u_true_vec = np.array(x_true[0].data)
    u_noise_vec = noise_level * np.random.randn(*u_true_vec.shape) * np.abs(u_true_vec)

    x = [x_true[0].copy(), x_true[1].copy(), None]
    for ii in range(len(x[0].data)):
        x[0].data[ii][:] = u_true_vec[ii,:] + u_noise_vec[ii,:]
    misfit.observe(x, misfit.d)

    obs_hippy = misfit.d.copy()
    obs_hippy.zero()
    misfit.observe(x, obs_hippy)
    obs = np.array([di[:].copy() for di in obs_hippy.data])

    return AdvUniverse(
        mesh_and_function_space, wind_object, problem, misfit, regularization,
        true_initial_condition, obs, true_obs)


import scipy.sparse.linalg as spla

noise_level=0.05
kappa=1e-3

ADV = make_adv_universe(noise_level, kappa)

p = np.random.rand(2)
phi = dl.Function(ADV.Vh)
phi.vector()[:] = ADV.get_misfit_hessian_impulse_response(p)
plt.figure()
cm = dl.plot(phi)
plt.colorbar(cm)
plt.plot(p[0], p[1], '*r')
plt.title('Impulse response')

m0 = np.zeros(ADV.N) # np.random.randn(ADV.N) # np.zeros(ADV.N)
areg = 1e-4

J0 = ADV.cost(m0, areg)
print('J0=', J0)
g0 = ADV.gradient(m0, areg)

H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg))
result = nhf.custom_cg(H_linop, -g0, display=True, maxiter=1000, tol=1e-4)

plt.figure()
mstar = dl.Function(ADV.Vh)
mstar.vector()[:] = m0 + result[0]
cm = dl.plot(mstar)
plt.colorbar(cm)

#

print('Making row and column cluster trees')
ct = hpro.build_cluster_tree_from_pointcloud(ADV.mesh_and_function_space.dof_coords, cluster_size_cutoff=50)

print('Making block cluster trees')
bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=2.0)

HR_hmatrix = ADV.regularization.Cov.make_invC_hmatrix(bct, 1.0)

psf_preconditioner = PSFHessianPreconditioner(
    ADV.apply_misfit_hessian, ADV.Vh, ADV.mesh_and_function_space.mass_lumps,
    HR_hmatrix, display=True)

psf_preconditioner.build_hessian_preconditioner()

psf_preconditioner.shifted_inverse_interpolator.insert_new_mu(areg)
psf_preconditioner.update_deflation(areg)

areg = 1e-4
P_linop = spla.LinearOperator(
    (ADV.N, ADV.N),
    matvec=lambda x: psf_preconditioner.solve_hessian_preconditioner(x, areg))
H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg))
result2 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=1000, tol=1e-8) # 73 iter

plt.figure()
mstar2 = dl.Function(ADV.Vh)
mstar2.vector()[:] = m0 + result2[0]
cm = dl.plot(mstar2)
plt.colorbar(cm)

#

psf_preconditioner.current_preconditioning_type = 'reg'
result3 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=1000, tol=1e-8) # 932 iter

#

psf_preconditioner.current_preconditioning_type = 'none'
result3 = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=1000, tol=1e-8) # 388 iter

#

predicted_obs = ADV.parameter_to_observable_map(mstar2.vector()[:])

discrepancy = np.linalg.norm(predicted_obs - ADV.obs)
noise_discrepancy = np.linalg.norm(ADV.noise)

print('discrepancy=', discrepancy)
print('noise_discrepancy=', noise_discrepancy)


# more (25) batches

psf_preconditioner.psf_options['num_initial_batches'] = 25

psf_preconditioner.build_hessian_preconditioner()

psf_preconditioner.shifted_inverse_interpolator.insert_new_mu(areg)
psf_preconditioner.update_deflation(areg)

areg = 1e-5
P_linop = spla.LinearOperator(
    (ADV.N, ADV.N),
    matvec=lambda x: psf_preconditioner.solve_hessian_preconditioner(x, areg))
H_linop = spla.LinearOperator((ADV.N, ADV.N), matvec=lambda x: ADV.apply_hessian(x, areg))
result = nhf.custom_cg(H_linop, -g0, M=P_linop, display=True, maxiter=1000, tol=1e-8) # 23 iter


mstar = dl.Function(ADV.Vh)
mstar.vector()[:] = m0 + result[0]
plt.figure()
cm = dl.plot(mstar)
plt.colorbar(cm)

predicted_obs = ADV.parameter_to_observable_map(mstar.vector()[:])

discrepancy = np.linalg.norm(predicted_obs - ADV.obs)
noise_discrepancy = np.linalg.norm(ADV.noise)

print('areg=', areg)
print('discrepancy=', discrepancy)
print('noise_discrepancy=', noise_discrepancy)

#

plt.figure()
obs_func = dl.Function(ADV.Vh)
obs_func.vector()[:] = ADV.obs.reshape(-1)
cm = dl.plot(obs_func)
plt.colorbar(cm)
plt.title('obs')

#
# ADP = AdvectionDiffusionProblem(kappa, t_final, gamma)
#
# cmap = 'gray'  # 'binary_r' #'gist_gray' #'binary' # 'afmhot'
#
# plt.figure(figsize=(5, 5))
# cm = dl.plot(ADP.utrue_final, cmap=cmap)
# plt.axis('off')
# cm.set_clim(0.0, 1.0)
# # cm.extend = 'both'
# plt.colorbar(cm, fraction=0.046, pad=0.04)
# plt.gca().set_aspect('equal')
# plt.show()
#
# plt.set_cmap('viridis')
#
# plt.figure(figsize=(5, 5))
# p = np.array([0.25, 0.75])
# impulse_response0 = dl.Function(Vh)
# impulse_response0.vector()[:] = ADP.get_impulse_response(find_nearest_dof(p))
# cm = dl.plot(impulse_response0)
# plt.colorbar(cm)
# plt.title('impulse response near ' + str(p))
#
#
# ic_func = checkerboard_function(num_checkers_x, num_checkers_y, Vh)
# true_initial_condition = ic_func.vector()
#
# utrue_initial = dl.Function(Vh)
# utrue_initial.vector()[:] = true_initial_condition
#
# cmap = 'gray' #'binary_r' #'gist_gray' #'binary' # 'afmhot'
#
# plt.figure(figsize=(5,5))
# cm = dl.plot(utrue_initial, cmap=cmap)
# plt.axis('off')
# plt.colorbar(cm,fraction=0.046, pad=0.04)
# plt.gca().set_aspect('equal')
# plt.show()
#
