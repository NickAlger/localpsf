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
from .bilaplacian_regularization_lumped import BilaplacianRegularization, BiLaplacianCovariance, make_bilaplacian_covariance
from .morozov_discrepancy import compute_morozov_regularization_parameter
from .inverse_problem_objective import PSFHessianPreconditioner
import nalger_helper_functions as nhf
from .filesystem_helpers import localpsf_root

import hlibpro_python_wrapper as hpro
from .product_convolution_kernel import ProductConvolutionKernel
from .product_convolution_hmatrix import make_hmatrix_from_kernel


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
    mfile_name = str(localpsf_root) + "/numerical_examples/advection_diffusion/ad_20.xml"
    # mesh = dl.Mesh("ad_20.xml")
    mesh = dl.Mesh(mfile_name)
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
    # s = '1.0 - (' + sx + sy + '0.5 + 0.5)'
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
    psf_preconditioner: PSFHessianPreconditioner

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

    @cached_property
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

    def vec2func(me, u_vec: np.ndarray) -> dl.Function:
        assert(u_vec.shape == (me.N,))
        u_func = dl.Function(me.Vh)
        u_func.vector()[:] = u_vec.copy()
        return u_func

    @cached_property
    def true_initial_condition_func(me) -> dl.Function:
        return me.vec2func(me.true_initial_condition)

    @cached_property
    def obs_func(me) -> dl.Function:
        return me.vec2func(me.obs[-1,:])

    @cached_property
    def true_obs_func(me) -> dl.Function:
        return me.vec2func(me.true_obs[-1,:])

    @cached_property
    def noise_func(me) -> dl.Function:
        return me.vec2func(me.noise)

    def solve_inverse_problem(
            me, areg: float, cg_options: typ.Dict[str, typ.Any]=None,
    ) -> typ.Tuple[np.ndarray, typ.Tuple]:
        cg_options2 = {'tol' : 1e-8,
                       'display' : True,
                       'maxiter' : 2000,
                       }
        if cg_options is not None:
            cg_options2.update(cg_options)

        m0 = np.zeros(me.N)
        g0 = me.gradient(m0, areg)

        P_linop = spla.LinearOperator(
            (me.N, me.N),
            matvec=lambda x: me.psf_preconditioner.solve_hessian_preconditioner(x, areg))

        H_linop = spla.LinearOperator((me.N, me.N), matvec=lambda x: me.apply_hessian(x, areg))
        result = nhf.custom_cg(H_linop, -g0, M=P_linop, **cg_options2)

        mstar_vec = m0 + result[0]
        return mstar_vec, result

    def compute_morozov_discrepancy(me, areg: float, cg_options: typ.Dict[str, typ.Any]=None) -> float:
        cg_options2 = {'tol' : 1e-5}
        if cg_options is not None:
            cg_options2.update(cg_options)

        mstar_vec, _ = me.solve_inverse_problem(areg, cg_options=cg_options2)
        predicted_obs = me.parameter_to_observable_map(mstar_vec)

        discrepancy = np.linalg.norm(predicted_obs - me.obs)
        noise_discrepancy = np.linalg.norm(me.noise)

        print('areg=', areg, ', discrepancy=', discrepancy, ', noise_discrepancy=', noise_discrepancy)
        return discrepancy

    def compute_morozov_regularization_parameter(
            me, areg_initial_guess: float,
            morozov_options: typ.Dict[str, typ.Any]=None,
            cg_options: typ.Dict[str, typ.Any]=None
    ) -> typ.Tuple[float,       # optimal morozov regularization parmaeter
                   np.ndarray,  # all regularization parameters
                   np.ndarray]: # all morozov discrepancies:
        f = lambda a: me.compute_morozov_discrepancy(a, cg_options=cg_options)
        morozov_options2 = dict()
        if morozov_options is not None:
            morozov_options2.update(morozov_options)

        return compute_morozov_regularization_parameter(
            areg_initial_guess, f, me.noise_norm, **morozov_options2)


def make_adv_universe(
        noise_level: float, # 0.05, i.e., 5% noise, is typical
        kappa: float, # 1e-3 is default for hippylib
        t_final: float,
        reynolds_number: float=1e2,
        num_mesh_refinements=2,
        t_init = 0.0,
        dt: float=0.1,
        prior_correlation_length: float=0.25,
        num_checkers_x: int=8,
        num_checkers_y: int=8,
        smoothing_time: float=1e-4,
        admissibility_eta=1.0
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

    ic_func = checkerboard_function(num_checkers_x, num_checkers_y, Vh, smoothing_time=smoothing_time)
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

    print('Making row and column cluster trees')
    ct = hpro.build_cluster_tree_from_pointcloud(mesh_and_function_space.dof_coords, cluster_size_cutoff=50)

    print('Making block cluster trees')
    bct = hpro.build_block_cluster_tree(ct, ct, admissibility_eta=admissibility_eta)

    HR_hmatrix = regularization.Cov.make_invC_hmatrix(bct, 1.0)

    psf_preconditioner = PSFHessianPreconditioner(
        None, Vh, mesh_and_function_space.mass_lumps,
        HR_hmatrix, display=True)

    ADV = AdvUniverse(
        mesh_and_function_space, wind_object, problem, misfit, regularization,
        true_initial_condition, obs, true_obs, psf_preconditioner)

    ADV.psf_preconditioner.apply_misfit_gauss_newton_hessian = ADV.apply_misfit_hessian

    return ADV
