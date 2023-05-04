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
from localpsf.bilaplacian_regularization import BiLaplacianRegularization
from localpsf.morozov_discrepancy import compute_morozov_regularization_parameter
import nalger_helper_functions as nhf

import hlibpro_python_wrapper as hpro
from localpsf.product_convolution_kernel import ProductConvolutionKernel
from localpsf.product_convolution_hmatrix import make_hmatrix_from_kernel


###############################    OPTIONS    ###############################

num_refinements = 2
# kappa=5.e-5 # marginally too small
# kappa=1e-3 # standard for hippylib. Quite smoothing
# kappa = 3.e-4 # Good value
# kappa = 1.e-4 # pretty small but still good
# kappa = 2.e-3 # big but good
kappa = 5e-4
# kappa = 1.e-2

t_final        = 0.5 #1.0
dt             = .1 #0.05

gamma = 1.
# delta = 8.
# gamma = 0.25
# gamma = 0.1
# delta = 8.
prior_correlation_length = 0.25

num_checkers_x = 6
num_checkers_y = 6

rel_noise=0.01


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

dt = 0.1
t_init = 0.
obs_coords = Vh.tabulate_dof_coordinates()  # observe everywhere
print("Number of observation points: {0}".format(obs_coords.shape[0]))


@dataclass
class AdvInverseProblemUnregularized:
    mesh_and_function_space: AdvMeshAndFunctionSpace
    problem: TimeDependentAD
    misfit: SpaceTimePointwiseStateObservation
    true_initial_condition: np.ndarray
    obs: np.ndarray
    true_obs: np.ndarray

    @cached_property
    def N(me) -> int:
        return me.mesh_and_function_space.Vh.dim()

    def noise(me) -> np.ndarray:
        return me.obs - me.true_obs

    def update_noise(me, new_noise: np.ndarray) -> None:
        me.obs[:] = me.true_obs + new_noise

    def misfit_cost(me, m_vec: np.ndarray) -> float:
        [u, m, p] = me.problem.generate_vector()
        m[:] = m_vec.copy()
        me.problem.solveFwd(u, [u, m, p])
        me.problem.solveAdj(p, [u, m, p])
        total_cost, reg_cost, misfit_cost = me.problem.cost([u, m, p])
        return misfit_cost

    def misfit_gradient(me, m_vec: np.ndarray) -> np.ndarray:
        [u, m, p] = me.problem.generate_vector()
        m[:] = m_vec.copy()
        me.problem.solveFwd(u, [u, m, p])
        me.problem.solveAdj(p, [u, m, p])
        mg = me.problem.generate_vector(PARAMETER)
        grad_norm = me.problem.evalGradientParameter([u, m, p], mg, misfit_only=True)
        return mg[:].copy()

    def apply_misfit_hessian(me, x: np.ndarray) -> np.ndarray:
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


    def __init__(me, kappa, t_final, gamma):
        print('AdvectionDiffusionMisfitStuff: kappa=', kappa, ', t_final=', t_final)
        me.kappa = kappa
        me.t_final = t_final
        me._gamma = gamma

        me.REG = BiLaplacianRegularization(me._gamma, prior_correlation_length,
                                           dl.Function(Vh), prior_mean_func)

        me.simulation_times = np.arange(t_init, me.t_final + .5 * dt, dt)
        me.observation_times = np.array([me.t_final])

        me.misfit = SpaceTimePointwiseStateObservation(Vh,
                                                       me.observation_times,
                                                       obs_coords)

        fake_gamma = 1.0
        fake_delta = 1.0

        fake_prior = BiLaplacianPrior(Vh, fake_gamma, fake_delta, robin_bc=True)

        me.problem = TimeDependentAD(mesh, [Vh, Vh, Vh], fake_prior,
                                     me.misfit, me.simulation_times,
                                     wind_velocity, True, kappa=me.kappa)

        me.utrue = me.problem.generate_vector(STATE)
        x = [me.utrue, true_initial_condition, None]
        me.problem.solveFwd(x[STATE], x)
        me.misfit.observe(x, me.misfit.d)

        me.true_obs_numpy = me.misfit.d.data[0][:].copy()

        MAX = me.misfit.d.norm("linf", "linf")
        noise_std_dev = rel_noise * MAX
        parRandom.normal_perturb(noise_std_dev, me.misfit.d)
        me.misfit.noise_variance = noise_std_dev * noise_std_dev

        me.obs_numpy = me.misfit.d.data[0][:].copy()

        me.noise_numpy = me.true_obs_numpy - me.obs_numpy
        me.noise_norm = np.linalg.norm(me.noise_numpy)
        print('noise_norm=', me.noise_norm)

        uf_petsc = dl.Function(Vh).vector()
        x[0].retrieve(uf_petsc, t_final)

        me.utrue_final = dl.Function(Vh)
        me.utrue_final.vector()[:] = uf_petsc

        me.H = ReducedHessian(me.problem, misfit_only=True)

        me.apply_Hr = me.REG.apply_hessian_petsc
        me.solve_Hr = me.REG.solve_hessian_petsc

    @property
    def gamma(me):
        return me._gamma

    def update_gamma(me, new_gamma):
        me._gamma = gamma
        me.REG.update_gamma(new_gamma)

    def eval_misfit_objective(me, m_petsc):
        [u, m, p] = me.problem.generate_vector()
        m[:] = m_petsc
        me.problem.solveFwd(u, [u, m, p])
        me.problem.solveAdj(p, [u, m, p])
        total_cost, reg_cost, misfit_cost = me.problem.cost([u, m, p])
        return misfit_cost

    def eval_misfit_gradient(me, m_petsc, return_gradnorm=False):
        [u, m, p] = me.problem.generate_vector()
        m[:] = m_petsc
        me.problem.solveFwd(u, [u, m, p])
        me.problem.solveAdj(p, [u, m, p])
        mg = me.problem.generate_vector(PARAMETER)
        grad_norm = me.problem.evalGradientParameter([u, m, p], mg, misfit_only=True)
        if return_gradnorm:
            return mg, grad_norm
        else:
            return mg

    def apply_Hd(me, x_petsc):
        y_petsc = dl.Function(Vh).vector()
        me.H.mult(x_petsc, y_petsc)
        return y_petsc

    def apply_Hd_numpy(me, x_numpy):
        x_petsc = dl.Function(Vh).vector()
        x_petsc[:] = x_numpy
        return me.apply_Hd(x_petsc)[:]

    def eval_regularization_objective(me, m_petsc):
        me.REG.parameter.vector()[:] = m_petsc
        return me.REG.cost()

    def eval_objective(me, m_petsc):
        return me.eval_misfit_objective(m_petsc) + me.eval_regularization_objective(m_petsc)

    def eval_regularization_gradient(me, m_petsc):
        me.REG.parameter.vector()[:] = m_petsc
        return me.REG.gradient_petsc()

    def eval_gradient(me, m_petsc):
        return me.eval_misfit_gradient(m_petsc) + me.eval_regularization_gradient(m_petsc)

    def eval_gradient_numpy(me, m_numpy):
        m_petsc = dl.Function(Vh).vector()
        m_petsc[:] = m_numpy
        return me.eval_gradient(m_petsc)[:]

    def apply_H(me, x_petsc):
        return me.apply_Hd(x_petsc) + me.apply_Hr(x_petsc)

    def apply_H_numpy(me, x_numpy):
        x_petsc = dl.Function(Vh).vector()
        x_petsc[:] = x_numpy
        return me.apply_H(x_petsc)[:]

    def apply_Hr_numpy(me, x_numpy):
        x_petsc = dl.Function(Vh).vector()
        x_petsc[:] = x_numpy
        return me.apply_Hr(x_petsc)[:]

    def solve_Hr_numpy(me, x_numpy):
        x_petsc = dl.Function(Vh).vector()
        x_petsc[:] = x_numpy
        return me.solve_Hr(x_petsc)[:]

    def get_impulse_response(me, k: int, lumped_mass=False
                             ) -> np.ndarray:  # impulse response at kth dof location
        ek = np.zeros(Vh.dim())
        ek[k] = 1.0
        if lumped_mass:
            return solve_ML_numpy(me.apply_Hd_numpy(solve_ML_numpy(ek)))
        else:
            return solve_M_numpy(me.apply_Hd_numpy(solve_M_numpy(ek)))

    def compute_morozov_discrepancy(me, m_star_numpy):
        m_star = dl.Function(Vh).vector()
        m_star[:] = m_star_numpy
        u_star = me.problem.generate_vector(STATE)
        x = [u_star, m_star, None]
        me.problem.solveFwd(x[STATE], x)
        new_obs = me.misfit.d.copy()
        new_obs.zero()
        me.misfit.observe(x, new_obs)
        return np.linalg.norm(new_obs.data[0][:] - me.obs_numpy)


ADP = AdvectionDiffusionProblem(kappa, t_final, gamma)

cmap = 'gray'  # 'binary_r' #'gist_gray' #'binary' # 'afmhot'

plt.figure(figsize=(5, 5))
cm = dl.plot(ADP.utrue_final, cmap=cmap)
plt.axis('off')
cm.set_clim(0.0, 1.0)
# cm.extend = 'both'
plt.colorbar(cm, fraction=0.046, pad=0.04)
plt.gca().set_aspect('equal')
plt.show()

plt.set_cmap('viridis')

plt.figure(figsize=(5, 5))
p = np.array([0.25, 0.75])
impulse_response0 = dl.Function(Vh)
impulse_response0.vector()[:] = ADP.get_impulse_response(find_nearest_dof(p))
cm = dl.plot(impulse_response0)
plt.colorbar(cm)
plt.title('impulse response near ' + str(p))


ic_func = checkerboard_function(num_checkers_x, num_checkers_y, Vh)
true_initial_condition = ic_func.vector()

utrue_initial = dl.Function(Vh)
utrue_initial.vector()[:] = true_initial_condition

cmap = 'gray' #'binary_r' #'gist_gray' #'binary' # 'afmhot'

plt.figure(figsize=(5,5))
cm = dl.plot(utrue_initial, cmap=cmap)
plt.axis('off')
plt.colorbar(cm,fraction=0.046, pad=0.04)
plt.gca().set_aspect('equal')
plt.show()

