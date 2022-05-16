import numpy as np
import dolfin as dl
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
# from pathlib import Path
# import os

from nalger_helper_functions import *


class HeatInverseProblem:
    def __init__(me,
                 mesh_h=3e-2,
                 finite_element_order=1,
                 final_time=3e-4,
                 num_timesteps=35,
                 noise_level=5e-2,
                 mesh_type='circle',
                 initial_condition_type='angel_peak',
                 prior_correlation_length=0.05,
                 regularization_parameter=1e-1,
                 perform_checks=False,
                 make_plots=True,
                 save_plots=True,
                 conductivity_type='wiggly'):
        ########    INITIALIZE OPTIONS    ########

        me.mesh_h = mesh_h
        me.finite_element_order = finite_element_order
        me.final_time = final_time
        me.num_timesteps = num_timesteps
        me.noise_level = noise_level
        me.mesh_type = mesh_type
        me.conductivity_type = conductivity_type
        me.initial_condition_type = initial_condition_type
        me.prior_correlation_length = prior_correlation_length
        me.regularization_parameter = regularization_parameter

        me.perform_checks = perform_checks
        me.make_plots = make_plots
        me.save_plots = save_plots


        ########    MESH    ########

        if me.mesh_type == 'circle':
            mesh_center = np.array([0.5, 0.5])
            mesh_radius = 0.5
            me.mesh = circle_mesh(mesh_center, mesh_radius, mesh_h)
        elif me.mesh_type == 'square':
            me.mesh = dl.UnitSquareMesh(int(1./mesh_h)+1, int(1./mesh_h)+1)
        else:
            raise RuntimeError('mesh_type must be circle or square')


        ########    FUNCTION SPACE    ########

        me.V = dl.FunctionSpace(me.mesh, 'CG', finite_element_order)

        me.dof_coords = me.V.tabulate_dof_coordinates()
        me.N = me.V.dim()
        me.d = me.mesh.geometric_dimension()


        ########    THERMAL CONDUCTIVITY FIELD KAPPA    ########

        if me.conductivity_type == 'wiggly':
            me.kappa = wiggly_function(me.V)
        else:
            raise RuntimeError('conductivity_type must be wiggly')


        ########    TRUE INITIAL CONCENTRATION (INVERSION PARAMETER)    ########

        image_dir = get_project_root() / 'localpsf'

        if me.initial_condition_type == 'angel_peak':
            image_file = image_dir / 'angel_peak_badlands.png'
            me.u0_true = load_image_into_fenics(me.V, image_file)
        elif me.initial_condition_type == 'aces_building':
            image_file = image_dir / 'aces_building.png'
            me.u0_true = load_image_into_fenics(me.V, image_file)
        else:
            raise RuntimeError('initial_condition_type must be angel_peak or aces_building')


        ########    MASS AND STIFFNESS MATRICES    ########

        u_trial = dl.TrialFunction(me.V)
        v_test = dl.TestFunction(me.V)

        me.mass_form = u_trial * v_test * dl.dx
        me.mass_matrix = dl.assemble(me.mass_form)
        me.solve_mass_matrix_petsc = make_fenics_amg_solver(me.mass_matrix)

        me.stiffness_form = dl.inner(me.kappa * dl.grad(u_trial), dl.grad(v_test)) * dl.dx
        me.stiffness_matrix = dl.assemble(me.stiffness_form)


        ########    TIMESTEPPING OPERATORS    ########

        me.delta_t = me.final_time / me.num_timesteps
        me.Z_minus = me.mass_matrix
        me.Z_plus = me.mass_matrix + me.delta_t * me.stiffness_matrix

        me.solve_Z_plus = make_fenics_amg_solver(me.Z_plus)


        ########    TRUE FINAL CONCENTRATION AND NOISY OBSERVATIONS    ########

        me.uT_true = dl.Function(me.V)
        me.uT_true.vector()[:] = me.forward_map(me.u0_true.vector())

        uT_true_Mnorm = np.sqrt(me.uT_true.vector().inner(me.mass_matrix * me.uT_true.vector()))

        me.noise = dl.Function(me.V)
        me.noise.vector()[:] = np.random.randn(me.N)
        noise_Mnorm0 = np.sqrt(me.noise.vector().inner(me.mass_matrix * me.noise.vector()))
        me.noise.vector()[:] = me.noise_level * (uT_true_Mnorm / noise_Mnorm0) * me.noise.vector()

        me.noise_Mnorm = np.sqrt(me.noise.vector().inner(me.mass_matrix * me.noise.vector()))

        if me.perform_checks:
            print('noise_level=', noise_level, ', noise_Mnorm/uT_true_Mnorm=', me.noise_Mnorm/uT_true_Mnorm)

        me.uT_obs = dl.Function(me.V)
        me.uT_obs.vector()[:] = me.uT_true.vector()[:] + me.noise.vector()[:]


        ########    REGULARIZATION / PRIOR    ########

        K_form = dl.inner(dl.grad(u_trial), dl.grad(v_test)) * dl.dx
        K = dl.assemble(K_form)

        K_csr = csr_fenics2scipy(K)
        M_csr = csr_fenics2scipy(me.mass_matrix)

        diag_M = M_csr.diagonal()
        me.M_lumped_scipy = sps.diags(diag_M, 0).tocsr()
        me.iM_lumped_scipy = sps.diags(1. / diag_M, 0).tocsr()

        me.sqrt_R0_scipy = me.prior_correlation_length ** 2 * K_csr + M_csr
        me.R0_scipy = me.sqrt_R0_scipy.T * (me.iM_lumped_scipy * me.sqrt_R0_scipy)

        me.M_lumped = csr_scipy2fenics(me.M_lumped_scipy)
        me.iM_lumped = csr_scipy2fenics(me.iM_lumped_scipy)
        me.sqrt_R0 = csr_scipy2fenics(me.sqrt_R0_scipy)
        me.R0 = csr_scipy2fenics(me.R0_scipy)

        me.solve_sqrt_R0 = make_fenics_amg_solver(me.sqrt_R0)
        me.solve_R0 = make_fenics_amg_solver(me.R0)


        ########    Hessian and Regularization as linear operators    ########

        me.Hd_linop = spla.LinearOperator((me.N, me.N), matvec=me.apply_Hd_numpy)
        me.H_linop = spla.LinearOperator((me.N, me.N), matvec=me.apply_H_numpy)
        me.R_linop = spla.LinearOperator((me.N, me.N), matvec = me.apply_R_numpy)
        me.solve_R_linop = spla.LinearOperator((me.N, me.N), matvec=me.solve_R_numpy)



        ########    FINITE DIFFERENCE CHECKS    ########

        if perform_checks:
            me.perform_adjoint_correctness_check()
            me.perform_finite_difference_checks()
            me.check_R_solver()

    @property
    def options(me):
        return {'mesh_h': me.mesh_h,
                'finite_element_order': me.finite_element_order,
                'final_time': me.final_time,
                'num_timesteps': me.num_timesteps,
                'noise_level': me.noise_level,
                'mesh_type': me.mesh_type,
                'conductivity_type': me.conductivity_type,
                'initial_condition_type': me.initial_condition_type,
                'prior_correlation_length': me.prior_correlation_length,
                'regularization_parameter': me.regularization_parameter}

    def __hash__(me):
        return hash(tuple(me.options.items()))

    def apply_R_petsc(me, p_petsc):
        return me.regularization_parameter * (me.R0 * p_petsc)

    def solve_R_petsc(me, p_petsc):
        # return me.solve_R0(p_petsc) / me.regularization_parameter
        return me.solve_sqrt_R0(me.M_lumped * me.solve_sqrt_R0(p_petsc)) / me.regularization_parameter

    def apply_R_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_R_petsc, p_numpy)

    def solve_R_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.solve_R_petsc, p_numpy)

    def check_R_solver(me):
        x = np.random.randn(me.N)
        err_R_solver = np.linalg.norm(me.solve_R_numpy(me.apply_R_numpy(x)) - x) / np.linalg.norm(x)
        print('err_R_solver=', err_R_solver)

    def forward_map(me, u0_petsc):
        uT_petsc = dl.Vector(u0_petsc)
        for k in range(me.num_timesteps):
            uT_petsc = me.solve_Z_plus(me.Z_minus * uT_petsc)
        return uT_petsc

    def adjoint_map(me, vT_petsc):
        v0_petsc = dl.Vector(vT_petsc)
        for k in range(me.num_timesteps):
            v0_petsc = me.Z_minus * me.solve_Z_plus(v0_petsc)
        return v0_petsc

    def misfit_objective(me, u0_petsc):
        uT_petsc = me.forward_map(u0_petsc)
        discrepancy = uT_petsc - me.uT_obs.vector()
        J = 0.5 * discrepancy.inner(me.mass_matrix * discrepancy)
        return J

    def regularization_objective(me, u0_petsc):
        return 0.5 * u0_petsc.inner(me.apply_R_petsc(u0_petsc))

    def objective(me, u0_petsc):
        return me.misfit_objective(u0_petsc) + me.regularization_objective(u0_petsc)

    def morozov_discrepancy(me, u0_petsc):
        return np.sqrt(2.0 * me.misfit_objective(u0_petsc))

    def misfit_gradient(me, u0_petsc):
        uT_petsc = me.forward_map(u0_petsc)
        discrepancy = uT_petsc - me.uT_obs.vector()
        return me.adjoint_map(me.mass_matrix * discrepancy)

    def regularization_gradient(me, u0_petsc):
        return me.apply_R_petsc(u0_petsc)

    def gradient(me, u0_petsc):
        return me.misfit_gradient(u0_petsc) + me.regularization_gradient(u0_petsc)

    def apply_misfit_hessian_petsc(me, p_petsc):
        return me.adjoint_map(me.mass_matrix * me.forward_map(p_petsc))

    def apply_regularization_hessian_petsc(me, p_petsc):
        return me.apply_R_petsc(p_petsc)

    def apply_hessian_petsc(me, p_petsc):
        return me.apply_misfit_hessian_petsc(p_petsc) + me.apply_regularization_hessian_petsc(p_petsc)

    def apply_mass_matrix_petsc(me, p_petsc):
        return me.mass_matrix * p_petsc

    def g_petsc(me, u0_petsc):
        return me.gradient(u0_petsc)

    def g_numpy(me, u0_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.g_petsc, u0_numpy)

    def apply_M_petsc(me, p_petsc):
        return me.apply_mass_matrix_petsc(p_petsc)

    def solve_M_petsc(me, p_petsc):
        return me.solve_mass_matrix_petsc(p_petsc)

    def apply_Hr_petsc(me, p_petsc):
        return me.apply_regularization_hessian_petsc(p_petsc)

    def apply_Hd_petsc(me, p_petsc):
        return me.apply_misfit_hessian_petsc(p_petsc)

    def apply_H_petsc(me, p_petsc):
        return me.apply_hessian_petsc(p_petsc)

    def apply_iM_Hd_iM_petsc(me, p_petsc):
        return me.solve_M_petsc(me.apply_Hd_petsc(me.solve_M_petsc(p_petsc)))

    def apply_Hd_iM_petsc(me, p_petsc):
        return me.apply_Hd_petsc(me.solve_M_petsc(p_petsc))

    def apply_iM_Hd_petsc(me, p_petsc):
        return me.solve_M_petsc(me.apply_Hd_petsc(p_petsc))

    def petsc2numpy(me, p_petsc):
        return p_petsc[:]

    def numpy2petsc(me, p_numpy):
        p = dl.Function(me.V)
        p.vector()[:] = p_numpy
        return p.vector()

    def numpy_wrapper_for_petsc_function_call(me, func_petsc, p_numpy):
        return me.petsc2numpy(func_petsc(me.numpy2petsc(p_numpy)))

    def apply_misfit_hessian_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_misfit_hessian_petsc, p_numpy)

    def apply_mass_matrix_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_mass_matrix_petsc, p_numpy)

    def solve_mass_matrix_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.solve_mass_matrix_petsc, p_numpy)

    def apply_M_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_M_petsc, p_numpy)

    def solve_M_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.solve_M_petsc, p_numpy)

    def apply_Hd_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_Hd_petsc, p_numpy)

    def apply_Hr_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_Hr_petsc, p_numpy)

    def apply_H_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_H_petsc, p_numpy)

    def apply_iM_Hd_iM_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_iM_Hd_iM_petsc, p_numpy)

    def apply_Hd_iM_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_Hd_iM_petsc, p_numpy)

    def apply_iM_Hd_numpy(me, p_numpy):
        return me.numpy_wrapper_for_petsc_function_call(me.apply_iM_Hd_petsc, p_numpy)

    def perform_adjoint_correctness_check(me):
        x = dl.Function(me.V)
        y = dl.Function(me.V)
        x.vector()[:] = np.random.randn(me.N)
        y.vector()[:] = np.random.randn(me.N)
        adjoint_err = np.abs(me.forward_map(x.vector()).inner(y.vector())
                             - x.vector().inner(me.adjoint_map(y.vector())))
        print('adjoint_err=', adjoint_err)

    def perform_finite_difference_checks(me):
        u0 = dl.Function(me.V).vector()
        u0[:] = np.random.randn(me.N)

        J = me.objective(u0)
        g = me.gradient(u0)

        du = dl.Function(me.V).vector()
        du[:] = np.random.randn(me.N)

        ss = np.logspace(-15, 0, 11)
        grad_errs = np.zeros(len(ss))
        for k in range(len(ss)):
            s = ss[k]
            u0_2 = u0 + s * du

            J2 = me.objective(u0_2)
            dJ_diff = (J2 - J) / s
            dJ = g.inner(du)
            grad_err = np.abs(dJ - dJ_diff) / np.abs(dJ_diff)
            grad_errs[k] = grad_err

            print('s=', s, ', grad_err=', grad_err)

        plt.figure()
        plt.loglog(ss, grad_errs)
        plt.title('gradient finite difference check')
        plt.xlabel('s')
        plt.ylabel('relative gradient error')

        u0_2 = dl.Function(me.V).vector()
        u0_2[:] = np.random.randn(me.N)
        dg = me.apply_hessian_petsc(u0_2 - u0)
        g2 = me.gradient(u0_2)
        dg_diff = g2 - g
        hess_err = np.linalg.norm(dg[:] - dg_diff[:]) / np.linalg.norm(dg_diff[:])
        print('hess_err=', hess_err)

    def interactive_hessian_impulse_response_plot(me):
        interactive_impulse_response_plot(me.apply_iM_Hd_iM_petsc, me.V)


def wiggly_function(V0):
    n=150
    mesh = dl.RectangleMesh(dl.Point(-1.,-1.), dl.Point(2., 2.), n,n)
    V = dl.FunctionSpace(mesh, 'CG', 2)
#     u = dl.interpolate(dl.Expression('2*(0.125 + 0.1*sin(30*x[0]))',domain=mesh, degree=5), V)
    u = dl.interpolate(dl.Expression('2*(0.2 + 0.1*sin(30*x[0]))',domain=mesh, degree=5), V)
    old_coords = mesh.coordinates()

    xx0 = old_coords[:,0]
    yy0 = old_coords[:,1]

    xx1 = xx0
    yy1 = yy0 + 0.2 * np.cos(3.5*xx0)

    xx2 = yy1 + 0.3 * xx1
    yy2 = xx1 + 0.3 * np.sin(3.5*(yy1-0.35))

    xx3 = (xx2 + yy2)
    yy3 = (xx2 - yy2) + 0.2 * np.cos(4*(xx2 + yy2))

    new_coords = np.array([xx3, yy3]).T

    mesh.coordinates()[:] = new_coords

    u0 = dl.interpolate(u, V0)
    return u0


def solve_heat_inverse_problem_morozov(HIP, Hd_hmatrix, R0_hmatrix, a_reg0=1e0, a_reg_min=1e-8, gamma=3e-1, tol=1e-10,
                                       make_morozov_plot=True):
    a_reg = a_reg0
    u0 = dl.Function(HIP.V)
    morozov_discrepancies = list()
    noise_Mnorms = list()
    regularization_parameters = list()
    while a_reg > a_reg_min:
        regularization_parameters.append(a_reg)
        HIP.regularization_parameter = a_reg

        g0_numpy = HIP.g_numpy(np.zeros(HIP.N))
        H_hmatrix = Hd_hmatrix + a_reg * R0_hmatrix
        iH_hmatrix = H_hmatrix.inv()

        u0_numpy, info, residuals = custom_cg(HIP.H_linop, -g0_numpy,
                                              M=iH_hmatrix.as_linear_operator(),
                                              tol=tol, maxiter=500)
        u0.vector()[:] = u0_numpy

        morozov_discrepancy = HIP.morozov_discrepancy(u0.vector())
        noise_Mnorm = HIP.noise_Mnorm

        morozov_discrepancies.append(morozov_discrepancy)
        noise_Mnorms.append(noise_Mnorm)

        print('a_reg=', a_reg, ', noise_Mnorm=', noise_Mnorm, ', morozov_discrepancy=', morozov_discrepancy)
        if morozov_discrepancy < noise_Mnorm:
            break

        a_reg = gamma * a_reg

    regularization_parameters = np.array(regularization_parameters[::-1])
    morozov_discrepancies = np.array(morozov_discrepancies[::-1])
    noise_Mnorms = np.array(noise_Mnorms[::-1])
    a_reg_morozov = inverse_of_monotone_increasing_piecewise_loglinear_function(noise_Mnorms[0],
                                                                                regularization_parameters,
                                                                                morozov_discrepancies)

    print('a_reg_morozov=', a_reg_morozov)

    if make_morozov_plot:
        plt.figure()
        plt.loglog(regularization_parameters, morozov_discrepancies)
        plt.loglog(regularization_parameters, noise_Mnorms)
        plt.title('Morozov discrepancy')

    return u0, a_reg_morozov


def inverse_of_monotone_increasing_piecewise_loglinear_function(y_target, xx, yy):
    xx = np.array(xx)
    yy = np.array(yy)
    ind = np.argwhere(yy > y_target)[0, 0]
    y_minus = yy[ind-1]
    y_plus = yy[ind]
    x_minus = xx[ind-1]
    x_plus = xx[ind]
    slope_log = (np.log(y_plus) - np.log(y_minus)) / (np.log(x_plus) - np.log(x_minus))
    dx_log = (np.log(y_target) - np.log(y_minus)) / slope_log
    x_target_log = np.log(x_minus) + dx_log
    return np.exp(x_target_log)