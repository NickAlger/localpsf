import numpy as np
import fenics
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from fenics_to_scipy_sparse_csr_conversion import convert_fenics_csr_matrix_to_scipy_csr_matrix, vec2fct
from fenics_interactive_impulse_response_plot import fenics_interactive_impulse_response_plot
import mshr
import matplotlib.pyplot as plt


def make_amg_solver(A_petsc):
    prec = fenics.PETScPreconditioner('hypre_amg')
    fenics.PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = fenics.PETScKrylovSolver('cg', prec)
    solver.set_operator(A_petsc)

    def solve_A(b_petsc, atol=0.0, rtol=1e-10, maxiter=100, verbose=False):
        x_petsc = fenics.Vector(b_petsc)
        solver.parameters['absolute_tolerance'] = atol
        solver.parameters['relative_tolerance'] = rtol
        solver.parameters['maximum_iterations'] = maxiter
        solver.parameters['monitor_convergence'] = verbose
        return solver.solve(x_petsc, b_petsc)

    return solve_A


def random_function(function_space_V):
    x = fenics.Function(function_space_V)
    x.vector()[:] = np.random.randn(function_space_V.dim())
    return x


class HeatInverseProblem:
    def __init__(me, mesh_h=1e-2, finite_element_order=1, final_time_T=3e-4, num_timesteps=35, noise_level=5e-2,
                 perform_checks=True, uniform_kappa=False, lumped_mass_matrix=False):
        me.mesh_h = mesh_h
        me.T = final_time_T
        me.num_timesteps = num_timesteps
        me.lumped_mass_matrix = lumped_mass_matrix

        me.n = 1. / me.mesh_h
        me.delta_t = me.T / me.num_timesteps

        outer_circle = mshr.Circle(fenics.Point(0.5, 0.5), 0.5)
        mesh = mshr.generate_mesh(outer_circle, me.n)

        me.V = fenics.FunctionSpace(mesh, 'CG', finite_element_order)
        me.d = mesh.geometric_dimension()
        me.N = me.V.dim()

        u_trial = fenics.TrialFunction(me.V)
        v_test = fenics.TestFunction(me.V)

        mass_form = u_trial * v_test * fenics.dx
        me.M = fenics.assemble(mass_form)

        me.solve_M = make_amg_solver(me.M)

        if uniform_kappa:
            me.kappa = fenics.Constant(np.eye(me.d))
        else:
            me.kappa = random_conductivity_field(me.V)

        stiffness_form = fenics.inner(me.kappa * fenics.grad(u_trial), fenics.grad(v_test)) * fenics.dx
        me.A = fenics.assemble(stiffness_form)

        me.Z_minus = me.M
        me.Z_plus = me.M + me.delta_t * me.A

        me.solve_Z_plus = make_amg_solver(me.Z_plus)

        me.u0_true = random_function(me.V)

        me.uT_true = fenics.Function(me.V)
        me.uT_true.vector()[:] = me.forward_map(me.u0_true.vector())

        normalized_noise = np.random.randn(me.N) / np.sqrt(me.N)
        me.uT_obs = fenics.Function(me.V)
        me.uT_obs.vector()[:] = me.uT_true.vector()[:] + noise_level * normalized_noise * np.linalg.norm(me.uT_true.vector()[:])

        if perform_checks:
            plt.figure()
            fenics.plot(me.uT_true)
            plt.title('uT_true')

            me.perform_adjoint_correctness_check()
            me.perform_finite_difference_checks()


    def forward_map(me, u0_petsc):
        uT_petsc = fenics.Vector(u0_petsc)
        for k in range(me.num_timesteps):
            uT_petsc = me.solve_Z_plus(me.Z_minus * uT_petsc)
        return uT_petsc

    def adjoint_map(me, vT_petsc):
        v0_petsc = fenics.Vector(vT_petsc)
        for k in range(me.num_timesteps):
            v0_petsc = me.Z_minus * me.solve_Z_plus(v0_petsc)
        return v0_petsc

    def objective(me, u0_petsc):
        uT_petsc = me.forward_map(u0_petsc)
        discrepancy = uT_petsc - me.uT_obs.vector()
        J = 0.5 * discrepancy.inner(me.M * discrepancy)
        return J

    def gradient(me, u0_petsc):
        uT_petsc = me.forward_map(u0_petsc)
        discrepancy = uT_petsc - me.uT_obs
        return me.adjoint_map(me.M * discrepancy)

    def apply_hessian(me, p_petsc):
        return me.adjoint_map(me.M * me.forward_map(p_petsc))

    def perform_adjoint_correctness_check(me):
        x = fenics.Function(me.V)
        y = fenics.Function(me.V)
        x.vector()[:] = np.random.randn(me.N)
        y.vector()[:] = np.random.randn(me.N)
        adjoint_err = np.abs(me.forward_map(x.vector()).inner(y.vector())
                             - x.vector().inner(me.adjoint_map(y.vector())))
        print('adjoint_err=', adjoint_err)

    def perform_finite_difference_checks(me):
        u0 = np.random.randn(me.N)

        J = me.objective(u0)
        g = me.gradient(u0)

        du = np.random.randn(me.N)

        ss = np.logspace(-15, 0, 11)
        grad_errs = np.zeros(len(ss))
        for k in range(len(ss)):
            s = ss[k]
            u0_2 = u0 + s * du

            J2 = me.objective(u0_2)
            dJ_diff = (J2 - J) / s
            dJ = np.dot(g, du)
            grad_err = np.abs(dJ - dJ_diff) / np.abs(dJ_diff)
            grad_errs[k] = grad_err

            print('s=', s, ', grad_err=', grad_err)

        plt.figure()
        plt.loglog(ss, grad_errs)
        plt.title('gradient finite difference check')
        plt.xlabel('s')
        plt.ylabel('relative gradient error')

        u0_2 = np.random.randn(me.N)
        dg = me.apply_hessian(u0_2 - u0)
        g2 = me.gradient(u0_2)
        dg_diff = g2 - g
        hess_err = np.linalg.norm(dg - dg_diff) / np.linalg.norm(dg_diff)
        print('hess_err=', hess_err)

    def interactive_hessian_impulse_response_plot(me):
        iM_H_iM = lambda x: me.solve_M(me.apply_hessian(me.solve_M(x)))
        fenics_interactive_impulse_response_plot(iM_H_iM, me.V)

    def M_norm(me, x):
        return np.sqrt(np.dot(x, me.M * x))


def random_spd_matrix(d):
    U,ss,_ = np.linalg.svd(np.random.randn(d,d))
    A = np.dot(U, np.dot(np.diag(ss**3), U.T))
    return A


def random_conductivity_field(V):
    u_trial = fenics.TrialFunction(V)
    v_test = fenics.TestFunction(V)

    mass_form = u_trial * v_test * fenics.dx
    M = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(mass_form))

    stiffness_form = fenics.inner(fenics.grad(u_trial), fenics.grad(v_test)) * fenics.dx
    K = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(stiffness_form))

    smoother = spla.factorized(M + K)
    def random_smooth_vec(min=0.01, max=1):
        w0_vec = smoother(np.random.rand(V.dim()))
        m = np.min(w0_vec)
        M = np.max(w0_vec)
        w_vec = min + (w0_vec - m) * (max - min) / (M - m)
        return w_vec

    def random_smooth_partition_functions(nspd):
        ww0 = np.array([random_smooth_vec()**6 for _ in range(nspd)])
        ww = ww0 / np.sum(ww0, axis=0)
        ww_fct = [fenics.Function(V) for _ in range(nspd)]
        for k in range(nspd):
            ww_fct[k].vector()[:] = np.copy(ww[k,:])
        return ww_fct

    # C0 = fenics.Constant(np.array([[2.5, 1.5],[1.5,2.5]]))
    # C1 = fenics.Constant(np.array([[4,0],[0,0.5]]))
    # C2 = fenics.Constant(np.array([[0.5,-0.5],[-0.5,2.5]]))
    # conductivity_matrices = [C0, C1, C2]
    #
    # C0 = fenics.Constant(np.array([[10, 0.0],[0.0,0.1]]))
    # C1 = fenics.Constant(np.array([[0.1, 0.0],[0.0,10]]))
    # conductivity_matrices = [C0, C1]

    d = V.mesh().geometric_dimension()
    conductivity_matrices = [fenics.Constant(random_spd_matrix(d)) for _ in range(3)]
    nspd = len(conductivity_matrices)
    smooth_conductivity_weights = random_smooth_partition_functions(nspd)

    kappa = fenics.Constant(np.zeros((d, d)))
    for k in range(nspd):
        C = conductivity_matrices[k]
        w = smooth_conductivity_weights[k]
        kappa = kappa + C * w

    return kappa