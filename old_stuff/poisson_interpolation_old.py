import numpy as np
import fenics
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from fenics_to_scipy_sparse_csr_conversion import *
import matplotlib.pyplot as plt

run_test = True

def poisson_squared_weighting_functions(function_space_V, points_inds_ii, test_pinvA=False):
    V = function_space_V
    ii = point_inds_ii

    num_pts = len(point_inds_ii)
    N = V.dim()

    u_trial = fenics.TrialFunction(V)
    v_test = fenics.TestFunction(V)

    stiffness_form = fenics.inner(fenics.grad(u_trial), fenics.grad(v_test)) * fenics.dx
    A = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(stiffness_form))

    mass_form = u_trial * v_test * fenics.dx
    M = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(mass_form))
    solve_M = spla.factorized(M)

    one_dual_vec = fenics.assemble(fenics.Constant(1.0)*v_test*fenics.dx)[:].reshape((1,-1))
    one_vec = solve_M(one_dual_vec.reshape((-1,1)))
    # one_vec = one_dual_vec.reshape((-1, 1))
    # AE = sps.bmat([[A,              one_dual_vec.T],
    #                [one_dual_vec, None]]).tocsr()
    AE = sps.bmat([[A,         one_vec],
                   [one_vec.T, None]]).tocsr()
    solve_AE = spla.factorized(AE)
    def apply_pinvA(y):
        y = np.concatenate([y, np.array([0])])
        x = solve_AE(y)
        return x[:-1]

    if test_pinvA:
        pinvA = np.linalg.pinv(A.toarray())
        y = np.random.randn(N)
        x1 = apply_pinvA(y)
        x2 = np.dot(pinvA, y)
        err_pinvA = np.linalg.norm(x2-x1)/np.linalg.norm(x2)
        print('err_pinvA=', err_pinvA)

    B = sps.coo_matrix((np.ones(num_pts), (np.arange(num_pts), point_inds_ii)), shape=(num_pts, N)).tocsr()

    X = np.zeros((N, num_pts))
    for k in range(num_pts):
        X[:,k] = apply_pinvA(B[k,:].toarray())

    psi = B * one_vec
    S = np.dot(X.T, np.dot(M, X))
    S_lu, S_pivot = sla.lu_factor(S)
    solve_S = lambda b: sla.lu_solve((S_lu, S_pivot), b)





class PoissonInterpolator:
    def __init__(me, function_space_V, point_inds_ii, order_p=2, rho=1e3, screened=False):
        me.V = function_space_V
        me.ii = point_inds_ii
        me.p = order_p
        me.rho = rho
        me.screened = screened

        me.num_pts = len(point_inds_ii)
        me.N = me.V.dim()

        u_trial = fenics.TrialFunction(me.V)
        v_test = fenics.TestFunction(me.V)

        stiffness_form = fenics.inner(fenics.grad(u_trial), fenics.grad(v_test)) * fenics.dx
        me.A = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(stiffness_form))

        mass_form = u_trial * v_test * fenics.dx
        me.M = convert_fenics_csr_matrix_to_scipy_csr_matrix(fenics.assemble(mass_form))
        me.solve_M = spla.factorized(me.M)

        me.B = sps.coo_matrix((np.ones(num_pts), (np.arange(num_pts), point_inds_ii)), shape=(num_pts,me.N)).tocsr()

        diag_M = me.M.diagonal()
        me.M_lumped = sps.diags(diag_M).tocsr()
        me.iM_lumped = sps.diags(1. / diag_M).tocsr()
        me.solve_M_lumped = lambda x: x / diag_M

        if p==2:
            me.AA_BB = me.A.T * me.iM_lumped * me.A + me.rho * me.B.T * me.B
            me.solve_AA_BB = spla.factorized(me.AA_BB)
            lumped_only = True
            if lumped_only:
                me.K = sps.bmat([[me.A.T * me.iM_lumped * me.A, me.B.T],
                                 [me.B,                         None]]).tocsr()
                me.state_dim = me.N
                me.solve_P1 = me.solve_AA_BB
            else:
                me.AA_BB_extended = sps.bmat([[me.rho * me.B.T * me.B, me.A.T],
                                              [me.A,                   -me.M_lumped]])
                me.solve_P1 = lambda x: solve_block_2x2_system(x, me.solve_AA_BB, me.A.T, me.A, lambda x: -me.solve_M_lumped(x))
                me.K = sps.bmat([[None, me.A.T, me.B.T],
                                 [me.A, -me.M,  None],
                                 [me.B, None,   None]]).tocsr()
                me.state_dim = 2*me.N
        elif p==1:
            if screened:
                me.AA_BB = me.A + me.M + me.rho * me.B.T * me.B
                me.K = sps.bmat([[me.A + me.M, me.B.T],
                                 [me.B,        None]]).tocsr()
            else:
                me.AA_BB = me.A + me.rho * me.B.T * me.B
                me.K = sps.bmat([[me.A, me.B.T],
                                 [me.B, None]]).tocsr()
            me.solve_P1 = spla.factorized(me.AA_BB)

            me.state_dim = me.N
        else:
            raise RuntimeError('must have p=1 or p=2')
        me.solve_P2 = lambda x: rho * x

        me.solve_P_linop = spla.LinearOperator(me.K.shape, matvec=me.solve_P)

    def solve_P(me, x):
        x1 = x[:-me.num_pts]
        x2 = x[-me.num_pts:]
        return np.concatenate([me.solve_P1(x1), me.solve_P2(x2)])

    def interpolate_values(me, values_y, maxiter=100, tol=1e-8, return_fullspace_vector=False):
        rhs = np.concatenate([np.zeros(me.state_dim), values_y])
        # x_vec = spla.minres(me.K, rhs, M=me.solve_P_linop, maxiter=maxiter, tol=tol)[0]
        x_vec = spla.gmres(me.K, rhs, M=me.solve_P_linop, maxiter=maxiter, tol=tol)[0]
        # x_vec2 = spla.spsolve(me.K, rhs)
        u_vec = x_vec[:me.N]
        if return_fullspace_vector:
            return x_vec
        else:
            return u_vec

    def make_poisson_weighting_functions(me, normalize=True, return_extras=False):
        ww = np.zeros((me.N, me.num_pts))
        ww_positive = np.zeros((me.N, me.num_pts))
        ww_positive_connected = np.zeros((me.N, me.num_pts))
        for k in range(me.num_pts):
            y = np.zeros(num_pts)
            y[k] = 1.0
            wk = me.interpolate_values(y)
            ww[:, k] = wk

            positive_component = (wk > 0)
            ww_positive[positive_component, k] = wk[positive_component]

            connectivity_csr_matrix = me.A[positive_component, :].tocsc()[:, positive_component].tocsr()
            n_components, labels = sps.csgraph.connected_components(connectivity_csr_matrix)

            # connectivity_csr_matrix = me.A[positive_component,:].tocsc()[:, positive_component].tocsr()
            # starting_ind = me.ii[k]
            # positive_connected_component = get_connected_component(starting_ind, connectivity_csr_matrix)
            # ww_positive_connected[positive_connected_component, k] = wk[positive_connected_component]

        if normalize:
            ww = ww / np.sum(ww, axis=1).reshape((me.N,1))
            ww_positive = ww_positive / np.sum(ww_positive, axis=1).reshape((me.N, 1))
            # ww_positive_connected = ww_positive_connected / np.sum(ww_positive_connected, axis=1).reshape((me.N, 1))

        if return_extras:
            # return ww, ww_positive, ww_positive_connected
            return ww, ww_positive
        else:
            return ww

def solve_block_2x2_system(x, solve_S11, A12, A21, solve_A22):
    n1, n2 = A12.shape
    x1 = x[:n1]
    x2 = x[n1:]
    z1 = solve_S11(x1 - A12 * solve_A22(x2))
    z2 = solve_A22(x2 - A21 * z1)
    return np.concatenate([z1,z2])


# def get_connected_component(starting_ind, connectivity_csr_matrix):
#     # Depth first search
#     visited_inds = set()
#     active_inds = {starting_ind}
#     while active_inds:
#         current_ind = active_inds.pop()
#         visited_inds.add(current_ind)
#         neighbors_of_current_ind = set(connectivity_csr_matrix.getrow(current_ind).indices)
#         new_neighbors = neighbors_of_current_ind - visited_inds
#         active_inds = active_inds.update(new_neighbors)
#     return list(visited_inds)






if run_test:
    from matplotlib_mouse_click import Click

    def nearest_ind(q, xx):
        return np.argmin(np.linalg.norm(q - xx, axis=1))

    def choose_random_mesh_nodes(V, num_pts0):
        np.random.seed(0)
        xx0 = np.random.rand(num_pts0, mesh.geometric_dimension()) * 0.5 + 0.25
        # xx0 = np.array([[0.25, 0.5],
        #                 [0.75, 0.5]])
        coords = V.tabulate_dof_coordinates()
        closest_inds = []
        for k in range(xx0.shape[0]):
            closest_inds.append(nearest_ind(xx0[k, :], coords))

        point_inds_ii = np.unique(closest_inds)
        xx = coords[point_inds_ii, :]
        num_pts = xx.shape[0]
        return point_inds_ii, xx, num_pts

    #### Test case 1 (randn, p=2) ####
    n=4*5
    num_pts0 = 81
    rho = 1.0e3
    p=2
    maxiter = 5
    tol = 1e-15

    mesh = fenics.UnitSquareMesh(n,n)
    V = fenics.FunctionSpace(mesh, 'CG', 1)

    point_inds_ii, xx, num_pts = choose_random_mesh_nodes(V, num_pts0)
    y = np.random.randn(num_pts)

    poisson_squared_weighting_functions(V, xx, test_pinvA=True)

    PI = PoissonInterpolator(V,point_inds_ii, order_p=p, rho=rho)

    test_poisson_extended_solve = False
    if test_poisson_extended_solve:
        x = np.random.randn(PI.AA_BB_extended.shape[1])
        z = PI.solve_P1(x)
        z2 = spla.spsolve(PI.AA_BB_extended, x)
        err_solve_AA_BB_extended = np.linalg.norm(z2 - z) / np.linalg.norm(z2)
        print('err_solve_AA_BB_extended=', err_solve_AA_BB_extended)

    x_vec = PI.interpolate_values(y, maxiter=maxiter, tol=tol, return_fullspace_vector=True)
    rhs = np.concatenate([np.zeros(PI.state_dim), y])
    err_p2_randn = np.linalg.norm(PI.K * x_vec - rhs)/np.linalg.norm(rhs)
    print('rho=', rho, ', maxiter=',maxiter, ', err_p2_randn=', err_p2_randn)

    u_vec = PI.interpolate_values(y, maxiter=maxiter, tol=tol, return_fullspace_vector=False)
    ui_fct = vec2fct(u_vec, V)

    plt.figure()
    c = fenics.plot(ui_fct)
    for k in range(xx.shape[0]):
        plt.plot(xx[k, 0], xx[k, 1], '.r')
    plt.colorbar(c)
    plt.title('p=2, randn')

    y_interp = PI.B * u_vec
    err_y = np.linalg.norm(y_interp - y) / np.linalg.norm(y)
    print('err_y=', err_y)

    #### Test2 (p=1) ####
    p = 1

    PI = PoissonInterpolator(V, point_inds_ii, order_p=p, rho=rho)

    x_vec = PI.interpolate_values(y, maxiter=maxiter, tol=tol, return_fullspace_vector=True)
    rhs = np.concatenate([np.zeros(PI.state_dim), y])
    err_p1_randn = np.linalg.norm(PI.K * x_vec - rhs) / np.linalg.norm(rhs)
    print('rho=', rho, ', maxiter=', maxiter, ', err_p1_randn=', err_p1_randn)

    u_vec = PI.interpolate_values(y, maxiter=maxiter, tol=tol, return_fullspace_vector=False)
    ui_fct = vec2fct(u_vec, V)

    plt.figure()
    c = fenics.plot(ui_fct)
    for k in range(xx.shape[0]):
        plt.plot(xx[k, 0], xx[k, 1], '.r')
    plt.colorbar(c)
    plt.title('p=1, randn')

    y_interp = PI.B * u_vec
    err_y = np.linalg.norm(y_interp - y) / np.linalg.norm(y)
    print('err_y=', err_y)

    #### Test case 3 (p=2, zero-one)
    p=2

    y = np.zeros(num_pts)
    y[0] = 1.0

    PI = PoissonInterpolator(V, point_inds_ii, order_p=p, rho=rho)

    x_vec = PI.interpolate_values(y, maxiter=maxiter, tol=tol, return_fullspace_vector=True)
    rhs = np.concatenate([np.zeros(PI.state_dim), y])
    err_p2_zero_one = np.linalg.norm(PI.K * x_vec - rhs) / np.linalg.norm(rhs)
    print('rho=', rho, ', maxiter=', maxiter, ', err_p2_zero_one=', err_p2_zero_one)

    u_vec = PI.interpolate_values(y, maxiter=maxiter, tol=tol, return_fullspace_vector=False)
    ui_fct = vec2fct(u_vec, V)

    plt.figure()
    c = fenics.plot(ui_fct)
    for k in range(xx.shape[0]):
        plt.plot(xx[k, 0], xx[k, 1], '.r')
    plt.colorbar(c)
    plt.title('p=2, zero-one')

    y_interp = PI.B * u_vec
    err_y = np.linalg.norm(y_interp - y) / np.linalg.norm(y)
    print('err_y=', err_y)

    #### Test case 4 (p=1, zero-one)
    p=2

    PI = PoissonInterpolator(V, point_inds_ii, order_p=p, rho=rho)

    x_vec = PI.interpolate_values(y, maxiter=maxiter, tol=tol, return_fullspace_vector=True)
    rhs = np.concatenate([np.zeros(PI.state_dim), y])
    err_p1_zero_one = np.linalg.norm(PI.K * x_vec - rhs) / np.linalg.norm(rhs)
    print('rho=', rho, ', maxiter=', maxiter, ', err_p1_zero_one=', err_p1_zero_one)

    u_vec = PI.interpolate_values(y, maxiter=maxiter, tol=tol, return_fullspace_vector=False)
    ui_fct = vec2fct(u_vec, V)

    plt.figure()
    c = fenics.plot(ui_fct)
    for k in range(xx.shape[0]):
        plt.plot(xx[k, 0], xx[k, 1], '.r')
    plt.colorbar(c)
    plt.title('p=1, zero-one')

    y_interp = PI.B * u_vec
    err_y = np.linalg.norm(y_interp - y) / np.linalg.norm(y)
    print('err_y=', err_y)

    #### Poisson weighting functions

    # min_u_i 0.5 u_i^T A u_i
    # such that u_i(x_j) = delta_ij for all sample points x_j

    # L = 0.5 * u_i^T A u_i + lambda^T (B u - y)
    # y1 = [0,0,...,0,1,0,...,0]
    # y = [1,1,1,1,1,1,1,1,1,1] -> u = constant 1 function

    # 0 = [L'u_i   ] =    [A B^T][u_i     ] + [0 ]
    #     [L'lambda]      [B 0  ][lambda]     [-y]

    # K x1 = y1  x is weighting function for point 1
    # K x2 = y2  x is weighting function for point 2
    # K x3 = y3  x is weighting function for point 3
    # +      +
    # ---------
    # K(x1 + x2 + ...) = y1 + y2 + ...
    # y1 + y2 + ... = [1,1,1,1,1,1...]
    # K(x1 + x2 + ...) = [1,1,1,1,1,1...]
    # K^-1[1,1,1,1,...] = constant ones function
    # -> (u1 + u2 + ...) = constant 1 function


    # u = A^(-1) B^T (B A^(-1) B^T)^-1 y

    ww, wwp = PI.make_poisson_weighting_functions(normalize=True, return_extras=True)
    # ww, wwp, wwpc = PI.make_poisson_weighting_functions(normalize=True, return_extras=True)
    # ww = PI.make_poisson_weighting_functions(remove_negative=False)
    # ww = PI.make_poisson_weighting_functions()

    def plot_kth_weighting_function(k):
        fenics.plot(vec2fct(ww[:, k], V))
        for k in range(xx.shape[0]):
            plt.plot(xx[k, 0], xx[k, 1], '.r')

    fig = plt.figure()
    plot_kth_weighting_function(0)
    plt.title('weighting function')

    def onclick(event, ax):
        q = np.array([event.xdata, event.ydata])
        k = nearest_ind(q, xx)
        plot_kth_weighting_function(k)

    click = Click(plt.gca(), onclick, button=1)

    #

    def plot_kth_weighting_function_p(k):
        fenics.plot(vec2fct(wwp[:, k], V))
        for k in range(xx.shape[0]):
            plt.plot(xx[k, 0], xx[k, 1], '.r')

    fig = plt.figure()
    plot_kth_weighting_function_p(0)
    plt.title('weighting function, positive')

    def onclick_p(event, ax):
        q = np.array([event.xdata, event.ydata])
        k = nearest_ind(q, xx)
        plot_kth_weighting_function_p(k)

    click_p = Click(plt.gca(), onclick_p, button=1)


    #

    # def plot_kth_weighting_function_pc(k):
    #     fenics.plot(vec2fct(wwpc[:, k], V))
    #     for k in range(xx.shape[0]):
    #         plt.plot(xx[k, 0], xx[k, 1], '.r')
    #
    #
    # fig = plt.figure()
    # plot_kth_weighting_function_pc(0)
    # plt.title('weighting function, positive connected')
    #
    #
    # def onclick_pc(event, ax):
    #     q = np.array([event.xdata, event.ydata])
    #     k = nearest_ind(q, xx)
    #     plot_kth_weighting_function_pc(k)
    #
    #
    # click_pc = Click(plt.gca(), onclick_pc, button=1)

    # cid = fig.canvas.mpl_connect('button_press_event', onclick)

    # for k in range(ww.shape[1]):
    #     # k=3
    #     wk_fct = vec2fct(ww[:,k], V)
    #     plt.figure()
    #     c = fenics.plot(wk_fct)
    #     for k in range(xx.shape[0]):
    #         plt.plot(xx[k, 0], xx[k, 1], '.r')
    #     plt.colorbar(c)
    #     plt.title('p=1, normalized weighting function, k=' + str(k))

        # res = PI.A * ww[:,k]
        # res_fct = vec2fct(res, V)
        # plt.figure()
        # fenics.plot(res_fct)
        # plt.title('residual, k=' + str(k))

    # xx = V.tabulate_dof_coordinates()[np.abs(V.tabulate_dof_coordinates()[:, 1] - 0.5) < 1e-6, 0]
    # yy = ww[np.abs(V.tabulate_dof_coordinates()[:, 1] - 0.5) < 1e-6, 0]
    # plt.figure()
    # plt.plot(xx, yy)