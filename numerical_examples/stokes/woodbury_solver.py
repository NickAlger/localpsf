import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from time import time


class WoodburySolver:
    def __init__(me, A0, max_r=50, rtol=1e-9,
                 solve_tol=1e-12, gmres_restart=20, gmres_maxiter=2,
                 spd=False, cg_maxiter=50,
                 run_checks=True, display=True):
        if not sps.issparse(A0):
            raise RuntimeError('WoodburySolver: Initial matrix A0 must be scipy sparse matrix.')

        me.A = A0
        me.shape = me.A.shape
        me.max_r = max_r
        me.rtol = rtol
        me.solve_tol = solve_tol
        me.gmres_restart = gmres_restart
        me.gmres_maxiter = gmres_maxiter
        me.spd = spd
        me.cg_maxiter = cg_maxiter
        me.run_checks = run_checks
        me.display = display

        me.printmaybe('WoodburySolver: Factorizing initial sparse matrix')
        t = time()
        me.solve_A0 = spla.factorized(A0)
        dt_initial_factorization = time() - t
        me.printmaybe('WoodburySolver: dt_initial_factorization=', dt_initial_factorization)

        # A^-1 =approx= P^-1 := A0^-1 + BC
        me.B = np.zeros((me.shape[0], 0))
        me.C = np.zeros((0, me.shape[1]))

        me.new_xx = []
        me.new_yy = []
        me.new_rr = []

    @property
    def r(me):
        return me.B.shape[1]

    def printmaybe(me, *args, **kwargs):
        if me.display:
            print(*args, **kwargs)

    def update_A(me, new_A):
        me.A = new_A

    def refactor_A(me):
        me.printmaybe('WoodburySolver: Refactoring sparse matrix')
        t = time()

        me.solve_A0 = spla.factorized(me.A)
        me.B = np.zeros((me.shape[0], 0))
        me.C = np.zeros((0, me.shape[1]))
        me.new_xx.clear()
        me.new_yy.clear()
        me.new_rr.clear()

        dt_refactor = time() - t
        me.printmaybe('WoodburySolver: dt_refactor=', dt_refactor)

    def apply_P(me, Y):
        return me.solve_A0(Y) + np.dot(me.B, np.dot(me.C, Y))

    def update_low_rank_factors(me):
        if me.spd:
            me.update_low_rank_factors_spd()
        else:
            me.update_low_rank_factors_nonsym()

    def get_nonredundant_X_Y(me):
        X0 = np.array(me.new_xx).T
        Y0 = np.array(me.new_yy).T

        U_X, ss_X, VT_X = np.linalg.svd(X0,0)
        inds_X = (ss_X > me.rtol * np.max(ss_X))
        X = U_X[:, inds_X]
        Coeff = np.dot(VT_X[inds_X, :].T, np.diag(1./ss_X[inds_X]))
        Y = np.dot(Y0, Coeff)
        return X0, Y0, X, Y

    def update_low_rank_factors_spd(me):
        # P(k+1) = (I-X H^-1 Y^T)P(k)(I-Y H^-1 X^T) + X H^-1 X^T
        #        = (I-Z Y^T)P(k)(I-Y Z^T) + Z X^T
        #        = P(k) - Z M^T - M Z^T + Z Y^T M Z^T + Z X^T
        #        = P(k) + Z(X - M)^T + (Z Y^T M - M) Z^T
        # where
        #   H = X^T Y
        #   Z = X H^-1
        #   M = P(k) Y
        t = time()

        X0, Y0, X, Y = me.get_nonredundant_X_Y()

        err_XY = np.linalg.norm(Y - me.A @ X) / np.linalg.norm(Y)
        print('err_XY=', err_XY)

        if X.size > 0:
            H = np.dot(X.T, Y)
            err_H_nonsym = np.linalg.norm(H - H.T) / np.linalg.norm(H)
            print('err_H_nonsym=', err_H_nonsym)
            Z = np.linalg.solve(H, X.T).T
            M = me.apply_P(Y)

            new_B = np.hstack([me.B, Z, Z @ (Y.T @ M) - M])
            new_C = np.vstack([me.C, (X-M).T, Z.T])

            me.B, me.C = recompress_low_rank_matrix(new_B, new_C,
                                                    rtol=me.rtol, check_truncation_error=me.run_checks)

            if me.run_checks:
                X_new = me.apply_P(Y)
                err_low_rank_update = np.linalg.norm(X_new - X) / np.linalg.norm(X)
                print('WoodburySolver: err_low_rank_update=', err_low_rank_update)
                # X0_new = me.apply_P(Y0)
                # err_low_rank_update = np.linalg.norm(X0_new - X0) / np.linalg.norm(X0)
                # print('WoodburySolver: err_low_rank_update=', err_low_rank_update)

        me.new_xx.clear()
        me.new_yy.clear()
        me.new_rr.clear()

        dt_update_spd = time() - t
        me.printmaybe('WoodburySolver: dt_update_spd=', dt_update_spd)

    def update_low_rank_factors_nonsym(me):
        # P(k+1) = P(k) + R (R^T Y)^-1 R^T =
        #        = A0^-1 + B C + new_B new_C = A0^-1 + [B, new_B] [C; new_C]
        #   where
        #       R = X - P(k) Y
        #       new_B = R
        #       new_C = (R^T Y)^-1 R^T
        t = time()

        X0, Y0, X, Y = me.get_nonredundant_X_Y()

        if X.size > 0:
            new_B = X - me.apply_P(Y)
            new_C = np.linalg.solve(np.dot(X.T, Y), X.T)

            me.B, me.C = recompress_low_rank_matrix(np.hstack([me.B, new_B]), np.vstack([me.C, new_C]),
                                                    rtol=me.rtol, check_truncation_error=me.run_checks)

            if me.run_checks:
                X0_new = me.apply_P(Y0)
                err_low_rank_update = np.linalg.norm(X0_new - X0) / np.linalg.norm(X0)
                print('WoodburySolver: err_low_rank_update=', err_low_rank_update)

        me.new_xx.clear()
        me.new_yy.clear()
        me.new_rr.clear()

        dt_update_nonsym = time() - t
        me.printmaybe('WoodburySolver: dt_update_nonsym=', dt_update_nonsym)

    def apply_A(me, X):
        Y = me.A @ X
        R = X - me.apply_P(Y)
        if np.linalg.norm(R) > me.rtol * np.linalg.norm(X):
            if len(X.shape) == 1:
                me.new_xx.append(X)
                me.new_yy.append(Y)
                me.new_rr.append(R)
            else:
                for k in range(X.shape[1]):
                    me.new_xx.append(X[:,k])
                    me.new_yy.append(Y[:,k])
                    me.new_rr.append(R[:,k])
        return Y

    def solve_A(me, y, **kwargs):
        me.apply_A_linop = spla.LinearOperator(me.shape, matvec=me.apply_A)
        me.apply_P_linop = spla.LinearOperator(me.shape, matvec=me.apply_P)

        if me.spd:
            t = time()
            x, info = spla.cg(me.apply_A_linop, y, M=me.apply_P_linop, tol=me.solve_tol, maxiter=me.cg_maxiter, **kwargs)
            dt_cg = time() - t
            me.printmaybe('WoodburySolver: dt_cg=', dt_cg)
        else:
            t = time()
            x, info = spla.gmres(me.apply_A_linop, y, M=me.apply_P_linop, tol=me.solve_tol, restart=me.gmres_restart, maxiter=me.gmres_maxiter, **kwargs)
            dt_gmres = time() - t
            me.printmaybe('WoodburySolver: dt_gmres=', dt_gmres)

        if info != 0:
            if me.spd:
                me.printmaybe('WoodburySolver: CG did not converge to tolerance', me.solve_tol,
                              'in', me.cg_maxiter, 'iterations')
            else:
                me.printmaybe('WoodburySolver: GMRES did not converge to tolerance', me.solve_tol,
                              'in', me.gmres_maxiter, 'outer iterations, with',
                              me.gmres_restart, 'inner iterations each')
            me.refactor_A()
            x = me.solve_A0(y)
        else:
            if me.r + len(me.new_xx) > me.max_r:
                print('WoodburySolver: new rank of ' + str(me.r + len(me.new_xx)) + ' is greater than max_r=' + str(me.max_r))
                me.refactor_A()
            elif len(me.new_xx) > 0:
                me.update_low_rank_factors()

        return x


def recompress_low_rank_matrix(B, C, rtol=1e-9, check_truncation_error=True):
    QB, RB = np.linalg.qr(B, mode='reduced')
    QCT, RCT = np.linalg.qr(C.T, mode='reduced')

    Z = np.dot(RB, RCT.T)
    U, ss, VT = np.linalg.svd(Z)
    inds = (ss > rtol*np.max(ss))
    ZL = U[:,inds]
    ZR = np.dot(np.diag(ss[inds]), VT[inds, :])

    B_new = np.dot(QB, ZL)
    C_new = np.dot(ZR, QCT.T)

    if check_truncation_error:
        old_rank = B.shape[1]
        new_rank = np.sum(inds)
        Omega = np.random.randn(C.shape[1], 25)
        Z = np.dot(B, np.dot(C, Omega))
        Z_new = np.dot(B_new, np.dot(C_new, Omega))
        err_truncation = np.linalg.norm(Z - Z_new) / np.linalg.norm(Z)
        print('recompress_low_rank_matrix: old_rank=', old_rank, ', new_rank=', new_rank, ', rtol=', rtol, ', err_truncation=', err_truncation)

    return B_new, C_new

_run_tests = True
if _run_tests:
    N = 1000

    A_dense = np.random.randn(N, N)
    A = sps.csr_matrix(A_dense)

    WS = WoodburySolver(A, rtol=1e-10, solve_tol=1e-12, display=True, run_checks=True)
    b = np.random.randn(N)
    x = WS.solve_A(b)
    res = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
    print('res=', res)

    for k in range(10):
        A = sps.csr_matrix(A + 0.1 * np.random.randn(*A.shape) / np.linalg.norm(A_dense))
        WS.update_A(A)
        # b = np.random.randn(N)
        x = WS.solve_A(b)
        res = np.linalg.norm(b - A @ x) / np.linalg.norm(b)
        print('res'+str(k)+'=', res)


    import dolfin as dl
    import matplotlib.pyplot as plt
    from nalger_helper_functions import csr_fenics2scipy

    print('\n\n\n\n')

    n = 10
    mesh = dl.UnitCubeMesh(n,n,n)
    # n = 50
    # mesh = dl.UnitSquareMesh(n,n)
    V = dl.FunctionSpace(mesh, 'CG', 2)
    u_trial = dl.TrialFunction(V)
    v_test = dl.TestFunction(V)
    gamma = dl.Function(V)
    a_form = dl.inner(dl.exp(gamma)*dl.grad(v_test), dl.grad(u_trial))*dl.dx + 0.1 * v_test * u_trial * dl.dx

    A0 = csr_fenics2scipy(dl.assemble(a_form))
    solve_A0 = spla.factorized(A0)

    WS = WoodburySolver(A0, display=True, run_checks=True, spd=True, solve_tol=1e-13)

    b = np.random.randn(V.dim())
    x = WS.solve_A(b)
    res = np.linalg.norm(b - WS.A @ x) / np.linalg.norm(b)
    print('res=', res)

    for k in range(5):
        dgamma_vec = solve_A0(np.random.randn(A0.shape[1]))
        dgamma_vec = dgamma_vec - np.min(dgamma_vec)
        dgamma_vec = dgamma_vec / (np.max(dgamma_vec) - np.min(dgamma_vec))
        dgamma_vec = 0.5*(2.0*dgamma_vec - 1.0)
        dgamma = dl.Function(V)
        dgamma.vector()[:] = dgamma_vec

        gamma.vector()[:] = gamma.vector()[:] + dgamma_vec
        A = csr_fenics2scipy(dl.assemble(a_form))
        WS.update_A(A)

        b = np.random.randn(V.dim())
        x = WS.solve_A(b)
        res = np.linalg.norm(b - WS.A @ x) / np.linalg.norm(b)
        print('res'+str(k)+'=', res)
