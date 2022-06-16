import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla

from time import time


class WoodburySolver:
    def __init__(me, A0, max_r=50, rtol=1e-9, gmres_tol=1e-12, gmres_restart=20, gmres_maxiter=2,
                 run_checks=True, display=True):
        if not sps.issparse(A0):
            raise RuntimeError('WoodburySolver: Initial matrix A0 must be scipy sparse matrix.')

        me.A = A0
        me.shape = me.A.shape
        me.max_r = max_r
        me.rtol = rtol
        me.gmres_tol=gmres_tol
        me.gmres_restart = gmres_restart
        me.gmres_maxiter = gmres_maxiter
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
        # P(k+1) = P(k) + R (R^T Y)^-1 R^T =
        #        = A0^-1 + B C + new_B new_C = A0^-1 + [B, new_B] [C; new_C]
        #   where
        #       R = X - P(k) Y
        #       new_B = R
        #       new_C = (R^T Y)^-1 R^T
        t = time()

        X = np.array(me.new_xx).T
        Y = np.array(me.new_yy).T

        U_X, ss_X, VT_X = np.linalg.svd(X,0)
        inds_X = (ss_X > me.rtol * np.max(ss_X))
        X2 = U_X[:, inds_X]
        Coeff = np.dot(VT_X[inds_X, :].T, np.diag(1./ss_X[inds_X]))
        Y2 = np.dot(Y, Coeff)

        if X2.size > 0:
            new_B = X2 - me.apply_P(Y2)
            new_C = np.linalg.solve(np.dot(X2.T, Y2), X2.T)

            me.B, me.C = recompress_low_rank_matrix(np.hstack([me.B, new_B]), np.vstack([me.C, new_C]),
                                                    rtol=me.rtol, check_truncation_error=me.run_checks)

            if me.run_checks:
                X2_new = me.apply_P(Y2)
                err_low_rank_update = np.linalg.norm(X2_new - X2) / np.linalg.norm(X2)
                print('WoodburySolver: err_low_rank_update=', err_low_rank_update)

        me.new_xx.clear()
        me.new_yy.clear()
        me.new_rr.clear()

        dt_update = time() - t
        print('WoodburySolver: dt_update=', dt_update)

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

        t = time()
        x, info = spla.gmres(me.apply_A_linop, y, M=me.apply_P_linop, tol=me.gmres_tol, restart=me.gmres_restart, maxiter=me.gmres_maxiter, **kwargs)
        dt_gmres = time() - t
        me.printmaybe('WoodburySolver: dt_gmres=', dt_gmres)

        if info != 0:
            me.printmaybe('WoodburySolver: GMRES did not converge to tolerance', me.gmres_tol,
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

_run_tests = False
if _run_tests:
    N = 1000

    A_dense = np.random.randn(N, N)
    A = sps.csr_matrix(A_dense)

    WS = WoodburySolver(A, rtol=1e-10, gmres_tol=1e-12, display=True, run_checks=True)
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

    n = 15
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

    WS = WoodburySolver(A0, display=True, run_checks=True)

    b = np.random.randn(V.dim())
    x = WS.solve_A(b)
    res = np.linalg.norm(b - WS.A @ x) / np.linalg.norm(b)
    print('res=', res)

    for k in range(5):
        dgamma_vec = solve_A0(np.random.randn(A0.shape[1]))
        dgamma_vec = dgamma_vec - np.min(dgamma_vec)
        dgamma_vec = dgamma_vec / (np.max(dgamma_vec) - np.min(dgamma_vec))
        dgamma_vec = 0.1*(2.0*dgamma_vec - 1.0)
        dgamma = dl.Function(V)
        dgamma.vector()[:] = dgamma_vec

        gamma.vector()[:] = 0.0*gamma.vector()[:] + dgamma_vec
        A = csr_fenics2scipy(dl.assemble(a_form))
        WS.update_A(A)

        b = np.random.randn(V.dim())
        x = WS.solve_A(b)
        res = np.linalg.norm(b - WS.A @ x) / np.linalg.norm(b)
        print('res'+str(k)+'=', res)