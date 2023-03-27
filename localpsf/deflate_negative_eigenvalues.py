import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import typing as typ
from dataclasses import dataclass
from functools import cached_property


vec2vec = typ.Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class DeflatedShiftedOperator:
    '''A - sigma*B + gamma * B @ U @ diag(dd) @ U.T @ B
    U.T @ A @ U = diag(dd)
    U.T @ B @ U = I
    solve_P(x) =approx= (A - sigma*B)^-1 @ x
    A and B are symmetric'''
    apply_A: vec2vec
    apply_B: vec2vec
    sigma: float
    solve_P: vec2vec # Approximates v -> (A - sigma*B)^-1 @ v
    gamma: float
    BU: np.ndarray # B @ U
    dd: np.ndarray

    @cached_property
    def N(me):
        return me.BU.shape[0]

    @cached_property
    def k(me):
        return len(me.dd)

    def __post_init__(me):
        assert(len(me.BU.shape) == 2)
        assert(len(me.dd.shape) == 1)
        assert(me.BU.shape == (me.N, me.k))
        assert(me.dd.shape == (me.k,))

        b = np.random.randn(me.N)
        x = me.solve_shifted(b)
        norm_r = np.linalg.norm(b - me.apply_shifted(x))
        norm_b = np.linalg.norm(b)
        print('shifted: norm_r/norm_b=', norm_r / norm_b)
        # assert(norm_r < 1e-6 * norm_b)

        b = np.random.randn(me.N)
        x = me.solve_shifted_deflated(b)
        norm_r = np.linalg.norm(b - me.apply_shifted_deflated(x))
        norm_b = np.linalg.norm(b)
        print('shifted and deflated: norm_r/norm_b=', norm_r / norm_b)
        # assert(norm_r < 1e-6 * norm_b)

    def apply_shifted(me, x: np.ndarray) -> np.ndarray: # x -> (A - sigma*B) @ x
        return me.apply_A(x) - me.sigma * me.apply_B(x)

    def solve_shifted(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B)^-1 @ b
        return me.solve_P(b)
        # return spla.gmres(spla.LinearOperator((me.N, me.N), matvec=me.apply_shifted), b,
        #                    M=spla.LinearOperator((me.N, me.N), matvec=me.solve_P),
        #                    tol=1e-10)[0]

    def apply_deflated(me, x: np.ndarray): # x -> (A + gamma*B @ U @ diag(dd) @ U.T @ B) @ x
        return me.apply_A(x) + gamma * me.BU @ (me.dd * (me.BU.T @ x))

    # A - sigma * B + gamma * B @ U @ diag(dd) @ B @ U.T
    @cached_property
    def diag_Phi(me): # Phi = ((gamma*diag(dd)))^-1 - B @ U.T @ (A - sigma*B)^-1 @ B @ U)^-1
        return (me.gamma*me.dd) * (me.dd - me.sigma) / ((me.dd - me.sigma) + (me.gamma*me.dd))

    def apply_shifted_deflated(me, x: np.ndarray) -> np.ndarray: # x -> (A - sigma*B + gamma*B @ U @ diag(dd) @ U.T @ B) @ x
        return me.apply_shifted(x) + me.gamma * me.BU @ (me.dd * (me.BU.T @ x))

    def solve_shifted_deflated(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B + gamma*B @ U @ diag(dd) @ U.T @ B)^-1 @ b
        return me.solve_shifted(b - me.BU @ (me.diag_Phi * (me.BU.T @ me.solve_shifted(b))))

    def get_eigs_near_sigma(me, target_num_eigs=10, tol=1e-7, ncv_factor=None, maxiter=3, mode='cayley') -> typ.Tuple[np.ndarray, np.ndarray]: # (dd, U)
        if ncv_factor is None:
            ncv = None
        else:
            ncv = ncv_factor*target_num_eigs
        try:
            dd, U = spla.eigsh(spla.LinearOperator((N,N), matvec=me.apply_deflated), target_num_eigs,
                                  sigma=me.sigma,
                                  mode=mode,
                                  M=spla.LinearOperator((N,N), matvec=me.apply_B),
                                  OPinv=spla.LinearOperator((N,N), matvec=me.solve_shifted_deflated),
                                  which='LM', return_eigenvectors=True,
                                  tol=tol,
                                  ncv=ncv,
                                  maxiter=maxiter)
        except spla.ArpackNoConvergence as ex:
            U = ex.eigenvectors
            dd = ex.eigenvalues
            print(ex)

        return dd, U

    def update_sigma(me, new_sigma: float, new_solve_P: vec2vec) -> 'DeflatedShiftedOperator':
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, new_sigma, new_solve_P, me.gamma, me.BU, me.dd)

    def update_gamma(me, new_gamma: float) -> 'DeflatedShiftedOperator':
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_P, new_gamma, me.BU, me.dd)

    def update_deflation(me, BU2: np.ndarray, dd2: np.ndarray) -> 'DeflatedShiftedOperator':
        new_BU = np.hstack([me.BU, BU2])
        new_dd = np.concatenate([me.dd, dd2])
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_P, me.gamma, new_BU, new_dd)


class CountedOperator:
    def __init__(me,
                 shape: typ.Tuple[int, int],
                 matvec: typ.Callable[[np.ndarray], np.ndarray],
                 display: bool = False,
                 name: str = '',
                 dtype=float
                 ):
        me.shape = shape
        me._matvec = matvec
        me.display = display
        me.count = 0
        me.name = name

    def matvec(me, x: np.ndarray) -> np.ndarray:
        assert(x.shape == (me.shape[1],))
        if me.display:
            print(me.name + ' count:', me.count)
        me.count += 1
        return me._matvec(x)

    def matmat(me, X: np.ndarray) -> np.ndarray:
        assert(len(X.shape) == 2)
        k = X.shape[1]
        assert(X.shape == (me.shape[1], k))
        Y = np.zeros((me.shape[0], k))
        for ii in range(k):
            Y[:,ii] = me.matvec(X[:,ii])
        return Y

    def __call__(me, x: np.ndarray) -> np.ndarray:
        if len(x.shape) == 1:
            return me.matvec(x)
        else:
            return me.matmat(x)

    def as_linear_operator(me):
        return spla.aslinearoperator(me)


def deflate_negative_eigs_near_sigma(DSO: DeflatedShiftedOperator,
                                     B_op: CountedOperator,
                                     threshold: float,
                                     chunk_size: int,
                                     ncv_factor: float,
                                     lanczos_maxiter: int,
                                     tol: float,
                                     display: True,
                                     ) -> typ.Tuple[DeflatedShiftedOperator, float, float]: # DSO, band lower bound, band upper bound
    assert(threshold < 0.0)
    sigma = DSO.sigma
    dd = np.zeros(0)
    for _ in range(10):
        dd_new, U_new = DSO.get_eigs_near_sigma(target_num_eigs=chunk_size, ncv_factor=ncv_factor,
                                        mode='cayley', maxiter=lanczos_maxiter, tol=tol)
        dd = np.concatenate([dd, dd_new])
        print('dd_new=', dd_new)
        if display:
            print('Updating deflation')
        if np.any(dd_new < 0.0):
            DSO = DSO.update_deflation(B_op.matmat(U_new[:, dd_new < 0.0]), dd_new[dd_new < 0.0])

        print('len(dd_new)=', len(dd_new))
        if len(dd_new) == 0 or np.any(dd >= threshold):
            break

    print('sigma=', sigma, ', dd=', dd)
    dd_plus = dd[dd < 0]

    if len(dd_plus) > 0:
        ee = (dd_plus + sigma) / (dd_plus - sigma)
        extremal_ind = np.argmin(np.abs(ee))
        e = ee[extremal_ind]
        if e <= 0:
            e2 = e
            e1 = -e2
        else:
            e1=e
            e2=-e1

        d_lower = sigma * (e1 + 1.0) / (e1 - 1.0)
        d_upper = sigma * (e2 + 1.0) / (e2 - 1.0)
    else:
        d_lower = None
        d_upper = None

    return DSO, d_lower, d_upper


def deflate_negative_eigenvalues(apply_A: vec2vec,
                                 apply_B: vec2vec,
                                 solve_B: vec2vec,
                                 N: int, # A.shape = B.shape = (N,N)
                                 make_OP_preconditioner: typ.Callable[[float], vec2vec], # Approximates sigma -> (v -> (A - sigma*B)^-1 @ v)
                                 threshold = -0.5,
                                 gamma: float=-1.0, # -1.0: set negative eigs to zero. -2.0: flip negative eigs
                                 sigma_factor: float=8.0,
                                 chunk_size=20,
                                 tol: float=1e-6,
                                 ncv_factor=None,
                                 lanczos_maxiter=2,
                                 display=True,
                                ) -> typ.Tuple[np.ndarray, np.ndarray]: # (dd, V)
    '''Form low rank update A -> A + V @ diag(dd) @ V.T such that
        eigs(A + V @ diag(dd) @ V.T, B) > threshold
    A must be symmetric
    B must be symmetric positive definite
    OP = A - sigma*B
    OP_preconditioner = make_OP_preconditioner(sigma)
    OP_preconditioner(b) =approx= OP^-1 @ b
    '''
    assert(N > 0)
    assert(threshold < 0)
    assert(tol > 0)
    assert(sigma_factor > 0)
    A_op = CountedOperator((N,N), apply_A, display=False, name='A')
    B_op = CountedOperator((N, N), apply_B, display=False, name='B')
    iB_op = CountedOperator((N, N), solve_B, display=False, name='invB')

    def printmaybe(*args, **kwargs):
        if display:
            print(*args, **kwargs)

    V0 = np.zeros((N, 0))
    dd0 = np.zeros((0,))

    # Get largest magnitude eigenvalue of matrix pencil (A,B)
    printmaybe('Getting largest magnitude eigenvalue')
    max_eig = spla.eigsh(spla.LinearOperator((N,N), matvec=A_op.matvec),
                         1,
                         M=spla.LinearOperator((N,N), matvec=B_op.matvec),
                         Minv=spla.LinearOperator((N,N), matvec=iB_op.matvec),
                         which='LM', return_eigenvectors=False,
                         tol=tol)[0]
    printmaybe('max_eig=', max_eig)

    if -1.0 < -np.abs(max_eig):
        return dd0, V0

    sigma = -1.0
    printmaybe('making A-sigma*B preconditioner. sigma=', sigma)
    solve_P = make_OP_preconditioner(sigma)
    DSO = DeflatedShiftedOperator(apply_A, apply_B, sigma, solve_P, gamma, V0, dd0)

    printmaybe('Getting eigs near sigma')
    DSO, d_lower, d_upper = deflate_negative_eigs_near_sigma(DSO, B_op, threshold, chunk_size,
                                                             ncv_factor, lanczos_maxiter, tol, display)
    print('d_lower=', d_lower, ', sigma=', sigma, ', d_upper=', d_upper)
    if d_lower is None:
        band_lower = sigma * sigma_factor
        band_upper = sigma / sigma_factor
    else:
        band_lower = d_lower
        band_upper = d_upper
    print('band_lower=', band_lower, ', sigma=', sigma, ', band_upper=', band_upper)

    while -np.abs(max_eig) < band_lower:
        proposed_sigma = band_lower * sigma_factor
        print('proposed_sigma=', proposed_sigma)
        sigma = np.max([-np.abs(max_eig) * 1.05, proposed_sigma])
        print('sigma=', sigma)

        printmaybe('making A-sigma*B preconditioner. sigma=', sigma)
        solve_P = make_OP_preconditioner(sigma)
        iP_op = CountedOperator((N,N), solve_P, display=False, name='invP')
        DSO = DSO.update_sigma(sigma, iP_op.matvec)
        DSO, d_lower, d_upper = deflate_negative_eigs_near_sigma(DSO, B_op, band_lower, chunk_size,
                                                                 ncv_factor, lanczos_maxiter, tol, display)
        print('d_lower=', d_lower, ', sigma=', sigma, ', d_upper=', d_upper)
        if d_lower is None:
            d_lower = sigma * sigma_factor
            d_upper = sigma / sigma_factor

        print('d_lower=', d_lower, ', sigma=', sigma, ', d_upper=', d_upper)
        band_lower = np.min([d_lower, sigma * sigma_factor])
        band_upper = np.max([d_upper, sigma / sigma_factor])
        print('band_lower=', band_lower, ', sigma=', sigma, ', band_upper=', band_upper)

    V = DSO.BU
    dd = gamma * DSO.dd

    return dd, V


N = 1000
A_diag = np.sort(np.random.randn(N))
apply_A = lambda x: A_diag * x

B_diag = np.random.randn(N)
B_diag = np.sqrt(B_diag * B_diag)
apply_B = lambda x: B_diag * x
solve_B = lambda x: x / B_diag

noise_diag = 0.00*np.random.randn(N)

def make_shifted_solver(shift):
    OP_diag = A_diag - shift * B_diag + noise_diag
    return lambda x: x / OP_diag

sigma = 1e0
solve_P = make_shifted_solver(sigma)

gamma = -1.0

dd, V = deflate_negative_eigenvalues(apply_A, apply_B, solve_B, N,
                                     make_shifted_solver,
                                     sigma_factor=10.0,
                                     chunk_size=50,
                                     ncv_factor=2,
                                     lanczos_maxiter=2,
                                     tol=1e-7,
                                     display=True,
                                     )
A = np.diag(A_diag)
B = np.diag(B_diag)
ee_true, U_true = sla.eigh(A, B)
# print('ee_true=', ee_true)
plt.figure()
plt.plot(ee_true)
plt.title('ee_true')

A_deflated = A + V @ np.diag(dd) @ V.T

Ray = U_true.T @ A @ U_true
Ray_deflated = U_true.T @ A_deflated @ U_true

nondiagonal_Ray = np.linalg.norm(Ray_deflated - np.diag(Ray_deflated.diagonal())) / np.linalg.norm(Ray_deflated)
print('nondiagonal_Ray=', nondiagonal_Ray)

rr = Ray.diagonal()
rr_deflated = Ray_deflated.diagonal()

plt.figure()
plt.plot(rr - rr_deflated)


####
#
# DSO = DeflatedShiftedOperator(apply_A, apply_B, sigma, solve_P, gamma, np.zeros((N,0)), np.zeros((0,)))
#
# dd, U = DSO.get_eigs_near_sigma(target_num_eigs=5)
#
# # dd, U = spla.eigsh(spla.LinearOperator((N,N), matvec=apply_A), k=5,
# #                    M=spla.LinearOperator((N,N), matvec=apply_B),
# #                    Minv=spla.LinearOperator((N,N), matvec=solve_B), which='LM')
#
# DSO2 = DSO.update_deflation(B @ U, dd)
#
# dd2, U2 = DSO2.get_eigs_near_sigma(target_num_eigs=5)
#
# DSO3 = DSO2.update_deflation(B @ U2, dd2)
#
# sigma = -0.1
# solve_P = make_shifted_solver(sigma)
# DSO4 = DSO3.update_sigma(sigma, solve_P)
#
# dd4, U4 = DSO4.get_eigs_near_sigma(target_num_eigs=50)
#
# sigma = 1000.0
# solve_P = make_shifted_solver(sigma)
# DSO4 = DSO3.update_sigma(sigma, solve_P)
#
# dd4, U4 = DSO4.get_eigs_near_sigma(target_num_eigs=5)
# print('sigma=', sigma, 'dd4=', dd4)

# dd2, U2 = DSO2.get_eigs_near_mu()


# ee_Hd = np.loadtxt('Hd_eigs.txt')
# ee_R = np.loadtxt('R_eigs.txt')
# ee_pre = ee_Hd / ee_R
#
# plt.figure()
# plt.plot(ee_Hd)
# plt.plot(ee_R)
# plt.legend(['Hd', 'R'])
# plt.title('Eigenvalues')
#
# plt.figure()
# plt.plot(ee_pre)
# plt.legend(['inv(R) @ Hd'])
# plt.title('Preconditioned eigenvalues')