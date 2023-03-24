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


def deflate_negative_eigenvalues(apply_A: vec2vec,
                                 apply_B: vec2vec,
                                 solve_B: vec2vec,
                                 N: int, # A.shape = B.shape = (N,N)
                                 make_OP_preconditioner: typ.Callable[[float], vec2vec], # Approximates sigma -> (v -> (A - sigma*B)^-1 @ v)
                                 threshold = -0.5,
                                 gamma: float=-1.0, # -1.0: set negative eigs to zero. -2.0: flip negative eigs
                                 sigma_reduction_factor: float=100.0,
                                 chunk_size=10,
                                 tol: float=1e-6,
                                 ncv_factor=None,
                                 lanczos_maxiter=2,
                                 lanczos_mode='cayley',
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
    assert(sigma_reduction_factor > 0)
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

    eig_bounds = (-np.abs(max_eig), threshold)
    printmaybe('eig_bounds=', eig_bounds)
    if eig_bounds[0] > eig_bounds[1]:
        return dd0, V0

    # num_shifts = int(np.ceil(np.log(eig_bounds[0] / eig_bounds[1]) / np.log(sigma_range)))
    # printmaybe('num_shifts=', num_shifts)
    # sigmas = -np.logspace(np.log10(-eig_bounds[1]), np.log10(-eig_bounds[0]), num_shifts * 2, endpoint=False)[1::2]
    # printmaybe('sigmas=', sigmas)

    safe_factor = 2.0

    sigma = eig_bounds[0] / safe_factor

    DSO = None
    # for sigma in sigmas:
    while sigma < eig_bounds[1]:
        printmaybe('making A-sigma*B preconditioner. sigma=', sigma)
        solve_P = make_OP_preconditioner(sigma)
        iP_op = CountedOperator((N,N), solve_P, display=False, name='invP')
        if DSO is None:
            DSO = DeflatedShiftedOperator(apply_A, apply_B, sigma, solve_P, gamma, V0, dd0)
        else:
            DSO = DSO.update_sigma(sigma, iP_op.matvec)

        printmaybe('Getting eigs near sigma')
        dd, U = DSO.get_eigs_near_sigma(target_num_eigs=chunk_size, ncv_factor=ncv_factor,
                                        mode=lanczos_mode, maxiter=lanczos_maxiter, tol=tol)
        print('iP_op.count=', iP_op.count)
        while len(dd) > 0:
            printmaybe('dd=', dd)
            printmaybe('Updating deflation')
            # good_inds = dd < threshold
            good_inds = dd < 0.0
            if np.any(good_inds):
                DSO = DSO.update_deflation(B_op.matmat(U[:,good_inds]), dd[good_inds])

            # if len(dd) == chunk_size and np.all(good_inds):
            if np.all(dd < threshold) and len(dd) == chunk_size:
                printmaybe('Getting eigs near sigma')
                dd, U = DSO.get_eigs_near_sigma(target_num_eigs=chunk_size, ncv_factor=ncv_factor,
                                                mode=lanczos_mode, maxiter=lanczos_maxiter, tol=tol)
            else:
                if len(DSO.dd) > 0:
                    e_bound = np.max(DSO.dd)
                    if sigma < e_bound:
                        sigma = e_bound / safe_factor
                    else:
                        sigma = sigma / sigma_reduction_factor
                else:
                    sigma = sigma / sigma_reduction_factor

                break

    if DSO is not None:
        V = DSO.BU
        dd = gamma * DSO.dd
    else:
        V = V0
        dd = dd0

    return dd, V


N = 1000
A = np.random.randn(N,N)
A = (A + A.T) / 2.0
ee, P = np.linalg.eigh(A)
# A = P @ np.diag(ee**3) @ P.T
apply_A = lambda x: A @ x

B = np.random.randn(N,N)
B = sla.sqrtm(B @ B.T)
apply_B = lambda x: B @ x

invB = np.linalg.inv(B)
solve_B = lambda x: invB @ x

ee_true, U_true = sla.eigh(A, B)

print('ee_true=', ee_true)

noise = 0.00*np.random.randn(N,N)
noise = (noise + noise.T) / 2.0

def make_shifted_solver(shift):
    inv_P = np.linalg.inv(A + noise - shift*B)
    return lambda x: inv_P @ x

sigma = 1e0
solve_P = make_shifted_solver(sigma)

gamma = -1.0

dd, V = deflate_negative_eigenvalues(apply_A, apply_B, solve_B, N,
                                     make_shifted_solver,
                                     sigma_reduction_factor=100.0,
                                     chunk_size=10,
                                     ncv_factor=3,
                                     lanczos_maxiter=2,
                                     tol=1e-6,
                                     display=True,
                                     )

A_deflated = A + V @ np.diag(dd) @ V.T
ee_deflated, U_deflated = sla.eigh(A_deflated, B)

Ray = U_true.T @ A @ U_true
Ray_deflated = U_true.T @ A_deflated @ U_true

nondiagonal_Ray = np.linalg.norm(Ray_deflated - np.diag(Ray_deflated.diagonal())) / np.linalg.norm(Ray_deflated)
print('nondiagonal_Ray=', nondiagonal_Ray)

rr = Ray.diagonal()
rr_deflated = Ray_deflated.diagonal()

plt.figure()
plt.plot(rr - rr_deflated)


####

DSO = DeflatedShiftedOperator(apply_A, apply_B, sigma, solve_P, gamma, np.zeros((N,0)), np.zeros((0,)))

dd, U = DSO.get_eigs_near_sigma(target_num_eigs=5)

# dd, U = spla.eigsh(spla.LinearOperator((N,N), matvec=apply_A), k=5,
#                    M=spla.LinearOperator((N,N), matvec=apply_B),
#                    Minv=spla.LinearOperator((N,N), matvec=solve_B), which='LM')

DSO2 = DSO.update_deflation(B @ U, dd)

dd2, U2 = DSO2.get_eigs_near_sigma(target_num_eigs=5)

DSO3 = DSO2.update_deflation(B @ U2, dd2)

sigma = -0.1
solve_P = make_shifted_solver(sigma)
DSO4 = DSO3.update_sigma(sigma, solve_P)

dd4, U4 = DSO4.get_eigs_near_sigma(target_num_eigs=50)

sigma = 1000.0
solve_P = make_shifted_solver(sigma)
DSO4 = DSO3.update_sigma(sigma, solve_P)

dd4, U4 = DSO4.get_eigs_near_sigma(target_num_eigs=5)
print('sigma=', sigma, 'dd4=', dd4)

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