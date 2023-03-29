import numpy as np
import scipy.sparse.linalg as spla
import typing as typ
from dataclasses import dataclass
from functools import cached_property


_GMRES_TOL = 1e-10

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
    solve_shifted_preconditioner: vec2vec # Approximates v -> (A - sigma*B)^-1 @ v
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
        # print('shifted: norm_r/norm_b=', norm_r / norm_b)
        # assert(norm_r < 1e-6 * norm_b)

        b = np.random.randn(me.N)
        x = me.solve_shifted_deflated(b)
        norm_r = np.linalg.norm(b - me.apply_shifted_deflated(x))
        norm_b = np.linalg.norm(b)
        # print('shifted and deflated: norm_r/norm_b=', norm_r / norm_b)
        # assert(norm_r < 1e-6 * norm_b)

    def apply_shifted(me, x: np.ndarray) -> np.ndarray: # x -> (A - sigma*B) @ x
        return me.apply_A(x) - me.sigma * me.apply_B(x)

    def solve_shifted(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B)^-1 @ b
        # return me.solve_P(b)
        return spla.gmres(spla.LinearOperator((me.N, me.N), matvec=me.apply_shifted), b,
                          M=spla.LinearOperator((me.N, me.N), matvec=me.solve_shifted_preconditioner),
                          tol=_GMRES_TOL)[0]

    def apply_deflated(me, x: np.ndarray): # x -> (A + gamma*B @ U @ diag(dd) @ U.T @ B) @ x
        return me.apply_A(x) + me.gamma * me.BU @ (me.dd * (me.BU.T @ x))

    # A - sigma * B + gamma * B @ U @ diag(dd) @ B @ U.T
    @cached_property
    def diag_Phi(me): # Phi = ((gamma*diag(dd)))^-1 - B @ U.T @ (A - sigma*B)^-1 @ B @ U)^-1
        return (me.gamma*me.dd) * (me.dd - me.sigma) / ((me.dd - me.sigma) + (me.gamma*me.dd))

    def apply_shifted_deflated(me, x: np.ndarray) -> np.ndarray: # x -> (A - sigma*B + gamma*B @ U @ diag(dd) @ U.T @ B) @ x
        return me.apply_shifted(x) + me.gamma * me.BU @ (me.dd * (me.BU.T @ x))

    def solve_shifted_deflated(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B + gamma*B @ U @ diag(dd) @ U.T @ B)^-1 @ b
        return me.solve_shifted(b - me.BU @ (me.diag_Phi * (me.BU.T @ me.solve_shifted(b))))

    def solve_shifted_preconditioner_deflated(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B + gamma*B @ U @ diag(dd) @ U.T @ B)^-1 @ b
        return me.solve_shifted(b - me.BU @ (me.diag_Phi * (me.BU.T @ me.solve_shifted_preconditioner(b))))

    def get_eigs_near_sigma(me, target_num_eigs=10, tol=1e-7, ncv_factor=None, maxiter=3, mode='cayley',
                            preconditioner_only=False) -> typ.Tuple[np.ndarray, np.ndarray]: # (dd, U)
        if ncv_factor is None:
            ncv = None
        else:
            ncv = ncv_factor*target_num_eigs

        if preconditioner_only:
            OPinv_matvec = me.solve_shifted_preconditioner_deflated
        else:
            OPinv_matvec = me.solve_shifted_deflated

        try:
            dd, U = spla.eigsh(spla.LinearOperator((me.N, me.N), matvec=me.apply_deflated), target_num_eigs,
                                  sigma=me.sigma,
                                  mode=mode,
                                  M=spla.LinearOperator((me.N, me.N), matvec=me.apply_B),
                                  OPinv=spla.LinearOperator((me.N, me.N), matvec=OPinv_matvec),
                                  which='LM', return_eigenvectors=True,
                                  tol=tol,
                                  ncv=ncv,
                                  maxiter=maxiter)
        except spla.ArpackNoConvergence as ex:
            U = ex.eigenvectors
            dd = ex.eigenvalues
            # print(ex)

        print(len(dd), ' / ', target_num_eigs, ' eigs found')

        return dd, U

    def update_sigma(me, new_sigma: float, new_solve_P: vec2vec) -> 'DeflatedShiftedOperator':
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, new_sigma, new_solve_P, me.gamma, me.BU, me.dd)

    def update_gamma(me, new_gamma: float) -> 'DeflatedShiftedOperator':
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_shifted_preconditioner, new_gamma, me.BU, me.dd)

    def update_deflation(me, BU2: np.ndarray, dd2: np.ndarray) -> 'DeflatedShiftedOperator':
        new_BU = np.hstack([me.BU, BU2])
        new_dd = np.concatenate([me.dd, dd2])
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_shifted_preconditioner, me.gamma, new_BU, new_dd)


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
                                     preconditioner_only: bool=False,
                                     display: bool=True,
                                     ) -> typ.Tuple[DeflatedShiftedOperator, float, float]: # DSO, band lower bound, band upper bound
    assert(threshold < 0.0)
    sigma = DSO.sigma
    dd = np.zeros(0)
    for _ in range(10):
        dd_new, U_new = DSO.get_eigs_near_sigma(target_num_eigs=chunk_size, ncv_factor=ncv_factor,
                                                mode='cayley', maxiter=lanczos_maxiter, tol=tol,
                                                preconditioner_only=preconditioner_only)
        dd = np.concatenate([dd, dd_new])
        if display:
            print('Updating deflation')
        if np.any(dd_new < 0.0):
            DSO = DSO.update_deflation(B_op.matmat(U_new[:, dd_new < 0.0]), dd_new[dd_new < 0.0])

        if len(dd_new) == 0 or np.any(dd >= threshold):
            break

    dd_minus = dd[dd < 0]

    if len(dd_minus) > 0:
        d_lower = np.min(dd_minus)
        d_upper = np.max(dd_minus)
    else:
        d_lower = None
        d_upper = None

    return DSO, d_lower, d_upper


def check_symmetry(apply_A: vec2vec, N: int, rtol: float=1e-10) -> None:
    x = np.random.randn(N)
    Ax = apply_A(x)
    assert(len(Ax) == N)
    y = np.random.randn(N)
    Ay = apply_A(y)
    t1 = np.dot(y, Ax)
    t2 = np.dot(x, Ay)
    assert(np.abs(t1 - t2) <= rtol * (np.abs(t1) + np.abs(t2)))


def deflate_negative_eigenvalues(apply_A: vec2vec,
                                 apply_B: vec2vec,
                                 solve_B: vec2vec,
                                 N: int, # A.shape = B.shape = (N,N)
                                 make_OP_preconditioner: typ.Callable[[float], vec2vec], # Approximates sigma -> (v -> (A - sigma*B)^-1 @ v)
                                 threshold = -0.5,
                                 gamma: float=-1.0, # -1.0: set negative eigs to zero. -2.0: flip negative eigs
                                 sigma_factor: float=10.0, # Sigma scaled up by this much above previous bound
                                 chunk_size=50,
                                 tol: float=1e-8,
                                 ncv_factor=3,
                                 lanczos_maxiter=2,
                                 preconditioner_only=False,
                                 display=True,
                                ) -> typ.Tuple[np.ndarray, np.ndarray]: # (dd, V)
    '''Form low rank update A -> A + V @ diag(dd) @ V.T such that
        eigs(A + V @ diag(dd) @ V.T, B) > threshold
    A must be symmetric
    B must be symmetric positive definite
    OP = A - sigma*B
    OP_preconditioner = make_OP_preconditioner(sigma)
    OP_preconditioner(b) =approx= OP^-1 @ b

    In:
        import numpy as np
        import scipy.linalg as sla

        N = 1000
        A_diag = np.sort(np.random.randn(N))
        apply_A = lambda x: A_diag * x

        B_diag = np.random.randn(N)
        B_diag = np.sqrt(B_diag * B_diag)
        apply_B = lambda x: B_diag * x
        solve_B = lambda x: x / B_diag

        noise_diag = 0.01*np.random.randn(N)

        def make_shifted_solver(shift):
            OP_diag = A_diag - shift * B_diag + noise_diag
            return lambda x: x / OP_diag

        threshold = -0.5
        dd, V = deflate_negative_eigenvalues(apply_A, apply_B, solve_B, N,
                                             make_shifted_solver,
                                             threshold=threshold,
                                             chunk_size=50,
                                             display=True,
                                             )

        A = np.diag(A_diag)
        B = np.diag(B_diag)
        ee_true, U_true = sla.eigh(A, B)

        A_deflated = A + V @ np.diag(dd) @ V.T
        Rayleigh = U_true.T @ A_deflated @ U_true

        nondiagonal_Rayleigh_error = np.linalg.norm(Rayleigh - np.diag(Rayleigh.diagonal())) / np.linalg.norm(Rayleigh)
        print('nondiagonal_Rayleigh_error=', nondiagonal_Rayleigh_error)

        ee = Rayleigh.diagonal()
        positive_inds = (ee_true >= 0.0)
        positive_error = np.linalg.norm(ee_true[positive_inds] - ee[positive_inds]) / np.linalg.norm(ee_true[positive_inds])
        print('positive_error=', positive_error)

        zeroing_inds = (ee_true < threshold)
        zeroing_error = np.linalg.norm(ee[zeroing_inds]) / np.linalg.norm(ee_true)
        print('zeroing_error=', zeroing_error)

        intermediate_inds = np.logical_and(threshold <= ee_true, ee_true < 0.0)
        ee_int = ee[intermediate_inds]
        ee_true_int = ee_true[intermediate_inds]
        delta1 = np.abs(ee_true_int - ee_int)
        delta2 = np.abs(ee_int)
        intermediate_error = np.linalg.norm(np.min([delta1, delta2], axis=0)) / np.linalg.norm(ee_true)
        print('intermediate_error=', intermediate_error)

    Out:
        LM_eig= -422.4888729005527
        making A-sigma*B preconditioner
        Getting eigs near sigma
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        band_lower= -2.144669296642468
        proposed_sigma= -21.446692966424678 , sigma= -21.446692966424678
        making A-sigma*B preconditioner
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        50  /  50  eigs found
        Updating deflation
        14  /  50  eigs found
        Updating deflation
        5  /  50  eigs found
        Updating deflation
        band_lower= -422.4888728940792
        proposed_sigma= -4224.888728940792 , sigma= -443.61331654558035
        making A-sigma*B preconditioner
        1  /  50  eigs found
        Updating deflation
        band_lower= -4436.1331654558035
        nondiagonal_Rayleigh_error= 3.598241444411492e-10
        positive_error= 5.359260979138846e-17
        zeroing_error= 2.2901072295801614e-11
        intermediate_error= 1.177572665517129e-12
    '''
    assert(N > 0)
    assert(threshold < 0.0)
    assert(gamma < 0.0)
    assert(tol > 0.0)
    assert(sigma_factor > 0.0)
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
    LM_eig = spla.eigsh(spla.LinearOperator((N,N), matvec=A_op.matvec),
                         1,
                         M=spla.LinearOperator((N,N), matvec=B_op.matvec),
                         Minv=spla.LinearOperator((N,N), matvec=iB_op.matvec),
                         which='LM', return_eigenvectors=False,
                         tol=tol)[0]
    printmaybe('LM_eig=', LM_eig)

    if -1.0 < -np.abs(LM_eig):
        return dd0, V0

    sigma = -1.0
    printmaybe('making A-sigma*B preconditioner')
    solve_P = make_OP_preconditioner(sigma)
    DSO = DeflatedShiftedOperator(apply_A, apply_B, sigma, solve_P, gamma, V0, dd0)

    printmaybe('Getting eigs near sigma')
    DSO, d_lower, _ = deflate_negative_eigs_near_sigma(DSO, B_op, threshold, chunk_size,
                                                       ncv_factor, lanczos_maxiter, tol, preconditioner_only, display)
    # printmaybe('d_lower=', d_lower)
    if d_lower is None:
        band_lower = sigma * sigma_factor
    else:
        band_lower = d_lower
    printmaybe('band_lower=', band_lower)

    while -np.abs(LM_eig) < band_lower:
        proposed_sigma = band_lower * sigma_factor
        sigma = np.max([-np.abs(LM_eig) * 1.05, proposed_sigma])
        printmaybe('proposed_sigma=', proposed_sigma, ', sigma=', sigma)

        printmaybe('making A-sigma*B preconditioner')
        solve_P = make_OP_preconditioner(sigma)
        iP_op = CountedOperator((N,N), solve_P, display=False, name='invP')
        DSO = DSO.update_sigma(sigma, iP_op.matvec)
        DSO, d_lower, _ = deflate_negative_eigs_near_sigma(DSO, B_op, band_lower, chunk_size,
                                                           ncv_factor, lanczos_maxiter, tol, preconditioner_only, display)
        if d_lower is None:
            d_lower = sigma * sigma_factor

        band_lower = np.min([d_lower, sigma * sigma_factor])
        printmaybe('band_lower=', band_lower)

    V = DSO.BU
    dd = gamma * DSO.dd

    return dd, V



