import numpy as np
import typing as typ
import scipy.linalg as sla
import scipy.sparse.linalg as spla


def get_negative_eig_correction_factors(A_shape, apply_A, cutoff, block_size=20, maxiter=50, display=True):
    cutoff = -np.abs(cutoff)

    U = np.zeros((A_shape[0], 0))
    negative_eigs = np.array([])
    E = np.diag(-2 * negative_eigs)

    for k in range(maxiter):
        A2_linop = spla.LinearOperator(A_shape, matvec=lambda x: apply_A(x) + np.dot(U, np.dot(E, np.dot(U.T, x))))

        min_eigs, min_evecs = spla.eigsh(A2_linop, block_size, which='SA')

        negative_inds = min_eigs < 0
        new_negative_eigs = min_eigs[negative_inds]
        new_negative_evecs = min_evecs[:, negative_inds]

        U = np.hstack([U, new_negative_evecs])
        negative_eigs = np.concatenate([negative_eigs, new_negative_eigs])
        E = np.diag(-2 * negative_eigs)

        if display:
            print('k=', k, 'min_eigs=', min_eigs, 'cutoff=', cutoff)
        if (np.max(min_eigs) > cutoff):
            print('negative eigs smaller than cutoff. Good.')
            break

    return U, E, A2_linop


class WoodburyObject: # (A + U*C*V) * x = b
    # # Example:
    # n = 35
    # r = 8
    # A = np.random.randn(n, n)
    # U = np.random.randn(n, r)
    # C = np.random.randn(r, r)
    # V = np.random.randn(r, n)
    # apply_A = lambda x: np.dot(A, x)
    # solve_A = lambda b: np.linalg.solve(A, b)
    # WO = WoodburyObject(apply_A, solve_A, U, C, V)
    #
    # x = np.random.randn(n)
    # err = np.linalg.norm(x - WO.solve_modified_A(apply_A(x) + np.dot(U, np.dot(C, np.dot(V, x)))))
    # print('Woodbury err=', err) # numerical zero
    def __init__(me, apply_A, solve_A, U, C, V, check_correctness=True):
        me.apply_A = apply_A
        me.solve_A = solve_A
        me.U = U
        me.C = C
        me.V = V

        me.shape = (U.shape[0], V.shape[1])
        me.modification_rank = C.shape[0]

        iA_U = np.zeros((me.shape[0], me.modification_rank))
        for ii in range(me.modification_rank):
            iA_U[:,ii] = solve_A(me.U[:,ii])

        me.capacitance_matrix = np.linalg.inv(C) + np.dot(V, iA_U)
        me.inv_capacitance_matrix = np.linalg.inv(me.capacitance_matrix) # should probably factor instead

        me.apply_modified_A_linop = spla.LinearOperator(me.shape, matvec=me.apply_modified_A)
        me.solve_modified_A_linop = spla.LinearOperator(me.shape, matvec=me.solve_modified_A)

        if check_correctness:
            z = np.random.randn(me.shape[1])
            err = np.linalg.norm(z - me.solve_modified_A(me.apply_modified_A(z))) / np.linalg.norm(z)
            print('Woodbury err=', err)

    def solve_capacitance_matrix(me, u):
        return np.dot(me.inv_capacitance_matrix, u)

    def apply_modified_A(me, x):
        return me.apply_A(x) + np.dot(me.U, np.dot(me.C, np.dot(me.V, x)))

    def solve_modified_A(me, b):
        y = me.solve_A(b)
        return y - me.solve_A(np.dot(me.U, me.solve_capacitance_matrix(np.dot(me.V, y))))


def eig_coeff(eig: float,  # bad eigenvalue A*u = eig*B*u
              tau: float,  # tolerance in (0,1)
              method='zero',
              ) -> float:  # correction coefficient
    if method == 'thresh':  # eig + 1 + c = tau
        c = tau - eig - 1.0
    elif method == 'zero':  # eig + 1 + c = 1
        c = -eig
    elif method == 'flip':  # eig + 1 + c = 1 + |eig|
        c = np.abs(eig) - eig
    return c


def bad_generalized_eigs(apply_A: typ.Callable[[np.ndarray], np.ndarray],  # shape=(N,N)
                         apply_B: typ.Callable[[np.ndarray], np.ndarray],  # shape=(N,N)
                         solve_B: typ.Callable[[np.ndarray], np.ndarray],  # shape=(N,N)
                         N: int,
                         tau: float = 0.5,  # tolerance in (0,1)
                         chunk_size: int = 5,
                         max_rank: int = 50,
                         display: bool = False,
                         ) -> typ.Tuple[np.ndarray,  # U=[u0, u1, ...]: shape=(N,k)
                                        np.ndarray,  # V=[B@u0, B@u1, ...]: shape=(N,k)
                                        np.ndarray]:  # eigs=[e0, e1, ...]: shape=(N,))
    '''Finds geigs A@ui=ei*B@ui such that ui.T@(A + B)@ui < tau.

    A is symmetric indefinite
    B is symmetric positive definite
    A.shape=B.shape=(N,N)
    apply_A(x) = A @ x
    apply_B(x) = B @ x
    solve_B(apply_B(x)) = apply_B(solve_B(x)) = x
    '''
    Bop = spla.LinearOperator((N, N), matvec=apply_B)
    iBop = spla.LinearOperator((N, N), matvec=solve_B)
    uu: typ.List[np.ndarray] = []
    vv: typ.List[np.ndarray] = []
    ee: typ.List[float] = []
    while len(ee) < max_rank:
        if ee:
            cc = np.array([eig_coeff(ei, tau, method='flip') for ei in ee])
            VT = np.array(vv)
            extra_term = lambda x: VT.T @ (cc * (VT @ x.reshape(-1)))
        else:
            extra_term = lambda x: 0.0 * x

        def apply_modified_A(x: np.ndarray) -> np.ndarray:
            return apply_A(x) + extra_term(x)

        modified_A_linop = spla.LinearOperator((N, N), matvec=apply_modified_A)

        if display:
            print('computing chunk, chunk_size=', chunk_size)
        ee_chunk, U_chunk = spla.eigsh(modified_A_linop, k=chunk_size,
                                       M=Bop, Minv=iBop, which='SA')
        done = False
        for ii in range(len(ee_chunk)):
            if display:
                print('eig=', ee_chunk[ii], ', tau-1.0=', tau - 1.0)
            if ee_chunk[ii] < tau - 1.0:
                ee.append(ee_chunk[ii])
                uu.append(U_chunk[:, ii])
                vv.append(apply_B(U_chunk[:, ii]))
            if ee_chunk[ii] >= tau - 1.0:
                #                 if display:
                #                     print('Done.')
                done = True
        if done:
            break

    U = np.array(vv).T
    V = np.array(vv).T
    eigs = np.array(ee)
    return U, V, eigs


def bad_eig_correction(apply_A: typ.Callable[[np.ndarray], np.ndarray],  # shape=(N,N)
                       apply_B: typ.Callable[[np.ndarray], np.ndarray],  # shape=(N,N)
                       solve_B: typ.Callable[[np.ndarray], np.ndarray],  # shape=(N,N)
                       N: int,
                       tau: float = 0.5,  # tolerance in (0,1)
                       chunk_size: int = 5,
                       max_rank: int = 50,
                       display: bool = False,
                       method='flip',
                       ) -> typ.Tuple[np.ndarray,  # V: shape=(N,k)
                                      np.ndarray]:  # cc: shape=(N,))
    '''Finds low rank correction for A+B to reduce impact of indefiniteness of A.

    A is symmetric indefinite
    B is symmetric positive definite
    A.shape=B.shape=(N,N)
    apply_A(x) = A @ x
    apply_B(x) = B @ x
    solve_B(apply_B(x)) = apply_B(solve_B(x)) = x

    Bad eigenpairs (ui, ei) of A @ ui = ei B @ ui satisfy
        ui.T @ (A + B) @ ui < tau * ui.T @ B @ ui = tau

    We form low rank correction
        A + B + V * diag(cc) * V.T
    such that:
    1) V * diag(cc) * V.T is diagonalized by the generalized eigenvectors of (A,B)
    2) For all good (ui, ei), we have
        ui.T @ (A + B + V * diag(cc) * V.T) @ ui = ei
    3) For all bad (ui, ei), we have:
        ui.T @ (A + B + V * diag(cc) * V.T) @ ui = tau    (method='thresh')
    or:
        ui.T @ (A + B + V * diag(cc) * V.T) @ ui = 1.0  (method='zero')
    or:
        ui.T @ (A + B + V * diag(cc) * V.T) @ ui = |ei|   (method='flip')

    In:
        import numpy as np
        import scipy.linalg as sla

        N = 250
        tau = 0.75

        tmp = np.random.randn(N,N)
        A = 0.5 * (tmp.T + tmp) + 10.0*np.eye(N)

        tmp2 = np.random.randn(N,N)
        B = sla.sqrtm(tmp2.T @ tmp2)

        # Flip bad geigs with brute force:
        ee, P = sla.eigh(A, B)
        ee_flip = np.zeros(len(ee))
        for ii in range(len(ee)):
            if ee[ii] + 1.0 < tau:
                ee_flip[ii] = 1.0 + np.abs(ee[ii])
            else:
                ee_flip[ii] = 1.0 + ee[ii]

        # Flip bad geigs using bad_eig_correction():
        apply_A = lambda x: A @ x
        apply_B = lambda x: B @ x
        Binv = np.linalg.inv(B) # yeah yeah..
        solve_B = lambda x: Binv @ x

        V, cc = bad_eig_correction(apply_A, apply_B, solve_B, N, tau=tau, display=True)

        # Check correctness:
        M_flip = A + B + V @ np.diag(cc) @ V.T
        rayleigh = P.T @ M_flip @ P
        err_offdiagonal = np.linalg.norm(rayleigh - np.diag(np.diag(rayleigh)))
        print('err_offdiagonal=', err_offdiagonal)

        ee_flip2 = np.diag(rayleigh)
        err_eigs = np.linalg.norm(ee_flip2 - ee_flip)
        print('err_eigs=', err_eigs)
    Out:
        computing chunk, chunk_size= 5
        eig= -2.1169078852087293 , tau-1.0= -0.25
        eig= -1.5119785357654094 , tau-1.0= -0.25
        eig= -1.4801150091446882 , tau-1.0= -0.25
        eig= -1.4142264301418281 , tau-1.0= -0.25
        eig= -1.185010446973965 , tau-1.0= -0.25
        computing chunk, chunk_size= 5
        eig= -1.1327176696065981 , tau-1.0= -0.25
        eig= -1.0916879589192034 , tau-1.0= -0.25
        eig= -1.063314361843518 , tau-1.0= -0.25
        eig= -1.0303695336441607 , tau-1.0= -0.25
        eig= -0.9709878075990548 , tau-1.0= -0.25
        computing chunk, chunk_size= 5
        eig= -0.8495969511308968 , tau-1.0= -0.25
        eig= -0.8291943636018237 , tau-1.0= -0.25
        eig= -0.8018363279042096 , tau-1.0= -0.25
        eig= -0.7830844793418776 , tau-1.0= -0.25
        eig= -0.7403524793086638 , tau-1.0= -0.25
        computing chunk, chunk_size= 5
        eig= -0.735298765100338 , tau-1.0= -0.25
        eig= -0.6636932846854344 , tau-1.0= -0.25
        eig= -0.6383936273234821 , tau-1.0= -0.25
        eig= -0.6144255514080768 , tau-1.0= -0.25
        eig= -0.5795848362962359 , tau-1.0= -0.25
        computing chunk, chunk_size= 5
        eig= -0.574869852579649 , tau-1.0= -0.25
        eig= -0.5432625433613644 , tau-1.0= -0.25
        eig= -0.5190436794411728 , tau-1.0= -0.25
        eig= -0.5131552033100857 , tau-1.0= -0.25
        eig= -0.47017694259277565 , tau-1.0= -0.25
        computing chunk, chunk_size= 5
        eig= -0.46084034665978996 , tau-1.0= -0.25
        eig= -0.4440927670801954 , tau-1.0= -0.25
        eig= -0.43208877326967887 , tau-1.0= -0.25
        eig= -0.41496521814721526 , tau-1.0= -0.25
        eig= -0.3969132555532589 , tau-1.0= -0.25
        computing chunk, chunk_size= 5
        eig= -0.37869939226993904 , tau-1.0= -0.25
        eig= -0.35074103027590675 , tau-1.0= -0.25
        eig= -0.3451955295364068 , tau-1.0= -0.25
        eig= -0.3113603526636124 , tau-1.0= -0.25
        eig= -0.3079814029501603 , tau-1.0= -0.25
        computing chunk, chunk_size= 5
        eig= -0.2973717331807069 , tau-1.0= -0.25
        eig= -0.28115048847204194 , tau-1.0= -0.25
        eig= -0.2746033707682397 , tau-1.0= -0.25
        eig= -0.253627512604388 , tau-1.0= -0.25
        eig= -0.22870982686316774 , tau-1.0= -0.25
        err_offdiagonal= 8.583318868457877e-11
        err_eigs= 2.6052423115160627e-11


    '''
    U, V, eigs = bad_generalized_eigs(apply_A, apply_B, solve_B,
                                      N, tau=tau, chunk_size=chunk_size,
                                      max_rank=max_rank, display=display)

    cc = np.array([eig_coeff(ei, tau, method=method) for ei in list(eigs)])
    return V, cc