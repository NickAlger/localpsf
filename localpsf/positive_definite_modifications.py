import numpy as np
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

