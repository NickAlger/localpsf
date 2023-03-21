import numpy as np
import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import typing as typ
from dataclasses import dataclass
from functools import cached_property


vec2vec = typ.Callable[[np.ndarray], np.ndarray]

@dataclass
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
        assert(norm_r < 1e-6 * norm_b)

        b = np.random.randn(me.N)
        x = me.solve_shifted_deflated(b)
        norm_r = np.linalg.norm(b - me.apply_shifted_deflated(x))
        norm_b = np.linalg.norm(b)
        print('shifted and deflated: norm_r/norm_b=', norm_r / norm_b)
        assert(norm_r < 1e-7 * norm_b)

    def apply_shifted(me, x: np.ndarray) -> np.ndarray: # x -> (A - sigma*B) @ x
        return me.apply_A(x) - me.sigma * me.apply_B(x)

    def solve_shifted(me, b: np.ndarray) -> np.ndarray: # b -> (A - sigma*B)^-1 @ b
        return spla.gmres(spla.LinearOperator((me.N, me.N), matvec=me.apply_shifted), b,
                           M=spla.LinearOperator((me.N, me.N), matvec=me.solve_P),
                           tol=1e-10)[0]

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

    def get_eigs_near_sigma(me, target_num_eigs=10, tol=1e-7, ncv_factor=3, maxiter=2, mode='cayley') -> typ.Tuple[np.ndarray, np.ndarray]: # (dd, U)
        try:
            dd, U = spla.eigsh(spla.LinearOperator((N,N), matvec=me.apply_deflated), target_num_eigs,
                                  sigma=me.sigma,
                                  mode=mode,
                                  M=spla.LinearOperator((N,N), matvec=me.apply_B),
                                  OPinv=spla.LinearOperator((N,N), matvec=me.solve_shifted_deflated),
                                  which='LM', return_eigenvectors=True,
                                  tol=tol,
                                  ncv=ncv_factor*target_num_eigs,
                                  maxiter=maxiter)
        except spla.ArpackNoConvergence as ex:
            U = ex.eigenvectors
            dd = ex.eigenvalues
            print(ex)

        # zz = evals + 0.5
        # dd = mu * zz / (1.0 - zz)
        return dd, U

    def update_sigma(me, new_sigma: float, new_solve_P: vec2vec) -> 'DeflatedShiftedOperator':
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, new_sigma, new_solve_P, me.gamma, me.BU, me.dd)

    def update_gamma(me, new_gamma: float) -> 'DeflatedShiftedOperator':
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_P, new_gamma, me.BU, me.dd)

    def update_deflation(me, BU2: np.ndarray, dd2: np.ndarray) -> 'DeflatedShiftedOperator':
        new_BU = np.hstack([me.BU, BU2])
        new_dd = np.concatenate([me.dd, dd2])
        return DeflatedShiftedOperator(me.apply_A, me.apply_B, me.sigma, me.solve_P, me.gamma, new_BU, new_dd)


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

noise = 0.05*np.random.randn(N,N)
noise = (noise + noise.T) / 2.0

def make_shifted_solver(shift):
    inv_P = np.linalg.inv(A + noise - shift*B)
    return lambda x: inv_P @ x

sigma = 1e0
solve_P = make_shifted_solver(sigma)

gamma = -1.0

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