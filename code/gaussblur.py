import numpy as np

_run_tests = False
# _run_tests = True

def gauss_exp(A, xi):
    eta = np.linalg.solve(A, xi)
    return np.exp(-(1./2.) * np.sum(eta * xi, axis=0))

def gauss_c0(A):
    d = A.shape[0]
    return np.power(2. * np.pi, -d / 2.)

def gauss_const(A):
    return gauss_c0(A) * np.power(np.linalg.det(A), -1./2.)

def gauss(A, xi):
    return gauss_const(A) * gauss_exp(A, xi)

def dgauss_exp(A,xi, dA):
    eta = np.linalg.solve(A, xi)
    return (1./2.) * np.sum(eta * np.dot(dA, eta), axis=0) * gauss_exp(A, xi)

def dgauss_const(A, dA):
    return -(1. / 2.) * gauss_c0(A) * np.power(np.linalg.det(A), -3./2.) * np.linalg.det(A) * np.trace(np.linalg.solve(A, dA))

def dgauss(A,xi, dA):
    return dgauss_const(A, dA) * gauss_exp(A,xi) + gauss_const(A) * dgauss_exp(A,xi, dA)

def grad_gauss_exp(A,xi):
    eta = np.linalg.solve(A, xi)
    return (1./2.) * np.einsum('n,in,jn->nij', gauss_exp(A, xi), eta, eta)

def grad_gauss_const(A):
    return -(1. / 2.) * gauss_c0(A) * np.power(np.linalg.det(A), -3./2.) * np.linalg.det(A) * np.linalg.inv(A).T

def grad_gauss(A,xi):
    return np.einsum('n,ij->nij', gauss_exp(A, xi), grad_gauss_const(A)) + gauss_const(A) * grad_gauss_exp(A, xi)

def periodize_pts(qq, min=np.array([-1,-1]), max=np.array([1,1])):
    delta = (max - min).reshape((-1, 1))
    minc = min.reshape((-1, 1))
    return np.mod(qq - minc, delta) + minc

def periodic_displacement(p,qq, min=np.array([-1,-1]), max=np.array([1,1])):
    p = p.reshape((-1,1))
    qq = qq.reshape((p.shape[0], -1))
    delta = (max-min).reshape((-1,1))
    minc = min.reshape((-1,1))
    return np.mod(qq - p - minc, delta) + minc

def make_regular_grid_2d(n, periodic=False):
    xx = np.linspace(-1, 1, n, endpoint=(not periodic))
    yy = xx.copy()

    X, Y = np.meshgrid(xx, yy)
    qq = np.vstack([X.reshape(-1), Y.reshape(-1)])
    return qq

def make_periodic_2d_laplacian(n):
    Laplacian1d = np.diag(2 * np.ones(n))
    Laplacian1d = Laplacian1d + np.diag(-np.ones(n-1),1)
    Laplacian1d = Laplacian1d + np.diag(-np.ones(n-1),-1)
    Laplacian1d[0,-1] = -1
    Laplacian1d[-1,0] = -1

    Laplacian2d = np.kron(Laplacian1d, np.eye(n)) + np.kron(np.eye(n), Laplacian1d)
    return Laplacian2d

def circular_translate(u, di, dj):
    n = int(np.sqrt(len(u)))
    U = u.reshape((n,n))
    Ubig = np.bmat([[U, U, U],
                    [U, U, U],
                    [U, U, U]])
    ox = n + di
    oy = n + dj
    return Ubig[ox:ox+n, oy:oy+n].reshape(-1).copy()

def make_blur_operator(kernel_function, CC, qq):
    N = qq.shape[1]
    B = np.zeros((N, N))
    for ii in range(N):
        p = qq[:,ii].reshape((2,-1))
        B[:, ii] = kernel_function(CC[ii, :, :], p, qq)
    return B

def make_gradblur_tensor(grad_kernel_function, CC, qq):
    N = qq.shape[1]
    gB = np.zeros((N, CC.shape[0], CC.shape[1], CC.shape[2]))
    for ii in range(N):
        p = qq[:,ii].reshape((2,-1))
        gB[ii, :, :, :] = grad_kernel_function(CC[ii, :, :], p, qq)
    return gB

class GaussBlurDerivatives:
    def __init__(me, CC, qq, Y_true, Omega, approx_kernel_function, grad_approx_kernel_function):
        me.CC = CC # N x d x d
        me.approx_kernel_function = approx_kernel_function
        me.grad_approx_kernel_function = grad_approx_kernel_function
        me.qq = qq # d x N
        me.Y_true = Y_true # N x m
        me.Omega = Omega # N x m

        me.N, me.m = me.Omega.shape
        me.d = me.qq.shape[0]

        me.CC_vec = np.zeros(me.N * me.d * me.d) # vectorized version of blurring covariance

        me.B = np.zeros((me.N, me.N)) # Blur operator
        me.Y = np.zeros((me.N, me.m)) # predicted observations based on current CC
        me.Res = np.zeros((me.N, me.m)) # residual
        me.gB = np.zeros((me.N, me.N, me.d, me.d)) # gradient of blur operator
        me.J = np.inf # Data misfit portion of objective function

        me.g = np.zeros((me.N, me.d, me.d)) # gradient of objective function
        me.F = np.zeros((me.N, me.m, me.N, me.d, me.d)) # Jacobian of blur operator

        me.g_vec = np.zeros(me.N * me.N * me.d) # vectorized version of gradient
        me.Hgn = np.zeros((me.N * me.d * me.d, me.N * me.d * me.d))

        me.recompute_everything()

    def update_CC(me, new_CC):
        me.CC = new_CC
        me.recompute_everything()

    def recompute_everything(me):
        me.CC_vec = me.CC.reshape(-1)

        me.B = me.compute_blur_operator()
        me.Y = me.compute_predicted_data()
        me.Res = me.compute_residual()
        me.gB = me.compute_blur_operator_gradient()
        me.J = me.compute_objective()

        me.g = me.compute_gradient()
        me.F = me.compute_Jacobian()

        me.g_vec = me.g.reshape(me.N * me.d * me.d)
        me.Hgn = me.compute_gauss_newton_hessian()

    def compute_blur_operator(me):
        return make_blur_operator(me.approx_kernel_function, me.CC, me.qq)

    def compute_predicted_data(me):
        return np.dot(me.B, me.Omega)

    def compute_residual(me):
        return me.Y_true - me.Y

    def compute_blur_operator_gradient(me):
        return make_gradblur_tensor(me.grad_approx_kernel_function, me.CC, me.qq)

    def apply_blur_gradient(me, dCC):
        return np.einsum('nmij,mij->nm', me.gB, dCC)

    def compute_objective(me):
        return 0.5 * np.linalg.norm(me.Res) ** 2

    def compute_gradient(me):
        return np.einsum('nz,nmij,mz->mij', -me.Res, me.gB, me.Omega)

    def compute_Jacobian(me):
        gB_m_nij = np.swapaxes(me.gB, 0, 1).reshape((me.N, me.N * me.d * me.d))
        Omega_m_z = me.Omega
        F_m_z_nij = np.einsum('mx,mz->mzx', gB_m_nij, Omega_m_z)
        F_m_z_n_i_j = F_m_z_nij.reshape((me.N, me.m, me.N, me.d, me.d))
        F_n_z_m_i_j = np.swapaxes(F_m_z_n_i_j, 0, 2)
        return F_n_z_m_i_j
        # return np.einsum('nmij,mz->nzmij', me.gB, me.Omega) # slow

    def compute_gauss_newton_hessian(me):
        F_nz_mij = me.F.reshape((me.N * me.m, me.N * me.d * me.d))
        return np.dot(F_nz_mij.T, F_nz_mij)
        # return np.einsum('nzx,nzy->xy', me.F_3tensor, me.F_3tensor) # slow

if _run_tests:
    d=2
    num_xi=25
    M = np.random.randn(d,d)
    M = np.dot(M, M.T)
    xi = np.random.randn(d,num_xi)
    dM = np.random.randn(d,d)
    dM = np.dot(dM, dM.T)
    s = 1e-6

    J0 = gauss_exp(M,xi)
    J1 = gauss_exp(M + s*dM, xi)
    dJ_diff = (J1-J0)/s
    dJ = dgauss_exp(M,xi,dM)
    err_dgauss_exp = np.linalg.norm(dJ - dJ_diff)
    print('s=', s, ', err_dgauss_exp=', err_dgauss_exp)

    K0 = gauss_const(M)
    K1 = gauss_const(M + s*dM)
    dK_diff = (K1-K0)/s
    dK = dgauss_const(M,dM)
    err_dgauss_const = np.linalg.norm(dK - dK_diff)
    print('s=', s, ', err_dgauss_const=', err_dgauss_const)

    L0 = gauss(M,xi)
    L1 = gauss(M + s*dM, xi)
    dL_diff = (L1-L0)/s
    dL = dgauss(M,xi,dM)
    err_dgauss = np.linalg.norm(dL - dL_diff)
    print('s=', s, ', err_dgauss=', err_dgauss)

    err_grad_gauss_exp = np.linalg.norm(np.dot(grad_gauss_exp(M,xi).reshape((num_xi,-1)), dM.reshape(-1)) - dJ)
    print('err_grad_gauss_exp=', err_grad_gauss_exp)

    err_grad_gauss_const = np.linalg.norm(np.dot(grad_gauss_const(M).reshape((-1)), dM.reshape(-1)) - dK)
    print('err_grad_gauss_const=', err_grad_gauss_const)

    err_grad_gauss = np.linalg.norm(np.dot(grad_gauss(M,xi).reshape((num_xi,-1)), dM.reshape(-1)) - dL)
    print('err_grad_gauss=', err_grad_gauss)

    import matplotlib.pyplot as plt

    n=50
    qq = make_regular_grid_2d(n)

    plt.matshow(np.sqrt(np.sum(periodic_displacement(np.array([0, 0]), qq) ** 2, axis=0)).reshape((n, n)))
    plt.matshow(np.sqrt(np.sum(periodic_displacement(np.array([0.3, 0.4]), qq) ** 2, axis=0)).reshape((n, n)))
    plt.matshow(np.sqrt(np.sum(periodic_displacement(np.array([-0.5, 0.5]), qq) ** 2, axis=0)).reshape((n, n)))
    plt.matshow(np.sqrt(np.sum(periodic_displacement(np.array([-0.5, -0.6]), qq) ** 2, axis=0)).reshape((n, n)))
    plt.matshow(np.sqrt(np.sum(periodic_displacement(np.array([0.2, -0.5]), qq) ** 2, axis=0)).reshape((n, n)))