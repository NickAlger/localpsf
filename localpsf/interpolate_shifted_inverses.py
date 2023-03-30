import numpy as np
import typing as typ
import numpy.typing as npt


_RTOL = 1e-10

vec2vec = typ.Callable[[np.ndarray], np.ndarray]

def shifted_inverse_interpolation_coeffs(mu: float,
                                         mu0: float,
                                         mu1: float,
                                         lambda_max: float) -> typ.Tuple[float, float]: # (c0, c1)
    '''Finds c0, c1 such that (A + mu*B)^-1 =approx= c0*(A + mu0*B)^-1 + c1*(A + mu1*B)^-1
    Requirements:
        mu0, mu1 > 0
        mu0 <= mu <= mu1
        A is symmetric positive semi-definite
        B is symmetric positive definite
        lambda_max is the largest eigenvalue of the generalized eigenvalue problem A@u = lambda*B@u

    Let f(t) = 1/(t + mu) and g(t) = c0/(t + mu0) + c1/(t + mu1)
    c0 and c1 are chosen so that g(0)=f(0) and g(lambda_max)=f(lambda_max)
    '''
    assert(mu0 > 0)
    assert(mu1 > 0)
    assert(mu0 <= mu)
    assert(mu <= mu1)
    assert(0.0 < lambda_max)
    M = np.array([[1.0 / mu0,                1.0 / mu1],
                  [1.0 / (mu0 + lambda_max), 1.0 / (mu1 + lambda_max)]])
    b = np.array([1.0 / mu, 1.0 / (mu + lambda_max)])
    cc = np.linalg.solve(M, b)
    c0 = cc[0]
    c1 = cc[1]

    f = lambda t: 1.0 / (t + mu)
    g = lambda t: c0 / (t + mu0) + c1 / (t + mu1)

    assert(np.abs(f(0.0) - g(0.0)) <= _RTOL * np.abs(f(0.0)))
    assert (np.abs(f(lambda_max) - g(lambda_max)) <= _RTOL * np.abs(f(lambda_max)))

    return c0, c1


def shifted_inverse_interpolation_preconditioner(b: np.ndarray,
                                                 mu: float,
                                                 known_mus: npt.ArrayLike,
                                                 known_shifted_solvers: typ.List[vec2vec],
                                                 lambda_max: float,
                                                 display=False
                                                 ) -> np.ndarray:
    '''Solves x = c0*(A + mu_low*B)^-1 @ b + c1*(A + mu_high*B)^-1 @ b =approx= (A + mu*B)^-1 b,
    where
        (mu_low, mu_high) is the smallest bracket containing mu among known_mus, i.e., mu_low <= mu <= mu_high
        c0 and c1 are chosen so that x =approx= (A + mu*B)^-1 b

    If mu is outside of the range of known_mus, only the nearest known mu is used.
    known_mus = [mu0, mu1, ...]
    known_shifted_solvers = [b -> (A + mu0*B)^-1 @ b, b -> (A + mu1*B)^-1 @ b, ...]

    In:
        import scipy.sparse.linalg as spla

        N = 1000
        diag_A = np.zeros(N)
        diag_A[:int(N / 2)] = np.abs(np.random.randn(int(N / 2)))
        diag_B = np.abs(np.random.randn(N))

        lambda_max = np.max(spla.eigsh(spla.LinearOperator((N, N), matvec=lambda x: x * diag_A), 1, which='LM',
                                       M=spla.LinearOperator((N, N), matvec=lambda x: x * diag_B),
                                       Minv=spla.LinearOperator((N, N), matvec=lambda x: x / diag_B))[0])

        known_mus = list(np.logspace(np.log10(lambda_max / 1.0e4), np.log10(lambda_max / 2.0), 3))
        known_shifted_solvers = [lambda b, mu_k=mu_k: b / (diag_A + mu_k * diag_B) for mu_k in known_mus]

        print('known_mus=', known_mus)

        unknown_mus = np.logspace(np.log10(np.min(known_mus)), np.log10(np.max(known_mus)), 15)
        for mu in list(unknown_mus):
            mu_applier = lambda x: x * (diag_A + mu * diag_B)
            mu_solver = lambda b: b / (diag_A + mu * diag_B)
            mu_preconditioner = lambda b: shifted_inverse_interpolation_preconditioner(b, mu, known_mus, known_shifted_solvers,
                                                                                       lambda_max, display=False)
            X = np.zeros((N, N))
            for ii in range(N):
                ei = np.zeros(N)
                ei[ii] = 1.0
                X[:, ii] = mu_applier(mu_preconditioner(ei))

            cond = np.max(diag_A + mu * diag_B) / np.min(diag_A + mu * diag_B)
            B_cond = np.max(diag_A / diag_B + mu) / np.min(diag_A / diag_B + mu)
            interp_cond = np.linalg.cond(X)
            print('mu=', mu, ', cond=', cond, ', B_cond=', B_cond, ', interp_cond=', interp_cond)
    Out:
        known_mus= [0.44801189590616924, 31.679224964749388, 2240.059479530847]
        mu= 0.44801189590616924 , cond= 1986.1157995478225 , B_cond= 10001.000000000002 , interp_cond= 1.0000000000000002
        mu= 0.8231930383493751 , cond= 1338.6468635288143 , B_cond= 5443.36740393845 , interp_cond= 1.5484929234252842
        mu= 1.5125642523760143 , cond= 986.2704802835584 , B_cond= 2962.936295945174 , interp_cond= 2.149070652500697
        mu= 2.779239511249139 , cond= 833.0160054655948 , B_cond= 1612.9945549594206 , interp_cond= 2.5661304600404184
        mu= 5.106673814851054 , cond= 791.1100787837074 , B_cond= 878.3066621237418 , interp_cond= 2.564046703368669
        mu= 9.383184625050367 , cond= 772.4352799440264 , B_cond= 478.46251812002953 , interp_cond= 2.1447501503876247
        mu= 17.240998133018525 , cond= 772.4352799440265 , B_cond= 260.8526445218819 , interp_cond= 1.5453985766418425
        mu= 31.679224964749388 , cond= 772.4352799440265 , B_cond= 142.42135623730957 , interp_cond= 1.0000000000000002
        mu= 58.208537964240065 , cond= 772.4352799440265 , B_cond= 77.9666979406701 , interp_cond= 1.475439727006768
        mu= 106.95444398354405 , cond= 772.4352799440265 , B_cond= 42.88810480610793 , interp_cond= 1.9279762519730617
        mu= 196.5219104945853 , cond= 772.4352799440265 , B_cond= 23.79704562095193 , interp_cond= 2.1693374597717066
        mu= 361.09636837889576 , cond= 1043.7244745696426 , B_cond= 13.406989799356657 , interp_cond= 2.08640096598324
        mu= 663.491347749847 , cond= 1805.0297006798069 , B_cond= 7.75233968650155 , interp_cond= 1.75205787121982
        mu= 1219.1226694282009 , cond= 2993.280961794361 , B_cond= 4.6748713410136 , interp_cond= 1.3451666729559972
        mu= 2240.059479530847 , cond= 4664.401127964695 , B_cond= 2.9999999999999996 , interp_cond= 1.0000000000000002
    '''
    assert(len(known_mus) == len(known_shifted_solvers))
    assert(len(known_mus) >= 1)
    known_mus = np.array(known_mus)
    sort_inds = np.argsort(known_mus)
    known_mus = known_mus[sort_inds]
    known_shifted_solvers = [known_shifted_solvers[ind] for ind in list(sort_inds)]
    min_mu = known_mus[0]
    max_mu = known_mus[-1]
    if mu <= min_mu:
        if display:
            print('shifted_inverse_interpolation_solve: (mu, min_mu, max_mu)=', (mu, min_mu, max_mu))
        x = known_shifted_solvers[0](b)
    elif max_mu <= mu:
        if display:
            print('shifted_inverse_interpolation_solve: (min_mu, max_mu, mu)=', (min_mu, max_mu, mu))
        x = known_shifted_solvers[-1](b)
    else:
        ind_high = np.searchsorted(known_mus, mu)
        ind_low = ind_high - 1
        mu_low = known_mus[ind_low]
        mu_high = known_mus[ind_high]
        if display:
            print('shifted_inverse_interpolation_solve: (mu_low, mu, mu_high)=', (mu_low, mu, mu_high))
        c_low, c_high = shifted_inverse_interpolation_coeffs(mu, mu_low, mu_high, lambda_max)
        x_low = known_shifted_solvers[ind_low](b)
        x_high = known_shifted_solvers[ind_high](b)
        x = c_low * x_low + c_high * x_high
    return x


import scipy.sparse.linalg as spla

N = 1000
diag_A = np.zeros(N)
diag_A[:int(N / 2)] = np.abs(np.random.randn(int(N / 2)))
diag_B = np.abs(np.random.randn(N))

lambda_max = np.max(spla.eigsh(spla.LinearOperator((N, N), matvec=lambda x: x * diag_A), 1, which='LM',
                               M=spla.LinearOperator((N, N), matvec=lambda x: x * diag_B),
                               Minv=spla.LinearOperator((N, N), matvec=lambda x: x / diag_B))[0])

known_mus = list(np.logspace(np.log10(lambda_max / 1.0e4), np.log10(lambda_max / 2.0), 3))
known_shifted_solvers = [lambda b, mu_k=mu_k: b / (diag_A + mu_k * diag_B) for mu_k in known_mus]

print('known_mus=', known_mus)

unknown_mus = np.logspace(np.log10(np.min(known_mus)), np.log10(np.max(known_mus)), 15)
for mu in list(unknown_mus):
    mu_applier = lambda x: x * (diag_A + mu * diag_B)
    mu_solver = lambda b: b / (diag_A + mu * diag_B)
    mu_preconditioner = lambda b: shifted_inverse_interpolation_preconditioner(b, mu, known_mus, known_shifted_solvers,
                                                                               lambda_max, display=False)
    X = np.zeros((N, N))
    for ii in range(N):
        ei = np.zeros(N)
        ei[ii] = 1.0
        X[:, ii] = mu_applier(mu_preconditioner(ei))

    cond = np.max(diag_A + mu * diag_B) / np.min(diag_A + mu * diag_B)
    B_cond = np.max(diag_A / diag_B + mu) / np.min(diag_A / diag_B + mu)
    interp_cond = np.linalg.cond(X)
    print('mu=', mu, ', cond=', cond, ', B_cond=', B_cond, ', interp_cond=', interp_cond)


