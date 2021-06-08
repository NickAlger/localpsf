from localpsf import *
import scipy.linalg as sla

#### Test geigh_numpy()

def random_spd_matrix(d):
    U, ss, _ = np.linalg.svd(np.random.randn(d, d))
    return np.dot(U, np.dot(np.diag(ss), U.T))

d=5
A = random_spd_matrix(d)
B = random_spd_matrix(d)

lambdas_true, Phi_true = eigh(A, B)
lambdas, Phi = geigh_numpy(A, B)

err_Raya = np.linalg.norm(np.dot(Phi.T, np.dot(A, Phi)) - np.diag(lambdas))
err_Rayb = np.linalg.norm(np.dot(Phi.T, np.dot(B, Phi)) - np.eye(5))

err_Raya_true = np.linalg.norm(np.dot(Phi_true.T, np.dot(A, Phi_true)) - np.diag(lambdas_true))
err_Rayb_true = np.linalg.norm(np.dot(Phi_true.T, np.dot(B, Phi_true)) - np.eye(5))

print('err_Raya=', err_Raya, ', err_Rayb=', err_Rayb, ', err_Raya_true=', err_Raya_true, ', err_Rayb_true=', err_Rayb_true)

from time import time
t = time()
for k in range(1000):
    lambdas_true, Phi_true = eigh(A, B)
dt_eigh = time() - t
print('dt_eigh=', dt_eigh)

t = time()
for k in range(1000):
    lambdas, Phi = geigh_numpy(A, B)
dt_numba = time() - t
print('dt_numba=', dt_numba)


#### Test ellipsoids_intersect()

def many_points_on_ellipsoid_boundary(Sigma, mu, N, tau):
    d = Sigma.shape[0]
    pp = np.random.randn(d, N)
    pp = tau * (pp / np.linalg.norm(pp, axis=0).reshape((1,-1)))
    return np.dot(sla.sqrtm(Sigma), pp).T + mu.reshape((1,d))


def ellipsoids_intersect_brute_force(Sigma_a, Sigma_b, mu_a, mu_b, tau, num_pts=int(1e5)):
    pp_a = many_points_on_ellipsoid_boundary(Sigma_a, mu_a, num_pts, tau)
    return not np.all(points_which_are_not_in_ellipsoid(Sigma_b, mu_b, pp_a, tau))


d=3
num_tests = int(1e2)
all_intersects = np.zeros(num_tests, dtype=bool)
all_intersects_brute = np.zeros(num_tests, dtype=bool)
for k in range(num_tests):
    Sigma_a = random_spd_matrix(d)
    Sigma_b = random_spd_matrix(d)
    mu_a = np.random.randn(d)
    mu_b = np.random.randn(d)
    # mu_b = mu_a
    tau = np.abs(np.random.randn())
    # tau = 1.
    intersect = ellipsoids_intersect(Sigma_a, Sigma_b, mu_a, mu_b, tau)
    intersect_brute = ellipsoids_intersect_brute_force(Sigma_a, Sigma_b, mu_a, mu_b, tau)
    all_intersects[k] = intersect
    all_intersects_brute[k] = intersect_brute

correct_intersects = (all_intersects == all_intersects_brute)
correct_intersects_percentage = np.sum(correct_intersects) / num_tests
fraction_intersecting = np.sum(all_intersects_brute) / num_tests
print('num_tests=', num_tests)
print('correct_intersects_percentage=', correct_intersects_percentage)
print('fraction_intersecting=', fraction_intersecting)

@njit
def many_intersections_numba_timing(Sigmas_a, Sigmas_b, mus_a, mus_b, taus):
    N = len(taus)
    intersects = np.zeros(N, dtype=np.bool_)
    for k in range(len(taus)):
        Sa = Sigmas_a[k,:,:]
        Sb = Sigmas_b[k,:,:]
        mua = mus_a[k,:]
        mub = mus_b[k,:]
        tau = taus[k]
        intersect = ellipsoids_intersect(Sa, Sb, mua, mub, tau)
        intersects[k] = intersect
    return intersects

num_tests = 100000
Sigmas_a = np.array([random_spd_matrix(d) for _ in range(num_tests)])
Sigmas_b = np.array([random_spd_matrix(d) for _ in range(num_tests)])
mus_a = np.random.randn(num_tests, d)
mus_b = np.random.randn(num_tests, d)
taus = np.abs(np.random.randn(num_tests))

intersects0 = many_intersections_numba_timing(Sigmas_a, Sigmas_b, mus_a, mus_b, taus) # compile if not already compiled

t = time()
intersects1 = many_intersections_numba_timing(Sigmas_a, Sigmas_b, mus_a, mus_b, taus)
dt_many_intersections_numba_timing = time() - t
print('num_tests=', num_tests, ', dt_many_intersections_numba_timing=', dt_many_intersections_numba_timing)

####

