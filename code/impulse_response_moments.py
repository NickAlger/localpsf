import dolfin as dl

from mass_matrix import make_mass_matrix
from make_fenics_amg_solver import make_fenics_amg_solver


def impulse_response_moments(function_space_V, apply_operator_transpose_At, solve_mass_matrix_M=None):
    '''Computes spatially varying volume, mean, and covariance of impulse response function for operator A

    Parameters
    ----------
    function_space_V: fenics FunctionSpace.
    apply_operator_transpose_At: callable. u -> A^T u. Maps fenics Vector to fenics Vector.
    solve_mass_matrix_M: (optional) callable. u -> M^-1 u. Maps fenics Vector to fenics Vector.

    Returns
    -------
    vol: scalar-valued fenics Function. spatially varying volume of impulse response
    mu: vector-valued fenics Function. spatially varying mean of impulse response
    Sigma: tensor-valued fenics Function. spatially varying covariance of impulse response

    '''
    V = function_space_V
    apply_At = apply_operator_transpose_At

    print('making mass matrix and solver')
    if solve_mass_matrix_M is None:
        M = make_mass_matrix(V)
        solve_M = make_fenics_amg_solver(M)
    else:
        solve_M = solve_mass_matrix_M

    print('getting spatially varying volume')
    vol = compute_spatially_varying_volume(V, apply_At, solve_M)

    print('getting spatially varying mean')
    mu = compute_spatially_varying_mean(V, apply_At, solve_M, vol)

    print('getting spatially varying covariance')
    Sigma = get_spatially_varying_covariance(V, apply_At, solve_M, vol, mu)

    return vol, mu, Sigma


def compute_spatially_varying_volume(function_space_V, apply_operator_transpose_At, solve_mass_matrix_M):
    V = function_space_V
    apply_Ht = apply_operator_transpose_At
    solve_M = solve_mass_matrix_M

    volume_function_vol = dl.Function(V)
    constant_fct = dl.interpolate(dl.Constant(1.0), V)
    volume_function_vol.vector()[:] = solve_M(apply_Ht(constant_fct.vector()))
    return volume_function_vol


def compute_spatially_varying_mean(function_space_V, apply_operator_transpose_At,
                                   solve_mass_matrix_M, volume_function_vol):
    V = function_space_V
    apply_Ht = apply_operator_transpose_At
    solve_M = solve_mass_matrix_M
    vol = volume_function_vol

    d = V.mesh().geometric_dimension()
    V_vec = dl.VectorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())

    mean_function_mu = dl.Function(V_vec)
    for k in range(d):
        linear_fct = dl.interpolate(dl.Expression('x[k]', element=V.ufl_element(), k=k), V)
        mu_k = dl.Function(V)
        mu_k.vector()[:] = solve_M(apply_Ht(linear_fct.vector()))
        mu_k = dl.project(mu_k / vol, V)
        dl.assign(mean_function_mu.sub(k), mu_k)

    mean_function_mu.set_allow_extrapolation(True)
    return mean_function_mu


def get_spatially_varying_covariance(function_space_V, apply_operator_transpose_At, solve_mass_matrix_M,
                                     volume_function_vol, mean_function_mu):
    V = function_space_V
    apply_Ht = apply_operator_transpose_At
    solve_M = solve_mass_matrix_M
    vol = volume_function_vol
    mu = mean_function_mu

    d = V.mesh().geometric_dimension()
    V_mat = dl.TensorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())

    covariance_function_Sigma = dl.Function(V_mat)
    for k in range(d):
        for j in range(k + 1):
            quadratic_fct = dl.interpolate(dl.Expression('x[k]*x[j]', element=V.ufl_element(), k=k, j=j), V)
            Sigma_kj = dl.Function(V)
            Sigma_kj.vector()[:] = solve_M(apply_Ht(quadratic_fct.vector()))
            Sigma_kj = dl.project(Sigma_kj / vol - mu.sub(k) * mu.sub(j), V)
            dl.assign(covariance_function_Sigma.sub(k + d * j), Sigma_kj)
            dl.assign(covariance_function_Sigma.sub(j + d * k), Sigma_kj)

    covariance_function_Sigma.set_allow_extrapolation(True)
    return covariance_function_Sigma