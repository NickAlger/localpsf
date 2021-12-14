import dolfin as dl


def impulse_response_moments(V_in, V_out, apply_At, solve_M_in):
    '''Computes spatially varying volume, mean, and covariance of impulse response function for operator
        A : V_in -> V_out

    Parameters
    ----------
    V_in: fenics FunctionSpace.
    V_out: fenis FunctionSpace.
    apply_At: callable. u -> A^T u. Maps fenics Vector to fenics Vector.
    solve_M_in: callable. u -> M_in^-1 u. Maps fenics Vector to fenics Vector.
        Solver for input space mass matrix

    Returns
    -------
    vol: scalar-valued fenics Function. spatially varying volume of impulse response
    mu: vector-valued fenics Function. spatially varying mean of impulse response
    Sigma: tensor-valued fenics Function. spatially varying covariance of impulse response

    '''
    print('getting spatially varying volume')
    vol = compute_spatially_varying_volume(V_in, V_out, apply_At, solve_M_in)

    print('getting spatially varying mean')
    mu = compute_spatially_varying_mean(V_in, V_out, apply_At, solve_M_in, vol)

    print('getting spatially varying covariance')
    Sigma = get_spatially_varying_covariance(V_in, V_out, apply_At, solve_M_in, vol, mu)

    return vol, mu, Sigma


def compute_spatially_varying_volume(V_in, V_out, apply_At, solve_M_in):
    volume_function_vol = dl.Function(V_in)
    constant_fct = dl.interpolate(dl.Constant(1.0), V_out)
    volume_function_vol.vector()[:] = solve_M_in(apply_At(constant_fct.vector()))
    return volume_function_vol


def compute_spatially_varying_mean(V_in, V_out, apply_At, solve_M_in, vol):
    d = V_in.mesh().geometric_dimension()
    V_in_vec = dl.VectorFunctionSpace(V_in.mesh(), V_in.ufl_element().family(), V_in.ufl_element().degree())

    mu = dl.Function(V_in_vec)
    for k in range(d):
        linear_fct = dl.interpolate(dl.Expression('x[k]', element=V_out.ufl_element(), k=k), V_out)
        mu_k = dl.Function(V_in)
        mu_k.vector()[:] = solve_M_in(apply_At(linear_fct.vector()))
        mu_k = dl.project(mu_k / vol, V_in)
        dl.assign(mu.sub(k), mu_k)

    mu.set_allow_extrapolation(True)
    return mu


def get_spatially_varying_covariance(V_in, V_out, apply_At, solve_M_in, vol, mu):
    d = V_in.mesh().geometric_dimension()
    V_in_mat = dl.TensorFunctionSpace(V_in.mesh(), V_in.ufl_element().family(), V_in.ufl_element().degree())

    covariance_function_Sigma = dl.Function(V_in_mat)
    for k in range(d):
        for j in range(k + 1):
            quadratic_fct = dl.interpolate(dl.Expression('x[k]*x[j]', element=V_out.ufl_element(), k=k, j=j), V_out)
            Sigma_kj = dl.Function(V_in)
            Sigma_kj.vector()[:] = solve_M_in(apply_At(quadratic_fct.vector()))
            #Sigma_kj.vector()[:] = Sigma_kj.vector()[:] / vol.vector()[:] - mu.sub(k).vector()[:] * mu.sub(j).vector()[:]
            #Sigma_kj = dl.interpolate(Sigma_kj / vol - mu.sub(k) * mu.sub(j), V_in)
            Sigma_kj = dl.interpolate(dl.Expression('Skj / v - muk * muj',
                                                    Skj=Sigma_kj, v=vol, muk=mu.sub(k), muj=mu.sub(j),
                                                    element=V_in.ufl_element()), V_in)
            # Sigma_kj = dl.project(Sigma_kj / vol - mu.sub(k) * mu.sub(j), V_in)
            dl.assign(covariance_function_Sigma.sub(k + d * j), Sigma_kj)
            dl.assign(covariance_function_Sigma.sub(j + d * k), Sigma_kj)

    covariance_function_Sigma.set_allow_extrapolation(True)
    return covariance_function_Sigma
