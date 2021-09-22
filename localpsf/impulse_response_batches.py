import dolfin as dl
from tqdm.auto import tqdm


def compute_impulse_response_batches(point_batches, V_in, V_out, apply_A, solve_M_in, solve_M_out):
    '''Computes impulse responses of operator
        A: V_in -> V_out
    to Dirac combs associated with batches of points.

    Parameters
    ----------
    point_batches: list of numpy arrays. Sample point batches
        sample_point_batches[b].shape = (num_sample_points_in_batch_b, spatial_dimension)
    V_in: fenics FunctionSpace.
    V_out: fenics FunctionSpace.
    apply_A: callable. Applies the operator A to a fenics vector, and returns a fenics vector.
    solve_M_in: callable. Applies the inverse of the mass matrix for the input space, V_in,
        to a fenics vector and returns a fenics vector
    solve_M_out: callable. Applies the inverse of the mass matrix for the output space, V_in,
        to a fenics vector and returns a fenics vector

    Returns
    -------
    ff: list of fenics Functions. Impulse response batches.
        ff[b] is the result of applying the operator A to the dirac comb associated with b'th batch of sample points.

    '''
    num_batches = len(point_batches)
    ff = list()
    print('Computing Dirac comb impulse responses')
    for b in tqdm(range(num_batches)):
        pp_batch = point_batches[b]
        f = get_one_dirac_comb_response(pp_batch, V_in, V_out, apply_A, solve_M_in, solve_M_out)
        ff.append(f)
    return ff


def get_one_dirac_comb_response(points_pp, V_in, V_out, apply_A, solve_M_in, solve_M_out):
    dirac_comb_dual_vector = make_dirac_comb_dual_vector(points_pp, V_in)
    dirac_comb_response = dl.Function(V_out)
    dirac_comb_response.vector()[:] = solve_M_out(apply_A(solve_M_in(dirac_comb_dual_vector)))
    return dirac_comb_response


def make_dirac_comb_dual_vector(pp, V):
    num_pts, d = pp.shape
    dirac_comb_dual_vector = dl.assemble(dl.Constant(0.0) * dl.TestFunction(V) * dl.dx)
    for k in range(num_pts):
        ps = dl.PointSource(V, dl.Point(pp[k,:]), 1.0)
        ps.apply(dirac_comb_dual_vector)
    return dirac_comb_dual_vector