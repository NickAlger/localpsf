import dolfin as dl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from nalger_helper_functions import *


def compute_impulse_response_batches(sample_point_batches, function_space_V,
                                     apply_operator_A, solve_mass_matrix_M):
    '''Computes impulse responses of operator A to Dirac combs associated with batches of points.

    Parameters
    ----------
    sample_point_batches: list of numpy arrays. Sample point batches
        sample_point_batches[b].shape = (num_sample_points_in_batch_b, spatial_dimension)
    function_space_V: fenics FunctionSpace. Function space for impulse response functions
    apply_operator_A: callable. Function that applies the linear operator A to a vector.
        maps fenics vector to fenics vector. apply_operator_A(v) := A v
    solve_mass_matrix_M: callable. Fuction that applies the inverse of the mass matrix to a vector
        maps fenics vector to fenics vector. solve_mass_matrix_M(v) := M^-1 v

    Returns
    -------
    ff: list of fenics Functions. Impulse response batches.
        ff[b] is the result of applying the operator A to the dirac comb associated with b'th batch of sample points.

    '''
    num_batches = len(sample_point_batches)
    ff = list()
    print('Computing Dirac comb impulse responses')
    for b in tqdm(range(num_batches)):
        pp_batch = sample_point_batches[b]
        f = get_one_dirac_comb_response(pp_batch, function_space_V, apply_operator_A, solve_mass_matrix_M)
        ff.append(f)
    return ff


def visualize_impulse_response_batch(impulse_response_batch_f, sample_points_batch, mu_batch, Sigma_batch, tau):
    f = impulse_response_batch_f
    pp = sample_points_batch

    plt.figure()

    cm = dl.plot(f)
    plt.colorbar(cm)

    plt.scatter(pp[:,0], pp[:,1], c='k', s=2)

    for k in range(mu_batch.shape[0]):
        plot_ellipse(mu_batch[k,:], Sigma_batch[k,:,:], n_std_tau=tau, facecolor='none', edgecolor='k', linewidth=1)


def get_one_dirac_comb_response(points_pp, function_space_V, apply_operator_A, solve_mass_matrix_M):
    apply_H = apply_operator_A
    solve_M = solve_mass_matrix_M

    dirac_comb_dual_vector = make_dirac_comb_dual_vector(points_pp, function_space_V)
    dirac_comb_response = dl.Function(function_space_V)
    dirac_comb_response.vector()[:] = solve_M(apply_H(solve_M(dirac_comb_dual_vector)))
    return dirac_comb_response


def make_dirac_comb_dual_vector(points_pp, function_space_V):
    pp = points_pp
    V = function_space_V
    num_pts, d = pp.shape
    dirac_comb_dual_vector = dl.assemble(dl.Constant(0.0) * dl.TestFunction(V) * dl.dx)
    for k in range(num_pts):
        ps = dl.PointSource(V, dl.Point(pp[k,:]), 1.0)
        ps.apply(dirac_comb_dual_vector)
    return dirac_comb_dual_vector