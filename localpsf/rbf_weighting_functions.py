import numpy as np
import dolfin as dl
# from scipy.optimize import root_scalar
from nalger_helper_functions import factorized


def make_rbf_weighting_functions(function_space_V, points_pp,
                                 kernel='polyharmonic_spline',
                                 kernel_parameter=2):
    print('Making Poisson weighting functions')
    N, d = points_pp.shape
    dof_coords = function_space_V.tabulate_dof_coords()
    Phi = rbf_interpolant_matrix(points_pp, kernel, kernel_parameter)

    solve_Phi = factorized(Phi)

    ww = list()
    for ii in range(N):
        ff = np.zeros(N)
        ff[ii] = 1.0
        coeffs = solve_Phi(ff)

        wi = dl.Function(function_space_V)
        wi.vector()[:] = eval_rbf_interpolant_at_points(dof_coords,
                                                        coeffs, points_pp,
                                                        kernel, kernel_parameter)
        ww.append(wi)
    return ww


def multiquadric(rr, e):
    return np.sqrt(1. + np.power(e * rr,2))


def inverse_multiquadric(rr, e):
    return 1./np.sqrt(1. + np.power(e * rr,2))


def gaussian(rr, e):
    return np.exp(-np.power(e * rr, 2))


def polyharmonic_spline(rr, k):
    ff = np.zeros(rr.shape)
    if np.mod(k,2) == 0:
        ff[rr > 0] = np.power(rr[rr>0], k) * np.log(rr[rr>0])
    else:
        ff = np.power(rr, k)
    return ff


def eval_rbf_at_points(eval_points, kernel_center, kernel, kernel_parameter):
    rr = np.linalg.norm(kernel_center.reshape((1,-1)) - eval_points, axis=1)
    if kernel == 'multiquadric':
        return multiquadric(rr, kernel_parameter)
    elif kernel == 'inverse_multiquadric':
        return inverse_multiquadric(rr, kernel_parameter)
    elif kernel == 'gaussian':
        return gaussian(rr, kernel_parameter)
    elif kernel == 'polyharmonic_spline':
        return polyharmonic_spline(rr, kernel_parameter)
    else:
        raise RuntimeError('kernel ' + kernel + ' not supported')


def eval_rbf_interpolant_at_points(eval_points, coeffs, kernel_centers, kernel, kernel_params):
    # eval_points = points to evaluate interpolant at, shape=(M,d)
    # coeffs = rbf weights
    # kernel_centers = rbf points, shape=(N,d)
    # kernel = name of kernel type (string)
    # kernel_parameter = parameter passed to kernel function
    ff = np.zeros(eval_points.shape[0])
    for k in range(len(coeffs)):
        ff += coeffs[k] * eval_rbf_at_points(eval_points, kernel_centers[k,:],
                                             kernel, kernel_params)
    return ff


def rbf_interpolant_matrix(kernel_centers, kernel, kernel_parameter):
    # Phi_ij = phi(||kernel_centers[i,:] - kernel_centers[j,:]||)
    N, d = kernel_centers.shape # N=num pts, d=spatial dimension
    Phi = np.zeros((N,N))
    for ii in range(N):
        Phi[ii,:] = eval_rbf_at_points(kernel_centers, kernel_centers[ii,:],
                                       kernel, kernel_parameter)
    return Phi

#
# def choose_shape_tuning_parameter(xx, kernel, kernel_parameter,
#                                   desired_cond=1e12, e_min=1e-5, e_max=1e5):
#     cond_fct = lambda e: np.linalg.cond(rbf_interpolant_matrix(xx, kernel, kernel_parameter))
#
#     tau=10
#     bracket_max = e_max
#     bracket_min = e_max / tau
#     while bracket_min > e_min:
#         cond = cond_fct(bracket_min)
#         print('bracket_min=', bracket_min, ', bracket_max=', bracket_max, ', cond=', cond)
#         if cond > desired_cond:
#             break
#         bracket_max = bracket_min
#         bracket_min = bracket_min / tau
#
#     f = lambda log_e: np.log(cond_fct(np.exp(log_e))) - np.log(desired_cond)
#     sol = root_scalar(f, bracket=[np.log(bracket_min), np.log(bracket_max)], rtol=1e-2)
#     print('sol=', sol)
#     e = np.exp(sol.root)
#     print('shape tuning parameter e=', e, ', cond_fct(e)=', cond_fct(e))
#     return e
#
#
#
# ####
#
# import matplotlib.pyplot as plt
#
# # N = 85
# N = 500
# d = 2
#
# xx = np.random.rand(N, d)
#
# # desired_cond=1e12 # 1e6
# # e = choose_multiquadric_parameter(xx, desired_cond=desired_cond)
# # kernel = 'multiquadric'
# # kernel_parameter = 1e3
# kernel = 'polyharmonic_spline'
# kernel_parameter = 2
# Phi = rbf_interpolant_matrix(xx, kernel, kernel_parameter)
#
# kk=2
#
# # ff = np.zeros(N)
# # ff[kk] = 1.0
# ff = np.ones(N)
# ww = np.linalg.solve(Phi, ff)
#
# X, Y = np.meshgrid(np.linspace(-0., 1., 200), np.linspace(-0., 1., 200))
# pp = np.vstack([X.reshape(-1), Y.reshape(-1)]).T
#
# zz = eval_rbf_interpolant_at_points(pp, ww, xx, kernel, kernel_parameter)
# Z = zz.reshape(X.shape)
#
# plt.figure()
# plt.pcolor(X, Y, Z)
# plt.colorbar()
# plt.title('kernel=' + kernel + ', kernel_parameter='+str(kernel_parameter))
# plt.scatter(xx[:,0], xx[:,1], s=3, c='k')
# plt.plot(xx[kk,0], xx[kk,1], '*r')
#
#
# ##
#
# ee = np.logspace(-5,10,20)
# cc = np.zeros(ee.shape)
# for k in range(len(ee)):
#     cc[k] = np.linalg.cond(multiquadric_matrix(xx, ee[k]))
#
# plt.figure()
# plt.loglog(ee,cc)