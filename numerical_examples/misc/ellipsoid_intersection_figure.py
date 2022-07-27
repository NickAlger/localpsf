import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import FormatStrFormatter

from localpsf import localpsf_root

save_dir = localpsf_root / 'numerical_examples' / 'misc' / 'data'
save_dir.mkdir(parents=True, exist_ok=True)
save_dir_str = str(save_dir)


def ellipsoid_intersection_test(Sigma_A, Sigma_B, mu_A, mu_B, tau):
    lambdas, Phi, v_squared = ellipsoid_intersection_test_helper(Sigma_A, Sigma_B, mu_A, mu_B)
    res = minimize_scalar(ellipsoid_K_function,
                          bracket=[0.0, 0.5, 1.0],
                          args=(lambdas, v_squared, tau))
    return (res.fun[0] >= 0)


def ellipsoid_intersection_test_helper(Sigma_A, Sigma_B, mu_A, mu_B):
    lambdas, Phi = eigh(Sigma_A, b=Sigma_B)
    v_squared = np.dot(Phi.T, mu_A - mu_B) ** 2
    return lambdas, Phi, v_squared


def ellipsoid_K_function(ss, lambdas, v_squared, tau):
    ss = np.array(ss).reshape((-1,1))
    lambdas = np.array(lambdas).reshape((1,-1))
    v_squared = np.array(v_squared).reshape((1,-1))
    return 1.-(1./tau**2)*np.sum(v_squared*((ss*(1.-ss))/(1.+ss*(lambdas-1.))), axis=1)



tau=1.3

Sigma_A = np.array([[0.7, 0.02 ], [0.02, 1.1]])
Sigma_B = np.array([[ 0.6, -0.5], [-0.5, 0.7]])

mu_A = np.array([ 0.9, -0.75])
mu_B0 = np.array([-1.4, 0.6])


def plot_ellipse(mu, Sigma, n_std_tau, ax=None, **kwargs):
    # Usage:
    #   https://github.com/NickAlger/nalger_helper_functions/tree/master/jupyter_notebooks/plot_ellipse.ipynb
    if ax is None:
        ax = plt.gca()

    ee, V = np.linalg.eigh(Sigma)
    e_big = ee[1]
    e_small = ee[0]
    v_big = V[:, 1]
    theta = np.arctan(v_big[1] / v_big[0]) * 180. / np.pi

    long_length = n_std_tau * 2. * np.sqrt(e_big)
    short_length = n_std_tau * 2. * np.sqrt(e_small)

    if not ('facecolor' in kwargs):
        kwargs['facecolor'] = 'none'

    ellipse = Ellipse(mu, width=long_length, height=short_length, angle=theta, **kwargs)
    ax.add_artist(ellipse)


dmu = np.array([1.0, 0.15])
# tt = np.linspace(0, 2.5, 20)
tt = np.array([0, 1.5, 3.35])
for ii  in range(len(tt)):
    mu_B = mu_B0 + tt[ii] * dmu

    # min_A, max_A = ellipsoid_bounding_box(mu_A, Sigma_A, tau)
    # min_B, max_B = ellipsoid_bounding_box(mu_B, Sigma_B, tau)

    plt.figure(figsize=(2,3))
    plt.subplot(2,1,1)
    plot_ellipse(mu_A, Sigma_A, n_std_tau=tau, facecolor='gray', edgecolor='k', alpha=0.5, linewidth=1)
    plot_ellipse(mu_B, Sigma_B, n_std_tau=tau, facecolor='gray', edgecolor='k', alpha=0.5, linewidth=1)
    plt.xlim([-2.5, 3.5])
    plt.ylim([-2.25, 2.25])
    plt.gca().set_aspect(1.0)
    plt.gca().axis('off')

    intersect = ellipsoid_intersection_test(Sigma_A, Sigma_B, mu_A, mu_B, tau)
    # plt.title('Ellipsoids intersect: ' + str(intersect))

    lambdas, Phi, v_squared = ellipsoid_intersection_test_helper(Sigma_A, Sigma_B, mu_A, mu_B)

    ss = np.linspace(0., 1., 1000)
    KK = ellipsoid_K_function(ss, lambdas, v_squared, tau)

    Kmin = np.min(KK)

    plt.subplot(2,1,2)
    plt.plot(ss, ss * 0.0, '--', c='gray')
    plt.plot(ss, KK, 'k')
    # if Kmin > 0:
    #     plt.title(r'$K_\mathrm{min} > 0$')
    # else:
    #     plt.title(r'$K_\mathrm{min} < 0$')
    if ii==0:
        plt.ylabel(r'$K(s)$')
    plt.xlabel(r'$s$', labelpad=-10)
    plt.ylim([-0.6, 1.0])
    # plt.xticks([0.0, 1.0])
    plt.xticks([0.0, 1.0])
    if ii==0:
        plt.yticks([0.0, 1.0])
    else:
        plt.yticks([])
        plt.gca().spines.left.set_visible(False)
    plt.gca().spines.right.set_visible(False)
    plt.gca().spines.top.set_visible(False)
    # plt.gca().spines.bottom.set_position('zero')
    plt.gca().spines.left.set_position('zero')
    plt.gca().spines.bottom.set_bounds((0, 1))
    plt.gca().set_aspect(0.5)
    # for tick in plt.gca().xaxis.get_majorticklabels():
    #     tick.set_horizontalalignment("left")
    #
    # for tick in plt.gca().yaxis.get_majorticklabels():
    #     tick.set_verticalalignment("bottom")

    # plt.text(0.03, -0.14, '0.0')
    # plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))


    plt.savefig(save_dir_str + '/ellipsoids_intersect'+str(ii)+'.pdf', bbox_inches='tight', dpi=200)