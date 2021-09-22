import dolfin as dl
import matplotlib.pyplot as plt

from nalger_helper_functions import plot_ellipse


def visualize_impulse_response_batch(impulse_response_batch_f, sample_points_batch, mu_batch, Sigma_batch, tau):
    f = impulse_response_batch_f
    pp = sample_points_batch

    plt.figure()

    cm = dl.plot(f)
    plt.colorbar(cm)

    plt.scatter(pp[:,0], pp[:,1], c='k', s=2)

    for k in range(mu_batch.shape[0]):
        plot_ellipse(mu_batch[k,:], Sigma_batch[k,:,:], n_std_tau=tau, facecolor='none', edgecolor='k', linewidth=1)


def visualize_weighting_function(weighting_functions_ww, points_pp, index_ii):
    plt.figure()
    w = weighting_functions_ww[index_ii]
    cm = dl.plot(w)
    plt.colorbar(cm)
    plt.title('Weighting function '+ str(index_ii))
    plt.scatter(points_pp[:, 0], points_pp[:, 1], c='k', s=2)
    plt.scatter(points_pp[index_ii, 0], points_pp[index_ii, 1], c='r', s=2)