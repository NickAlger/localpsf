import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def estimate_column_errors_randomized(apply_A_true_numpy, apply_A_numpy, function_space_V, n_random_error_matvecs):
    ncol_A = function_space_V.dim()
    Y_true = np.zeros((ncol_A, n_random_error_matvecs))
    Y = np.zeros((ncol_A, n_random_error_matvecs))
    for k in tqdm(range(n_random_error_matvecs)):
        omega = np.random.randn(ncol_A)
        Y_true[:, k] = apply_A_true_numpy(omega)
        Y[:, k] = apply_A_numpy(omega)

    norm_A = np.linalg.norm(Y_true) / np.sqrt(n_random_error_matvecs)
    norm_A_err = np.linalg.norm(Y_true - Y) / np.sqrt(n_random_error_matvecs)

    relative_A_err = norm_A_err / norm_A

    A_norm_vec = np.linalg.norm(Y_true, axis=1) / np.sqrt(n_random_error_matvecs)
    A_err_vec = np.linalg.norm(Y_true - Y, axis=1) / np.sqrt(n_random_error_matvecs)
    A_relative_err_vec = A_err_vec / A_norm_vec
    A_relative_err_fct = dl.Function(function_space_V)
    A_relative_err_fct.vector()[:] = A_relative_err_vec

    return relative_A_err, A_relative_err_fct


def column_error_plot(relative_err_fct, point_batches):
    plt.figure()
    cm = dl.plot(relative_err_fct)
    plt.colorbar(cm)

    num_batches = len(point_batches)
    pp = np.vstack(point_batches)
    plt.plot(pp[:, 0], pp[:, 1], '.r')

    plt.title('Hd columns relative error, ' + str(num_batches) + ' batches')