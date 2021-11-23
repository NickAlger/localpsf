import numpy as np
import dolfin as dl
import matplotlib.pyplot as plt


def column_error_plot(relative_err_vec, function_space_V, point_batches):
    relative_err_fct = dl.Function(function_space_V)
    relative_err_fct.vector()[:] = relative_err_vec

    plt.figure()
    cm = dl.plot(relative_err_fct)
    plt.colorbar(cm)

    num_batches = len(point_batches)
    pp = np.vstack(point_batches)
    plt.plot(pp[:, 0], pp[:, 1], '.r')

    plt.title('Hd columns relative error, ' + str(pp.shape[0]) + ' points')