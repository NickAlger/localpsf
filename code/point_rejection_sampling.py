import numpy as np
from gaussblur import periodic_displacement, periodize_pts, make_regular_grid_2d

run_test = False

def points_outside_ellipse(x, C, yy, nsigma=1.0, periodic=True):
    if periodic:
        z = periodic_displacement(x, yy)
    else:
        z = yy - x
    return (np.sum(z * np.linalg.solve(C, z), axis=0) > nsigma)

def select_well_spaced_points(qq, make_C, nsigma=1.0, periodic=True):
    perm = np.random.permutation(qq.shape[1])
    xx = qq[:,perm]

    remaining_inds = np.arange(xx.shape[1])
    selected_inds = []
    while len(remaining_inds) > 1:
        current_ind = remaining_inds[0]
        remaining_inds = remaining_inds[1:]

        x = xx[:, current_ind].reshape((-1, 1))
        C = make_C(x)

        selected_points = xx[:, selected_inds].reshape((xx.shape[0], -1))
        remaining_points = xx[:, remaining_inds].reshape((xx.shape[0], -1))
        select_x = np.all(points_outside_ellipse(x, C, selected_points, nsigma=nsigma, periodic=periodic))

        if select_x:
            selected_inds.append(current_ind)
            remaining_inds = remaining_inds[points_outside_ellipse(x, C, remaining_points, nsigma=nsigma, periodic=periodic)]

    return perm[selected_inds]


if run_test:
    import scipy.linalg as sla
    import matplotlib.pyplot as plt

    d = 2
    n = 50

    def sigma_fct(qq):
        return (0.1 + 0.09 * np.sin(np.pi * qq[0, 0]) * np.cos(np.pi * qq[1, 0]))

    def make_C(x):
        return sigma_fct(x) * np.eye(x.shape[0])

    def get_ellipse(x0, C):
        tt = np.linspace(0,2*np.pi,200)
        unit_circle = np.array([np.sin(tt), np.cos(tt)])
        return x0 + np.dot(sla.sqrtm(C), unit_circle)

    def plot_ellipse(x0, C):
        ee = periodize_pts(get_ellipse(x0,C))
        plt.plot(ee[0,:], ee[1,:],'.k', markersize=1)

    qq = make_regular_grid_2d(n)
    selected_inds = select_well_spaced_points(qq, make_C)

    plt.figure()
    selected_points = qq[:, selected_inds].reshape((qq.shape[0], -1))
    plt.plot(selected_points[0,:], selected_points[1,:], '.', color='blue')

    for i in range(len(selected_inds)):
        x0 = qq[:,selected_inds[i]].reshape((-1, 1))
        C = make_C(x0)
        plot_ellipse(x0,C)
