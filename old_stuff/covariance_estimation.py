import numpy as np

def unit_row_vector(d,k):
    ek = np.zeros(d)
    ek[k] = 1.
    return ek.reshape((1,-1))

def compute_col_moments(apply_At, apply_W, qq):
    # qq is d-by-N array of point coordinates (d=dimension (e.g., 2 or 3), N=num_pts)
    # apply_A(V) := A * v
    # apply_At(v) := A^T * v
    # apply_W(v) := W * v
    # A is N-by-N operator, mapping X' -> X
    # W is N-by-N mass matrix, mapping X -> X'
    d, N = qq.shape
    constant_function = np.ones(N)
    col_volumes = apply_At(apply_W(constant_function))
    apply_At_rescaled = lambda v: (1. / col_volumes) * apply_At(v)

    col_means = np.zeros((d,N))
    for ii in range(d):
        linear_function_ii_direction = np.dot(unit_row_vector(d,ii), qq).reshape(-1)
        col_means[ii,:] = apply_At_rescaled(apply_W(linear_function_ii_direction))

    col_vars = np.zeros((d,d,N))
    for ii in range(d):
        for jj in range(ii+1):
            if ii == jj:
                v = unit_row_vector(d, ii)
            else:
                v = unit_row_vector(d, ii) + unit_row_vector(d, jj)
            v = v/np.linalg.norm(v)
            linear_function_v_direction = np.dot(v, qq).reshape(-1)
            quadratic_function_v_direction = linear_function_v_direction**2
            E_x = np.dot(v, col_means)
            # E_x_b = apply_At_rescaled(apply_W(linear_function_v_direction))
            # print('err=', np.linalg.norm(E_x_b - E_x))
            E_x2 = apply_At_rescaled(apply_W(quadratic_function_v_direction))
            col_vars[ii, jj, :] = E_x2 - E_x**2
            col_vars[jj, ii, :] = col_vars[ii, jj, :]

    for ii in range(d):
        for jj in range(d):
            if not (ii == jj):
                col_vars[ii, jj, :] = col_vars[ii, jj, :] - col_vars[ii, ii, :]/2. - col_vars[jj, jj, :]/2.

    return col_volumes, col_means, col_vars


def distance_from_boundary_function(qq, boundary_inds):
    dd = np.zeros(qq.shape[1])
    for ii in range(qq.shape[1]):
        q = qq[:,ii].reshape((-1,1))
        bb = qq[:,boundary_inds].reshape((qq.shape[0], -1))
        dd[ii] = np.min(np.linalg.norm(q - bb, axis=0))
    return dd


def mark_points_near_boundary(qq, boundary_inds, boundary_distances):
    near_boundary = np.zeros(qq.shape[1], dtype=bool)
    for ii in range(len(boundary_inds)):
        bi = boundary_inds[ii]
        b = qq[:,bi].reshape((-1,1))
        dmax = boundary_distances[ii]
        near_boundary[np.linalg.norm(b - qq, axis=0) < dmax] = True
    return near_boundary


def compute_boundary_normal_std(apply_At, apply_W, qq, boundary_inds, col_volumes):
    linear_function_from_boundary = distance_from_boundary_function(qq, boundary_inds)
    quadratic_function_from_boundary = linear_function_from_boundary**2
    apply_At_rescaled = lambda v: (1. / (2. * col_volumes)) * apply_At(v) # Only half of col volume is measured near boundary
    E_x2 = apply_At_rescaled(apply_W(quadratic_function_from_boundary))
    boundary_normal_std = np.sqrt(2. * E_x2)[boundary_inds]
    return boundary_normal_std