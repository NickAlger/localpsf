import numpy as np
from scipy.io import savemat
import hlibpro_bindings as hpro_cpp

default_rtol = 1e-7
default_atol = 1e-12

hpro_cpp.initialize_hlibpro()

class HMatrixWrapper:
    def __init__(me, cpp_hmatrix_object, bct):
        me.cpp_object = cpp_hmatrix_object
        me.bct = bct
        me.shape = (me.cpp_object.rows(), me.cpp_object.cols())

    def row_ct(me):
        return me.bct.row_ct()

    def col_ct(me):
        return me.bct.col_ct()

    def copy(me):
        return HMatrixWrapper(me.cpp_object.copy(), me.bct)

    def copy_struct(me):
        return HMatrixWrapper(me.cpp_object.copy_struct(), me.bct)

    def matvec(me, x):
        return h_matvec(me, x)

    def __add__(me, other, rtol=default_rtol, atol=default_atol):
        if isinstance(other, HMatrixWrapper):
            return h_add(me, other, rtol=rtol, atol=atol)

        else:
            raise RuntimeError('cannot add HMatrixWrapper to ' + str(other.type))

    def __sub__(me, other, rtol=default_rtol, atol=default_atol):
        if isinstance(other, HMatrixWrapper):
            return h_add(me, other, beta=-1.0, rtol=rtol, atol=atol)

        else:
            raise RuntimeError('cannot add HMatrixWrapper to ' + str(other.type))

    def __mul__(me, other, rtol=default_rtol, atol=default_atol, display_progress=True):
        if isinstance(other, HMatrixWrapper):
            return h_mul(me, other, rtol=rtol, atol=atol, display_progress=display_progress)

        if isinstance(other, float) or isinstance(other, np.number):
            return h_scale(me, other)

        if isinstance(other, np.ndarray) and ( other.shape == (me.shape[1],) ):
            return me.matvec(other)

        else:
            raise RuntimeError('cannot multiply HMatrixWrapper with ' + str(other.type))

    def __rmul__(me, other, rtol=default_rtol, atol=default_atol, display_progress=True):
        if isinstance(other, HMatrixWrapper):
            return h_mul(other, me, rtol=rtol, atol=atol, display_progress=display_progress)

        if isinstance(other, float) or isinstance(other, np.number):
            return h_scale(me, other)

        else:
            raise RuntimeError('cannot right multiply HMatrixWrapper with ' + str(other.type))

    def factorized_inverse(me, rtol=default_rtol, atol=default_atol, display_progress=True):
        return h_factorized_inverse(me, rtol=rtol, atol=atol, display_progress=display_progress)


def h_add(A_hmatrix, B_hmatrix, alpha=1.0, beta=1.0, rtol=default_rtol, atol=default_atol):
    # C = A + alpha * B to tolerance given by truncation accuracy object acc
    acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    C_hmatrix = B_hmatrix.copy()
    hpro_cpp.add(alpha, A_hmatrix.cpp_object, beta, C_hmatrix.cpp_object, acc)
    return C_hmatrix


def h_scale(A_hmatrix, alpha):
    # C = alpha * A
    C_hmatrix = A_hmatrix.copy()
    C_hmatrix.cpp_object.scale(alpha)
    return C_hmatrix

def h_mul(A_hmatrix, B_hmatrix, alpha=1.0, rtol=default_rtol, atol=default_atol, display_progress=True):
    # C = A * B
    acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    C_hmatrix = A_hmatrix.copy_struct()
    # C_hmatrix = A_hmatrix.copy()
    if display_progress:
        hpro_cpp.multiply_with_progress_bar(alpha, hpro_cpp.apply_normal, A_hmatrix.cpp_object,
                                            hpro_cpp.apply_normal, B_hmatrix.cpp_object,
                                            0.0, C_hmatrix.cpp_object, acc)
    else:
        hpro_cpp.multiply_without_progress_bar(alpha, hpro_cpp.apply_normal, A_hmatrix.cpp_object,
                                               hpro_cpp.apply_normal, B_hmatrix.cpp_object,
                                               0.0, C_hmatrix.cpp_object, acc)
    return C_hmatrix

class FactorizedInverseHMatrixWrapper:
    def __init__(me, cpp_object, factors_cpp_object, inverse_bct):
        me.cpp_object = cpp_object
        me.bct = inverse_bct
        me._factors_cpp_object = factors_cpp_object  # Don't mess with this!! Could cause segfault if deleted
        me.shape = (me._factors_cpp_object.rows(), me._factors_cpp_object.cols())

    def row_ct(me):
        return me.bct.row_ct()

    def col_ct(me):
        return me.bct.col_ct()

    def matvec(me, x):
        return h_factorized_solve(me, x)


def h_factorized_inverse(A_hmatrix, rtol=default_rtol, atol=default_atol, display_progress=True):
    acc = hpro_cpp.TTruncAcc(relative_eps=rtol, absolute_eps=atol)
    factors_cpp_object = A_hmatrix.cpp_object.copy()
    cpp_object = hpro_cpp.factorize_inv_with_progress_bar(factors_cpp_object, acc)
    return FactorizedInverseHMatrixWrapper(cpp_object, factors_cpp_object, A_hmatrix.bct)


def build_hmatrix_from_scipy_sparse_matrix(A_csc, bct):
    A_csc[1,0] += 1e-14 # Force non-symmetry
    fname = "temp_sparse_matrix.mat"
    savemat(fname, {'A': A_csc})
    hmatrix_cpp_object = hpro_cpp.build_hmatrix_from_sparse_matfile(fname, bct)
    return HMatrixWrapper(hmatrix_cpp_object, bct)


def h_factorized_solve(iA_factorized, y):
    return hpro_cpp.h_factorized_inverse_matvec(iA_factorized.cpp_object,
                                                iA_factorized.row_ct(),
                                                iA_factorized.col_ct(), y)


def h_matvec(A_hmatrix, x):
    return hpro_cpp.h_matvec(A_hmatrix.cpp_object, A_hmatrix.row_ct(), A_hmatrix.col_ct(), x)

def visualize_hmatrix(A_hmatrix, title):
    hpro_cpp.visualize_hmatrix(A_hmatrix.cpp_object, title)

def visualize_inverse_factors(iA_factorized, title):
    hpro_cpp.visualize_hmatrix(iA_factorized._factors_cpp_object, title)

build_cluster_tree_from_dof_coords = hpro_cpp.build_cluster_tree_from_dof_coords
build_block_cluster_tree = hpro_cpp.build_block_cluster_tree
visualize_cluster_tree = hpro_cpp.visualize_cluster_tree
visualize_block_cluster_tree = hpro_cpp.visualize_block_cluster_tree


