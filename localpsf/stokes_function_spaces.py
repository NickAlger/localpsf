import numpy as np
import typing as typ
from scipy.spatial import KDTree
import dolfin as dl
from dataclasses import dataclass
from functools import cached_property
import warnings


@dataclass(frozen=True)
class StokesFunctionSpaces:
    Zh: dl.FunctionSpace    # Zh:  state, also adjoint
    Wh: dl.FunctionSpace    # Wh:  parameter, full 3D domain
    Vh3: dl.FunctionSpace   # Vh3: parameter, 2d basal manifold, 3d coords
    Vh2: dl.FunctionSpace   # Vh2: parameter, 2d basal flat space, 2d coords

    @cached_property
    def Xh(me) -> typ.List[dl.FunctionSpace]:
        return [me.Zh, me.Wh, me.Zh]  # Xh:  (state, parameter full 3D domain, adjoint)

    @cached_property
    def pp_Vh3(me) -> np.ndarray:
        return me.Vh3.tabulate_dof_coordinates()

    @cached_property
    def pp_Vh2(me) -> np.ndarray:
        return me.Vh2.tabulate_dof_coordinates()

    @cached_property
    def pp_Wh(me) -> np.ndarray:
        return me.Wh.tabulate_dof_coordinates()

    @cached_property
    def KDT_Wh(me) -> KDTree:
        return KDTree(me.pp_Wh)

    @cached_property
    def inds_Vh3_in_Wh(me) -> np.ndarray:
        inds = me.KDT_Wh.query(me.pp_Vh3)[1]
        if np.linalg.norm(me.pp_Vh3 - me.pp_Wh[inds, :]) > 1.e-12 * np.linalg.norm(me.pp_Vh3):
            warnings.warn('problem with basal function space inclusion')
        return inds

    @cached_property
    def pp_Vh3_2D(me) -> np.ndarray:
        return me.pp_Vh3[:, :2].copy()

    @cached_property
    def KDT_Vh3_2D(me) -> KDTree:
        return KDTree(me.pp_Vh3_2D)

    @cached_property
    def inds_Vh2_in_Vh3(me) -> np.ndarray:
        inds = me.KDT_Vh3_2D.query(me.pp_Vh2)[1]
        if np.linalg.norm(me.pp_Vh2 - me.pp_Vh3[inds, :2]) > 1.e-12 * np.linalg.norm(me.pp_Vh2):
            warnings.warn('inconsistency between manifold basal mesh and flat basal mesh')
        return inds

    @cached_property
    def inds_Vh2_in_Wh(me) -> np.ndarray:
        return me.inds_Vh3_in_Wh[me.inds_Vh2_in_Vh3]

    def Wh_to_Vh3_numpy(me, v_Wh_numpy: np.ndarray) -> np.ndarray:
        return function_space_restrict_numpy(v_Wh_numpy, me.inds_Vh3_in_Wh)

    def Vh3_to_Wh_numpy(me, v_Vh3_numpy: np.ndarray) -> np.ndarray:
        return function_space_prolongate_numpy(v_Vh3_numpy, me.Wh.dim(), me.inds_Vh3_in_Wh)

    def Vh3_to_Vh2_numpy(me, v_Vh3_numpy: np.ndarray) -> np.ndarray:
        return function_space_restrict_numpy(v_Vh3_numpy, me.inds_Vh2_in_Vh3)

    def Vh2_to_Vh3_numpy(me, v_Vh2_numpy: np.ndarray) -> np.ndarray:
        return function_space_prolongate_numpy(v_Vh2_numpy, me.Vh3.dim(), me.inds_Vh2_in_Vh3)

    def Wh_to_Vh2_numpy(me, v_Wh_numpy: np.ndarray) -> np.ndarray:
        return me.Vh3_to_Vh2_numpy(me.Wh_to_Vh3_numpy(v_Wh_numpy))

    def Vh2_to_Wh_numpy(me, v_Vh2_numpy: np.ndarray) -> np.ndarray:
        return me.Vh3_to_Wh_numpy(me.Vh2_to_Vh3_numpy(v_Vh2_numpy))

    def numpy2petsc_Vh2(me, v_numpy: np.ndarray) -> dl.Vector:
        assert(v_numpy.shape == (me.Vh2.dim()))
        v_petsc = dl.Function(me.Vh2).vector()
        v_petsc[:] = v_numpy.copy()
        return v_petsc

    def numpy2petsc_Vh3(me, v_numpy: np.ndarray) -> dl.Vector:
        assert(v_numpy.shape == (me.Vh3.dim()))
        v_petsc = dl.Function(me.Vh3).vector()
        v_petsc[:] = v_numpy.copy()
        return v_petsc

    def numpy2petsc_Wh(me, v_numpy: np.ndarray) -> dl.Vector:
        assert(v_numpy.shape == (me.Wh.dim()))
        v_petsc = dl.Function(me.Wh).vector()
        v_petsc[:] = v_numpy.copy()
        return v_petsc

    def Wh_to_Vh3_petsc(me, v_Wh_petsc: dl.Vector) -> dl.Vector:
        return me.numpy2petsc_Vh3(me.Wh_to_Vh3_numpy(v_Wh_petsc[:]))

    def Vh3_to_Wh_petsc(me, v_Vh3_petsc: dl.Vector) -> dl.Vector:
        return me.numpy2petsc_Wh(me.Vh3_to_Wh_numpy(v_Vh3_petsc[:]))

    def Vh3_to_Vh2_petsc(me, v_Vh3_petsc: dl.Vector) -> dl.Vector:
        return me.numpy2petsc_Vh2(me.Vh3_to_Vh2_numpy(v_Vh3_petsc[:]))

    def Vh2_to_Vh3_petsc(me, v_Vh2_petsc: dl.Vector) -> dl.Vector:
        return me.numpy2petsc_Vh3(me.Vh2_to_Vh3_numpy(v_Vh2_petsc[:]))

    def Wh_to_Vh2_petsc(me, v_Wh_petsc: dl.Vector) -> dl.Vector:
        return me.Vh3_to_Vh2_petsc(me.Wh_to_Vh3_petsc(v_Wh_petsc))

    def Vh2_to_Wh_petsc(me, v_Vh2_petsc: dl.Vector) -> dl.Vector:
        return me.Vh3_to_Wh_petsc(me.Vh2_to_Vh3_petsc(v_Vh2_petsc))


def make_stokes_function_spaces(
        ice_mesh_3d: dl.Mesh,
        basal_mesh_3d: dl.Mesh,
        basal_mesh_2d: dl.Mesh
) -> StokesFunctionSpaces:
    P1 = dl.FiniteElement("Lagrange", ice_mesh_3d.ufl_cell(), 1)
    P2 = dl.VectorElement("Lagrange", ice_mesh_3d.ufl_cell(), 2)
    TH = P2 * P1

    Zh = dl.FunctionSpace(ice_mesh_3d, TH)                  # Zh:  state, also adjoint
    Wh = dl.FunctionSpace(ice_mesh_3d, 'Lagrange', 1)       # Wh:  parameter, full 3D domain
    Vh3 = dl.FunctionSpace(basal_mesh_3d, 'Lagrange', 1)    # Vh3: parameter, 2d basal manifold, 3d coords
    Vh2 = dl.FunctionSpace(basal_mesh_2d, 'Lagrange', 1)    # Vh2: parameter, 2d basal flat space, 2d coords

    return StokesFunctionSpaces(Zh, Wh, Vh3, Vh2)


def function_space_prolongate_numpy(x_numpy, dim_Yh, inds_Xh_in_Yh):
    y_numpy = np.zeros(dim_Yh)
    y_numpy[inds_Xh_in_Yh] = x_numpy
    return y_numpy

def function_space_restrict_numpy(y_numpy, inds_Xh_in_Yh):
    return y_numpy[inds_Xh_in_Yh].copy()

# def function_space_prolongate_petsc(x_petsc, Yh, inds_Xh_in_Yh):
#     y_petsc = dl.Function(Yh).vector()
#     y_petsc[:] = function_space_prolongate_numpy(x_petsc[:], Yh.dim(), inds_Xh_in_Yh)
#     return y_petsc

# def function_space_restrict_petsc(y_petsc, Xh, inds_Xh_in_Yh):
#     x_petsc = dl.Function(Xh).vector()
#     x_petsc[:] = function_space_restrict_numpy(y_petsc[:], inds_Xh_in_Yh)
#     return x_petsc

# def make_prolongation_and_restriction_operators(Vsmall, Vbig, inds_Vsmall_in_Vbig):
#     Vbig_to_Vsmall_numpy = lambda vb: function_space_restrict_numpy(vb, inds_Vsmall_in_Vbig)
#     Vsmall_to_Vbig_numpy = lambda vs: function_space_prolongate_numpy(vs, Vbig.dim(), inds_Vsmall_in_Vbig)
#
#     Vbig_to_Vsmall_petsc = lambda vb: function_space_restrict_petsc(vb, Vsmall, inds_Vsmall_in_Vbig)
#     Vsmall_to_Vbig_petsc = lambda vs: function_space_prolongate_petsc(vs, Vbig, inds_Vsmall_in_Vbig)
#
#     return Vbig_to_Vsmall_numpy, Vsmall_to_Vbig_numpy, Vbig_to_Vsmall_petsc, Vsmall_to_Vbig_petsc

# pp_Vh3 = Vh3.tabulate_dof_coordinates()
# pp_Vh2 = Vh2.tabulate_dof_coordinates()
# pp_Wh = Wh.tabulate_dof_coordinates()
#
# KDT_Wh = scipy_KDTree(pp_Wh)
# inds_Vh3_in_Wh = KDT_Wh.query(pp_Vh3)[1]
# if np.linalg.norm(pp_Vh3 - pp_Wh[inds_Vh3_in_Wh, :]) / np.linalg.norm(pp_Vh3) > 1.e-12:
#     warnings.warn('problem with basal function space inclusion')
#
# pp_Vh3_2D = pp_Vh3[:,:2]
#
# KDT_Vh3_2D = scipy_KDTree(pp_Vh3_2D)
# inds_Vh2_in_Vh3 = KDT_Vh3_2D.query(pp_Vh2)[1]
# if np.linalg.norm(pp_Vh2 - pp_Vh3[inds_Vh2_in_Vh3, :2]) / np.linalg.norm(pp_Vh2) > 1.e-12:
#     warnings.warn('inconsistency between manifold basal mesh and flat basal mesh')
#
# inds_Vh2_in_Wh = inds_Vh3_in_Wh[inds_Vh2_in_Vh3]
#
# Wh_to_Vh3_numpy, Vh3_to_Wh_numpy, Wh_to_Vh3_petsc, Vh3_to_Wh_petsc = \
#     make_prolongation_and_restriction_operators(Vh3, Wh, inds_Vh3_in_Wh)
#
# Vh3_to_Vh2_numpy, Vh2_to_Vh3_numpy, Vh3_to_Vh2_petsc, Vh2_to_Vh3_petsc = \
#     make_prolongation_and_restriction_operators(Vh2, Vh3, inds_Vh2_in_Vh3)
#
# Wh_to_Vh2_numpy, Vh2_to_Wh_numpy, Wh_to_Vh2_petsc, Vh2_to_Wh_petsc = \
#     make_prolongation_and_restriction_operators(Vh2, Wh, inds_Vh2_in_Wh)