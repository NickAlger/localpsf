import numpy as np
import dolfin as dl
import ufl
from collections.abc import Callable
from typing import List, Tuple, Any

linear_solver_type = Callable[[dl.PETScVector,  # u in Au=f
                               dl.PETScVector], # f
                              None]

# transpose solve option somehow not implemented properly in PETScMatrix. Have to factor transpose too.
# Bug reported here: https://bitbucket.org/fenics-project/dolfin/issues/1060/petsclusolver-returns-the-same-results-for
def transpose_of_fenics_matrix(A):
    AT = dl.as_backend_type(A).mat().copy()
    AT.transpose()
    return dl.Matrix(dl.PETScMatrix(AT))

class StokesDerivativesAtPoint:
    def __init__(me,
                 misfit_form  : ufl.Form,
                 forward_form : ufl.Form,
                 bcs : List[dl.DirichletBC],
                 m : dl.Function, # Parameter
                 u : dl.Function, # State
                 p : dl.Function, # Adjoint
                 input_vector_transformation  : Callable[[Any], dl.PETScVector] = lambda x: x,
                 output_vector_transformation : Callable[[dl.PETScVector], Any] = lambda x: x,
                 solver_type='mumps'):
        me.misfit_form = misfit_form
        me.bcs = bcs
        me.m = m # Parameter
        me.u = u # State
        me.p = p # Adjoint
        me.input_vector_transformation = input_vector_transformation
        me.output_vector_transformation = output_vector_transformation
        me.solver_type = solver_type

        me.Mh = me.m.function_space()
        me.Uh = me.u.function_space()
        me.Ph = me.p.function_space()

        me.z = dl.Function(me.Mh)           # Direction that Hessian is applied in
        me.du_dm_z = dl.Function(me.Uh)     # Incremental forward
        me.dp_dm_z = dl.Function(me.Ph)     # Incremental adjoint
        me.dp_dm_z_GN = dl.Function(me.Ph)  # Gauss-Newton incremental adjoint

        replacements = dict()
        if dl.TrialFunction(me.Uh) in forward_form.arguments():
            replacements[dl.TrialFunction(me.Uh)] = me.u

        if dl.TestFunction(me.Ph) in forward_form.arguments():
            replacements[dl.TestFunction(me.Ph)] = me.p

        me.bcs0 = list()
        for bc in me.bcs:
            bc0 = dl.DirichletBC(bc)
            bc0.homogenize()
            me.bcs0.append(bc0)

        me.forward_form_ff = ufl.replace(forward_form, replacements)
        me.forward_form_0f = ufl.replace(me.forward_form_ff, {me.p: dl.TestFunction(me.Ph)})
        me.forward_form_01 = ufl.replace(me.forward_form_0f, {me.u: dl.TrialFunction(me.Uh)})

        me.forward_lhs_form = ufl.lhs(me.forward_form_01)
        me.forward_rhs_form = ufl.rhs(me.forward_form_01)

        me.lagrangian = me.misfit_form + me.forward_form_ff
        me.misfit_gradient_form = dl.derivative(me.lagrangian, me.m, dl.TestFunction(me.Mh))

        me.adjoint_form_0f = dl.derivative(me.lagrangian, me.u, dl.TestFunction(me.Uh))
        me.adjoint_form_01 = ufl.replace(me.adjoint_form_0f, {me.p: dl.TrialFunction(me.Ph)})
        me.adjoint_lhs_form = ufl.lhs(me.adjoint_form_01)
        me.adjoint_rhs_form = ufl.rhs(me.adjoint_form_01)

        me.misfit_hessian_action_form = (dl.derivative(me.misfit_gradient_form, me.m, me.z) +
                                         dl.derivative(me.misfit_gradient_form, me.u, me.du_dm_z) +
                                         dl.derivative(me.misfit_gradient_form, me.p, me.dp_dm_z))

        me.incremental_forward_form_0f = (dl.derivative(me.forward_form_0f, me.m, me.z) +
                                          dl.derivative(me.forward_form_0f, me.u, me.du_dm_z))

        me.incremental_forward_form_01 = ufl.replace(me.incremental_forward_form_0f, {me.du_dm_z: dl.TrialFunction(me.Uh)})
        me.incremental_forward_lhs_form = ufl.lhs(me.incremental_forward_form_01)
        me.incremental_forward_rhs_form = ufl.rhs(me.incremental_forward_form_01)

        me.incremental_adjoint_form_0f = (dl.derivative(me.adjoint_form_0f, me.m, me.z) +
                                          dl.derivative(me.adjoint_form_0f, me.u, me.du_dm_z) +
                                          dl.derivative(me.adjoint_form_0f, me.p, me.dp_dm_z))

        me.incremental_adjoint_form_01 = ufl.replace(me.incremental_adjoint_form_0f, {me.dp_dm_z: dl.TrialFunction(me.Ph)})
        me.incremental_adjoint_lhs_form = ufl.lhs(me.incremental_adjoint_form_01)
        me.incremental_adjoint_rhs_form = ufl.rhs(me.incremental_adjoint_form_01)

        GN_replacements = {p: dl.Function(me.Ph), me.dp_dm_z: me.dp_dm_z_GN}  # set p=0, replace incremental adjoint with GN version
        me.GN_misfit_hessian_action_form = ufl.replace(me.misfit_hessian_action_form, GN_replacements)
        me.GN_incremental_adjoint_form_0f = ufl.replace(me.incremental_adjoint_form_0f, GN_replacements)

        me.GN_incremental_adjoint_form_01 = ufl.replace(me.GN_incremental_adjoint_form_0f, {me.dp_dm_z_GN: dl.TrialFunction(me.Ph)})
        me.GN_incremental_adjoint_lhs_form = ufl.lhs(me.GN_incremental_adjoint_form_01)
        me.GN_incremental_adjoint_rhs_form = ufl.rhs(me.GN_incremental_adjoint_form_01)

        me.u_is_current = False
        me.p_is_current = False
        me.du_dm_z_is_current = False
        me.dp_dm_z_is_current = False
        me.dp_dm_z_GN_is_current = False

        me.coeff_matrix = None
        me.linearized_forward_solver = None
        me.coeff_matrix_T = None
        me.adjoint_solver = None

    def build_linearized_forward_solver(me):
        me.coeff_matrix = dl.assemble(me.forward_lhs_form)
        for bc in me.bcs:
            bc.apply(me.coeff_matrix)
        me.linearized_forward_solver = dl.PETScLUSolver(dl.as_backend_type(me.coeff_matrix), me.solver_type)

    def build_adjoint_solver(me):
        me.coeff_matrix_T = transpose_of_fenics_matrix(me.coeff_matrix)
        me.adjoint_solver = dl.PETScLUSolver(dl.as_backend_type(me.coeff_matrix_T), me.solver_type)

    def update_m(me, new_m):
        me.m.vector()[:] = me.input_vector_transformation(new_m)
        me.build_linearized_forward_solver()
        me.build_adjoint_solver()
        me.u_is_current = False
        me.p_is_current = False
        me.du_dm_z_is_current = False
        me.dp_dm_z_is_current = False
        me.dp_dm_z_GN_is_current = False

    def update_z(me, new_z):
        me.z.vector()[:] = me.input_vector_transformation(new_z)
        me.du_dm_z_is_current = False
        me.dp_dm_z_is_current = False
        me.dp_dm_z_GN_is_current = False

    def update_forward(me):
        if not me.u_is_current:
            forward_rhs_vector = dl.assemble(me.forward_rhs_form)
            for bc in me.bcs:
                bc.apply(forward_rhs_vector)
            me.linearized_forward_solver.solve(me.u.vector(), forward_rhs_vector)
            me.u_is_current = True

    def update_adjoint(me):
        if not me.p_is_current:
            adjoint_rhs_vector = dl.assemble(me.adjoint_rhs_form)
            for bc0 in me.bcs0:
                bc0.apply(adjoint_rhs_vector)
            me.adjoint_solver.solve(me.p.vector(), adjoint_rhs_vector)
            me.p_is_current = True

    def update_incremental_forward(me):
        if not me.du_dm_z_is_current:
            incremental_forward_rhs_vector = dl.assemble(me.incremental_forward_rhs_form)
            for bc0 in me.bcs0:
                bc0.apply(incremental_forward_rhs_vector)
            me.linearized_forward_solver.solve(me.du_dm_z.vector(), incremental_forward_rhs_vector)
            me.du_dm_z_is_current = True

    def update_incremental_adjoint(me):
        if not me.dp_dm_z_is_current:
            incremental_adjoint_rhs_vector = dl.assemble(me.incremental_adjoint_rhs_form)
            for bc0 in me.bcs0:
                bc0.apply(incremental_adjoint_rhs_vector)
            me.adjoint_solver.solve(me.dp_dm_z.vector(), incremental_adjoint_rhs_vector)
            me.du_dm_z_is_current = True

    def update_gauss_newton_incremental_adjoint(me):
        if not me.dp_dm_z_GN_is_current:
            GN_incremental_adjoint_rhs_vector = dl.assemble(me.GN_incremental_adjoint_rhs_form)
            for bc0 in me.bcs0:
                bc0.apply(GN_incremental_adjoint_rhs_vector)
            me.adjoint_solver.solve(me.dp_dm_z_GN.vector(), GN_incremental_adjoint_rhs_vector)
            me.dp_dm_z_GN_is_current = True

    def misfit(me):
        me.update_forward()
        return dl.assemble(me.misfit_form)

    def gradient(me):
        me.update_forward()
        me.update_adjoint()
        return me.output_vector_transformation(dl.assemble(me.misfit_gradient_form))

    def apply_hessian(me, z_vec):
        me.update_z(z_vec)
        me.update_forward()
        me.update_adjoint()
        me.update_incremental_forward()
        me.update_incremental_adjoint()
        return me.output_vector_transformation(dl.assemble(me.misfit_hessian_action_form))

    def apply_gauss_newton_hessian(me, z_vec):
        me.update_z(z_vec)
        me.update_forward()
        me.update_adjoint()
        me.update_incremental_forward()
        me.update_gauss_newton_incremental_adjoint()
        return me.output_vector_transformation(dl.assemble(me.GN_misfit_hessian_action_form))