import numpy as np
import dolfin as dl
from nalger_helper_functions import csr_fenics2scipy


class MassMatrixHelper:
    def __init__(me, V, solver_type='lu', lumping_type='diagonal'):
        me.M_fenics = dl.assemble(dl.TrialFunction(V) * dl.TestFunction(V) * dl.dx)

        me.diagonal_mass_lumps_fenics = dl.Vector()
        me.M_fenics.init_vector(me.diagonal_mass_lumps_fenics, 1)
        me.M_fenics.get_diagonal(me.diagonal_mass_lumps_fenics)

        me.simple_mass_lumps_fenics = dl.assemble(dl.Constant(1.0) * dl.TestFunction(V) * dl.dx)

        me.prec = dl.PETScPreconditioner('hypre_amg')
        dl.PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
        me.amg_solver = dl.PETScKrylovSolver('cg', me.prec)
        me.amg_solver.set_operator(me.M_fenics)

        me.lu_solver = dl.LUSolver(me.M_fenics)

        me.M_scipy = csr_fenics2scipy(me.M_fenics)
        me.diagonal_mass_lumps_numpy = me.diagonal_mass_lumps_fenics[:]
        me.simple_mass_lumps_numpy = me.simple_mass_lumps_fenics[:]

        me.solver_type = 'lu'
        me.solver = me.lu_solver
        me.set_solver_type(solver_type)

        me.lumping_type = 'diagonal'
        me.mass_lumps_fenics = me.diagonal_mass_lumps_fenics
        me.mass_lumps_numpy = me.diagonal_mass_lumps_numpy
        me.set_lumping_type(lumping_type)

    def set_solver_type(me, new_solver_type):
        if new_solver_type.lower() == 'amg':
            me.solver_type = new_solver_type
            me.solver = me.amg_solver
        elif new_solver_type.lower() == 'lu':
            me.solver_type = new_solver_type
            me.solver = me.lu_solver
        else:
            raise RuntimeError('solver_type ', new_solver_type, ' invalid. must be lu or amg')

    def set_lumping_type(me, new_lumping_type):
        if new_lumping_type.lower() == 'diagonal':
            me.lumping_type = 'diagonal'
            me.mass_lumps_fenics = me.diagonal_mass_lumps_fenics
            me.mass_lumps_numpy = me.diagonal_mass_lumps_numpy
        elif new_lumping_type.lower() == 'simple':
            me.lumping_type = 'simple'
            me.mass_lumps_fenics = me.simple_mass_lumps_fenics
            me.mass_lumps_numpy = me.simple_mass_lumps_numpy
        else:
            raise RuntimeError('lumping_type ', new_lumping_type, ' invalid. must be diagonal or simple')

    def apply_M_fenics(me, x_fenics): # b = M * x
        return me.M_fenics * x_fenics

    def apply_ML_fenics(me, x_fenics): # b = ML * x
        b_fenics = dl.Vector()
        me.M_fenics.init_vector(b_fenics, 0)
        b_fenics[:] = x_fenics[:] * me.mass_lumps_fenics[:]
        return b_fenics

    def solve_M_fenics(me, b): # x = M \ b
        x = dl.Vector()
        me.M_fenics.init_vector(x, 1)
        me.solver.solve(x, b)
        return x

    def solve_ML_fenics(me, b): # x = ML \ b
        x = dl.Vector()
        me.M_fenics.init_vector(x, 1)
        x[:] = b[:] / me.mass_lumps_fenics[:]
        return x

    def numpy2fenics(me, z_numpy, axis):
        z_fenics = dl.Vector()
        me.M_fenics.init_vector(z_fenics, axis)
        z_fenics[:] = z_numpy
        return z_fenics

    def apply_M_numpy(me, x_numpy): # b = M * x
        return me.apply_M_fenics(me.numpy2fenics(x_numpy, 1))[:]

    def apply_ML_numpy(me, x_numpy): # b = ML * x
        return x_numpy * me.mass_lumps_fenics[:]

    def solve_M_numpy(me, b_numpy): # x = M \ b
        return me.solve_M_fenics(me.numpy2fenics(b_numpy, 0))[:]

    def solve_ML_numpy(me, b_numpy):
        return me.solve_ML_fenics(me.numpy2fenics(b_numpy, 0))[:]
