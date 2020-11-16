import fenics

def make_fenics_amg_solver(A_petsc):
    prec = fenics.PETScPreconditioner('hypre_amg')
    fenics.PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    solver = fenics.PETScKrylovSolver('cg', prec)
    solver.set_operator(A_petsc)

    def solve_A(b_petsc, atol=0.0, rtol=1e-10, maxiter=100, verbose=False):
        x_petsc = fenics.Vector(b_petsc)
        solver.parameters['absolute_tolerance'] = atol
        solver.parameters['relative_tolerance'] = rtol
        solver.parameters['maximum_iterations'] = maxiter
        solver.parameters['monitor_convergence'] = verbose
        solver.solve(x_petsc, b_petsc)
        return x_petsc

    return solve_A