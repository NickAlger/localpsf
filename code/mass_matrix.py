import fenics

def make_mass_matrix(function_space_V):
    V = function_space_V
    u_trial = fenics.TrialFunction(V)
    v_test = fenics.TestFunction(V)
    mass_form = u_trial * v_test * fenics.dx
    M = fenics.assemble(mass_form)
    return M

def make_mass_matrix_and_solver(function_space_V):
    V = function_space_V
    # u_trial = fenics.TrialFunction(V)
    # v_test = fenics.TestFunction(V)
    # mass_form = u_trial * v_test * fenics.dx
    # M = fenics.assemble(mass_form)
    M = make_mass_matrix(V)
    prec = fenics.PETScPreconditioner('hypre_amg')
    fenics.PETScOptions.set('pc_hypre_boomeramg_relax_type_coarse', 'jacobi')
    M_solver = fenics.PETScKrylovSolver('cg', prec)
    M_solver.set_operator(M)
    M_solver.parameters['absolute_tolerance'] = 0.0
    M_solver.parameters['relative_tolerance'] = 1e-8
    M_solver.parameters['maximum_iterations'] = 100
    M_solver.parameters['monitor_convergence'] = False
    return M, M_solver