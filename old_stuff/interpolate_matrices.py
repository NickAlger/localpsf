import fenics


def interpolate_matrix(weighting_function_w, matrix_C):
    w = weighting_function_w
    C = matrix_C
    V = w.function_space()
    d = V.mesh().geometric_dimension()
    V_mat = fenics.TensorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())

    matrix_field = fenics.Function(V_mat)
    for k in range(d):
        for j in range(d):
            M_kj = fenics.Function(V)
            M_kj.vector()[:] = C[k ,j] * w.vector()
            fenics.assign(matrix_field.sub(k + d* j), M_kj)

    return matrix_field


def interpolate_matrices(weighting_functions_ww, matrices_CC):
    ww = weighting_functions_ww
    CC = matrices_CC
    V = ww[0].function_space()
    d = V.mesh().geometric_dimension()
    V_mat = fenics.TensorFunctionSpace(V.mesh(), V.ufl_element().family(), V.ufl_element().degree())
    matrix_field = fenics.Function(V_mat)
    for k in range(len(ww)):
        matrix_field.vector()[:] = matrix_field.vector() + interpolate_matrix(ww[k], CC[k]).vector()

    matrix_field.set_allow_extrapolation(True)
    return matrix_field