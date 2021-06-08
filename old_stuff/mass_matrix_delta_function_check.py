import numpy as np
import fenics
import matplotlib.pyplot as plt
from fenics_to_scipy_sparse_csr_conversion import vec2fct
from heat_inverse_problem import HeatInverseProblem


final_time_test = 5e-3
random_seed = 0

np.random.seed(random_seed)
HIP0 = HeatInverseProblem(mesh_h=1e-3, final_time_T=final_time_test, perform_checks=False, uniform_kappa=True)

delta0_fenics = fenics.assemble(fenics.Constant(0.0) * fenics.TestFunction(HIP0.V) * fenics.dx)
fenics.PointSource(HIP0.V, fenics.Point(0.5, 0.5), 1.0).apply(delta0_fenics)
delta0 = delta0_fenics[:]

def make_all_hessian_appliers(HIP):
    hessian_appliers = []
    hessian_appliers.append(lambda x: HIP.solve_M(HIP.apply_hessian(HIP.solve_M(x))))
    hessian_appliers.append(lambda x: HIP.solve_M(HIP.apply_hessian(x)))
    hessian_appliers.append(lambda x: HIP.solve_M(HIP.apply_hessian(HIP.M * x)))
    hessian_appliers.append(lambda x: HIP.apply_hessian(HIP.solve_M(x)))
    hessian_appliers.append(lambda x: HIP.apply_hessian(x))
    hessian_appliers.append(lambda x: HIP.apply_hessian(HIP.M * x))
    hessian_appliers.append(lambda x: HIP.M * HIP.apply_hessian(HIP.solve_M(x)))
    hessian_appliers.append(lambda x: HIP.M * HIP.apply_hessian(x))
    hessian_appliers.append(lambda x: HIP.M * HIP.apply_hessian(HIP.M * x))

    hessian_appliers_strings = ['iM H iM', 'iM H', 'iM H M', 'H iM', 'H', 'H M', 'M H iM', 'M H', 'M H M']
    return hessian_appliers, hessian_appliers_strings

hessian_appliers0, hessian_appliers_strings = make_all_hessian_appliers(HIP0)

hessian_responses0 = []
for apply_H, H_str in zip(hessian_appliers0, hessian_appliers_strings):
    plt.figure()
    H_delta0 = apply_H(delta0)
    hessian_responses0.append(H_delta0)
    fenics.plot(vec2fct(H_delta0, HIP0.V))
    plt.title(H_str)


mesh_hh = np.logspace(-3, -0.5, 5)
errs = np.zeros((len(hessian_appliers0), len(mesh_hh)))
for k in range(len(mesh_hh)):
    mesh_h = mesh_hh[k]
    HIP = HeatInverseProblem(mesh_h=mesh_h, final_time_T=final_time_test, perform_checks=False, uniform_kappa=True)
    hessian_appliers, _ = make_all_hessian_appliers(HIP)

    delta_fenics = fenics.assemble(fenics.Constant(0.0) * fenics.TestFunction(HIP.V) * fenics.dx)
    fenics.PointSource(HIP.V, fenics.Point(0.5, 0.5), 1.0).apply(delta_fenics)
    delta = delta_fenics[:]

    for j in range(len(hessian_appliers)):
        apply_H = hessian_appliers[j]
        H_delta_true = hessian_responses0[j]

        H_delta_fenics = vec2fct(apply_H(delta), HIP.V)
        H_delta_fenics.set_allow_extrapolation(True)
        H_delta = fenics.interpolate(H_delta_fenics, HIP0.V).vector()[:]

        errs[j,k] = HIP0.M_norm(H_delta - H_delta_true) / HIP0.M_norm(H_delta_true)

plt.figure()
for j in range(len(hessian_appliers0)):
    plt.loglog(mesh_hh, errs[j, :], label=hessian_appliers_strings[j])
    plt.legend()
    plt.xlabel('mesh_h')
    plt.ylabel('relative error')
    plt.title('impulse response relative error')