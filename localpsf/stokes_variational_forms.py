import numpy as np
import typing as typ
from dataclasses import dataclass
from functools import cached_property
import dolfin as dl
import ufl

from .assertion_helpers import *
from .filesystem_helpers import localpsf_root
from .stokes_mesh import StokesMeshes, make_stokes_meshes
from .stokes_function_spaces import StokesFunctionSpaces, make_stokes_function_spaces
from .derivatives_at_point import DerivativesAtPoint
from .bilaplacian_regularization_lumped import BilaplacianRegularization, make_bilaplacian_covariance
from .inverse_problem_objective import InverseProblemObjective

from nalger_helper_functions import load_image_into_fenics
import hlibpro_python_wrapper as hpro


@dataclass(frozen=True)
class LinearStokesInverseProblemUnregularized:
    meshes: StokesMeshes
    function_spaces: StokesFunctionSpaces
    derivatives: DerivativesAtPoint

    mtrue_Wh: dl.Function
    utrue: dl.Function
    uobs: dl.Function

    def __post_init__(me):
        assert_equal(me.function_spaces.Zh.mesh(), me.meshes.ice_mesh_3d)
        assert_equal(me.function_spaces.Wh.mesh(), me.meshes.ice_mesh_3d)
        assert_equal(me.function_spaces.Vh3.mesh(), me.meshes.basal_mesh_3d)
        assert_equal(me.function_spaces.Vh2.mesh(), me.meshes.basal_mesh_2d)
        
        assert_equal(me.m_Wh.function_space(), me.function_spaces.Wh)
        assert_equal(me.u.function_space(), me.function_spaces.Zh)
        assert_equal(me.p.function_space(), me.function_spaces.Zh)
        assert_equal(me.mtrue_Wh.function_space(), me.function_spaces.Wh)
        assert_equal(me.utrue.function_space(), me.function_spaces.Zh)
        assert_equal(me.uobs.function_space(), me.function_spaces.Zh)

    @cached_property
    def m_Wh(me) -> dl.Function:
        return me.derivatives.m

    @cached_property
    def u(me) -> dl.Function:
        return me.derivatives.u

    @cached_property
    def p(me) -> dl.Function:
        return me.derivatives.p

    @cached_property
    def m_Wh(me) -> dl.Function:
        return me.derivatives.m

    def m_Vh2(me) -> dl.Function:
        m_Vh2 = dl.Function(me.function_spaces.Vh2)
        m_Vh2.vector()[:] = me.function_spaces.Wh_to_Vh2_petsc(me.m_Wh.vector())
        return m_Vh2

    def m_Vh3(me) -> dl.Function:
        m_Vh3 = dl.Function(me.function_spaces.Vh3)
        m_Vh3.vector()[:] = me.function_spaces.Wh_to_Vh3_petsc(me.m_Wh.vector())
        return m_Vh3

    def mtrue_Vh2(me) -> dl.Function:
        mtrue_Vh2 = dl.Function(me.function_spaces.Vh2)
        mtrue_Vh2.vector()[:] = me.function_spaces.Wh_to_Vh2_petsc(me.mtrue_Wh.vector())
        return mtrue_Vh2

    def mtrue_Vh3(me) -> dl.Function:
        mtrue_Vh3 = dl.Function(me.function_spaces.Vh3)
        mtrue_Vh3.vector()[:] = me.function_spaces.Wh_to_Vh3_petsc(me.mtrue_Wh.vector())
        return mtrue_Vh3

    def velocity(me) -> dl.Function:
        return me.u.sub(0)

    def pressure(me) -> dl.Function:
        return me.u.sub(1)

    def velocity_true(me) -> dl.Function:
        return me.utrue.sub(0)

    def pressure_true(me) -> dl.Function:
        return me.utrue.sub(1)

    def velocity_obs(me) -> dl.Function:
        return me.uobs.sub(0)

    def pressure_obs(me) -> dl.Function:
        return me.uobs.sub(1)

    def generate_multiplicative_noise(me, noise_level: float) -> np.ndarray:
        return noise_level * np.random.randn(me.function_spaces.Zh.dim()) * np.abs(me.utrue.vector()[:])

    def update_noise(me, noise_vec: np.ndarray) -> None:
        me.uobs.vector()[:] = me.utrue.vector()[:] + noise_vec

    def noise_Zh_numpy(me) -> np.ndarray:
        return me.uobs.vector()[:] - me.utrue.vector()[:]

    def misfit_datanorm(me) -> float:
        norm1 = np.sqrt(2.0 * me.derivatives.misfit())
        norm2 = me.dataspace_norm(me.u.vector()[:])
        assert_le(np.abs(norm1 - norm2), 1e-8*(np.abs(norm1) + np.abs(norm2)))
        return norm1

    def dataspace_inner_product(me, p_Zh_numpy: np.ndarray, q_Zh_numpy: np.ndarray) -> float:
        p_Zh = dl.Function(me.function_spaces.Zh)
        p_Zh.vector()[:] = p_Zh_numpy
        q_Zh = dl.Function(me.function_spaces.Zh)
        q_Zh.vector()[:] = q_Zh_numpy
        p_velocity, p_pressure = dl.split(p_Zh)
        q_velocity, q_pressure = dl.split(q_Zh)

        normal = dl.FacetNormal(me.meshes.ice_mesh_3d)
        ds = dl.Measure("ds", domain=me.meshes.ice_mesh_3d, subdomain_data=me.meshes.boundary_markers)
        ds_top = ds(2)

        dataspace_inner_product_form = dl.inner(fenics_tangent(p_velocity, normal),
                                                fenics_tangent(q_velocity, normal)) * ds_top

        return dl.assemble(dataspace_inner_product_form)

    def dataspace_norm(me, p_Zh_numpy: np.ndarray) -> float:
        return np.sqrt(me.dataspace_inner_product(p_Zh_numpy, p_Zh_numpy))

    def noise_datanorm(me) -> float:
        return me.dataspace_norm(me.noise_Zh_numpy())
        # noise_Zh = dl.Function(me.function_spaces.Zh)
        # noise_Zh.vector()[:] = me.noise_Zh_numpy()
        # noise_velocity, noise_pressure = dl.split(noise_Zh)
        #
        # normal = dl.FacetNormal(me.meshes.ice_mesh_3d)
        # ds = dl.Measure("ds", domain=me.meshes.ice_mesh_3d, subdomain_data=me.meshes.boundary_markers)
        # ds_top = ds(2)
        #
        # noise_datanorm_form = dl.inner(fenics_tangent(noise_velocity, normal),
        #                                fenics_tangent(noise_velocity, normal)) * ds_top
        #
        # noise_datanorm = np.sqrt(dl.assemble(noise_datanorm_form))
        # return noise_datanorm


def fenics_tangent(vector_field, normal):
    return vector_field - dl.outer(normal, normal) * vector_field


def linear_stokes_inverse_problem_unregularized(
        mesh_type: str='fine', # mesh_type in {'coarse', 'medium', 'fine'}
        solver_type: str ='mumps', # solver_type in {'default', 'petsc', 'umfpack', 'superlu', 'mumps'}
        outflow_constant: float=1.0e6,
        grav: float=9.81,  # acceleration due to gravity
        rho: float= 910.0,  # volumetric mass density of ice
        stokes_A: float = 2.140373e-7,  # rheology constant. other option: 1.e-16
        smooth_strain: float = 1e-6, # rheology constant.
        true_parameter_options: typ.Dict[str, typ.Any]=None,
        initial_guess_options: typ.Dict[str, typ.Any]=None,
        noise_level: float=5e-2,
):
    meshes = make_stokes_meshes(mesh_type=mesh_type)
    function_spaces = make_stokes_function_spaces(
        meshes.ice_mesh_3d, meshes.basal_mesh_3d, meshes.basal_mesh_2d)

    m_Wh = dl.Function(function_spaces.Wh)  # Basal sliding friction
    u = dl.Function(function_spaces.Zh)  # Stokes state: (velocity, pressure)
    p = dl.Function(function_spaces.Zh)  # Stokes adjoint variable

    mtrue_Wh = dl.Function(function_spaces.Wh)  # True parameter
    utrue = dl.Function(function_spaces.Zh)  # True state
    uobs = dl.Function(function_spaces.Zh)  # observed state (true state plus noise)

    stokes_forcing = dl.Constant((0., 0., -rho * grav))

    ds = dl.Measure("ds", domain=meshes.ice_mesh_3d, subdomain_data=meshes.boundary_markers)
    ds = ds
    ds_base = ds(1)
    ds_top = ds(2)
    ds_lateral = ds(3)
    normal = dl.FacetNormal(meshes.ice_mesh_3d)
    # Strongly enforced Dirichlet conditions. The no outflow condition will be enforced weakly, via penalty parameter.
    bcs = []
    bcs0 = []

    # Define the Nonlinear Stokes varfs
    velocity, pressure = dl.split(u)
    strain_rate = dl.sym(dl.grad(velocity))
    normEu12 = 0.5 * dl.inner(strain_rate, strain_rate) + dl.Constant(smooth_strain)

    tangent_velocity = (velocity - dl.outer(normal, normal) * velocity)

    energy_t1 = dl.Constant(stokes_A) ** (-1.) * normEu12 * dl.dx
    energy_t2 = -dl.inner(stokes_forcing, velocity) * dl.dx
    energy_t3 = dl.Constant(.5) * dl.inner(dl.exp(m_Wh) * tangent_velocity, tangent_velocity) * ds_base
    energy_t4 = meshes.lam * dl.inner(velocity, normal) ** 2 * ds_base

    energy_t5 = dl.Constant(outflow_constant) * dl.inner(velocity, normal) ** 2 * ds_lateral  # Side outflow Robin BC

    energy = energy_t1 + energy_t2 + energy_t3 + energy_t4 + energy_t5
    energy_gradient = dl.derivative(energy, u, p)

    adjoint_velocity, adjoint_pressure = dl.split(p)
    div_constraint = dl.inner(-dl.div(velocity), adjoint_pressure) * dl.dx
    div_constraint_transpose = dl.inner(-dl.div(adjoint_velocity), pressure) * dl.dx

    forward_form_ff = energy_gradient + div_constraint + div_constraint_transpose  # A(u,v) + B(u,z) + B(v,p)

    # dummy_u1 = dl.Function(function_spaces.Zh)
    # dummy_u2 = dl.Function(function_spaces.Zh)
    #
    # dummy_velocity1, dummy_pressure1 = dl.split(dummy_u1)
    # dummy_velocity2, dummy_pressure2 = dl.split(dummy_u2)
    #
    # data_inner_product_form = dl.inner(Tang(dummy_velocity1, normal),
    #                                    Tang(dummy_velocity2, normal)) * ds_top
    #
    # def data_inner_product(u_petsc, v_petsc):
    #     dummy_u1.vector()[:] = u_petsc[:]
    #     dummy_u2.vector()[:] = v_petsc[:]
    #     return dl.assemble(data_inner_product_form)

    true_velocity, true_pressure = dl.split(utrue)
    observed_velocity, observed_pressure = dl.split(uobs)

    velocity_discrepancy = observed_velocity - velocity

    misfit_form = 0.5 * dl.inner(fenics_tangent(velocity_discrepancy, normal),
                                 fenics_tangent(velocity_discrepancy, normal)) * ds_top

    derivatives = DerivativesAtPoint(misfit_form, forward_form_ff, bcs,
                              m_Wh, u, p,
                              function_spaces.Vh2_to_Wh_numpy,
                              function_spaces.Wh_to_Vh2_numpy,
                                     solver_type=solver_type)

    true_parameter_options2 = {'which' : 'angel_peak'}
    if true_parameter_options is not None:
        true_parameter_options2.update(true_parameter_options)

    print('Solving forward problem with true parameter to get true observations')
    mtrue_Vh2 = load_stokes_parameter(function_spaces.Vh2, **true_parameter_options2)
    derivatives.update_parameter(mtrue_Vh2.vector()[:])
    derivatives.update_forward()
    utrue.vector()[:] = derivatives.u.vector()[:].copy()
    mtrue_Wh.vector()[:] = derivatives.m.vector()[:]

    initial_guess_options2 = {'which' : 'constant', 'Radius': meshes.Radius}
    if initial_guess_options is not None:
        initial_guess_options2.update(initial_guess_options)

    print('Solving forward problem with initial guess to reset')
    m0_Vh2 = load_stokes_parameter(function_spaces.Vh2, **initial_guess_options2)
    derivatives.update_parameter(m0_Vh2.vector()[:])

    UIP = LinearStokesInverseProblemUnregularized(
        meshes, function_spaces, derivatives, mtrue_Wh, utrue, uobs
    )

    UIP.update_noise(noise_level)
    noise_datanorm = UIP.noise_datanorm()
    print('noise_level=', noise_level, ', noise_datanorm=', noise_datanorm)

    return UIP


def load_stokes_parameter(
        Vh2: dl.FunctionSpace,
        which: str= 'angel_peak', # mtrue_type in {'angel_peak', 'aces_building', 'expression', 'constant'}
        expression: str = 'm0 - (m0 / 7.)*std::cos(2.*x[0]*pi/Radius)', # another choice: 'm0 - (m0 / 7.)*std::cos((x[0]*x[0]+x[1]*x[1])*pi/(Radius*Radius))'
        constant: float = 1.5 * 7.,
        Radius: float=None,
) -> dl.Function:
    if which.lower() == 'angel_peak' or which.lower() == 'aces_building':
        image_dir = localpsf_root / 'localpsf'

        if which.lower() == 'angel_peak':
            #         image_file = image_dir / 'angel_peak_badlands_cropped9.png' # good
            image_file = image_dir / 'angel_peak_badlands_cropped7.png'
        else:
            image_file = image_dir / 'aces_building.png'

        m_Vh2 = load_image_into_fenics(Vh2, image_file)
        minc = 9
        maxc = 12
        m_Vh2_vec = m_Vh2.vector()[:]
        m_Vh2_vec *= 3
        m_Vh2_vec += 9
        m_Vh2.vector()[:] = m_Vh2_vec
    elif which.lower() == 'expression':
        mtrue_expr = dl.Expression(expression, element=Vh2.ufl_element(),
                                   m0=constant, Radius=Radius)
        m_Vh2 = dl.interpolate(mtrue_expr, Vh2)
    elif which.lower() == 'constant':
        m_Vh2 = dl.interpolate(dl.Constant(constant), Vh2)
    else:
        raise RuntimeError('BAD mtrue_type')

    return m_Vh2


def check_stokes_gauss_newton_hessian(
        UIP: LinearStokesInverseProblemUnregularized,
        m0_Vh2: np.ndarray,
        dm1_Vh2: np.ndarray,
        dm2_Vh2: np.ndarray,
) -> float:
    old_m = UIP.derivatives.get_parameter()
    N = len(old_m)
    assert_equal(m0_Vh2.shape, (N,))
    assert_equal(dm1_Vh2.shape, (N,))
    assert_equal(dm2_Vh2.shape, (N,))

    UIP.derivatives.update_parameter(m0_Vh2)
    UIP.derivatives.update_forward()

    UIP.derivatives.update_z(dm1_Vh2)
    UIP.derivatives.update_incremental_forward()
    du_dm_1 = dl.Function(UIP.function_spaces.Zh)
    du_dm_1.vector()[:] = UIP.derivatives.du_dm_z.vector()[:].copy()

    UIP.derivatives.update_z(dm2_Vh2)
    UIP.derivatives.update_incremental_forward()
    du_dm_2 = dl.Function(UIP.function_spaces.Zh)
    du_dm_2.vector()[:] = UIP.derivatives.du_dm_z.vector()[:].copy()

    C1 = UIP.dataspace_inner_product(du_dm_1.vector(), du_dm_2.vector())

    Hgn_1 = UIP.derivatives.apply_gauss_newton_hessian(dm1_Vh2)
    C2 = np.dot(dm2_Vh2, Hgn_1)

    gauss_newton_error = np.abs(C1 - C2) / np.abs(C1)
    print('gauss_newton_error=', gauss_newton_error)

    UIP.derivatives.update_parameter(old_m)

    return gauss_newton_error


@dataclass(frozen=True)
class LinearStokesUniverse:
    unregularized_inverse_problem: LinearStokesInverseProblemUnregularized
    objective: InverseProblemObjective
    mass_lumps: np.ndarray
    bct: hpro.BlockClusterTree