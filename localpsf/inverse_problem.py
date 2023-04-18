import numpy as np
import typing as typ
from dataclasses import dataclass
from functools import cached_property

from .assertion_helpers import *

@dataclass
class InverseProblem:
    misfit: typ.Any
    regularization: typ.Any
    regularization_parameter: float

    def __post_init__(me):
        f1: typ.Callable[[np.ndarray, float], float]                    = me.regularization.cost # Jr(m) = cost(m, a_reg)
        f2: typ.Callable[[np.ndarray, float], np.ndarray]               = me.regularization.gradient # gr(m) = gradient(m, a_reg)
        f3: typ.Callable[[np.ndarray, np.ndarray, float], np.ndarray]   = me.regularization.apply_hessian # Hr(m)p = apply_hessian(p, m, a_reg)
        f4: typ.Callable[[np.ndarray, np.ndarray, float], np.ndarray]   = me.regularization.solve_hessian  # invHr(m)p = solve_hessian(p, m, a_reg)

        f5: typ.Callable[[], np.ndarray]            = me.misfit.get_parameter # m = get_parameter()
        f6: typ.Callable[[np.ndarray], None]        = me.misfit.update_parameter # update_parameter(new_m)
        f7: typ.Callable[[], float]                 = me.misfit.misfit # Jd(m) = misfit()
        f8: typ.Callable[[], np.ndarray]            = me.misfit.gradient # gd(m) = gradient()
        f9: typ.Callable[[np.ndarray], np.ndarray]  = me.misfit.apply_hessian # Hd(m)p = apply_hessian(p)
        f10: typ.Callable[[np.ndarray], np.ndarray] = me.misfit.apply_gauss_newton_hessian # Hdgn(m)p = apply_gauss_newton_hessian(p)

    @cached_property
    def N(me):
        return len(me.misfit.get_parameter())

    def get_optimization_variable(me) -> np.ndarray:
        m: np.ndarray = me.misfit.get_parameter()
        assert_equal(m.shape, (me.N,))
        return m

    def set_optimization_variable(me, new_m: np.ndarray) -> None:
        assert_equal(new_m.shape, (me.N,))
        me.misfit.update_parameter(new_m)

    def update_regularization_parameter(me, new_a_reg) -> None:
        assert_gt(new_a_reg, 0.0)
        me.regularization_parameter = new_a_reg

    def cost_triple(me) -> typ.Tuple[float, float, float]:
        Jd: float = me.misfit.misfit()
        Jr: float = me.regularization.cost(me.get_optimization_variable(), me.regularization_parameter)
        J: float = Jd + Jr
        return J, Jd, Jr

    def gradient_triple(me) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gd = me.misfit.gradient()
        assert_equal(gd.shape, (me.N,))
        gr = me.regularization.gradient(me.get_optimization_variable(), me.regularization_parameter)
        assert_equal(gr.shape, (me.N,))
        g = gd + gr
        return g, gd, gr

    def apply_hessian_triple(me, p: np.ndarray) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert_equal(p.shape, (me.N,))
        Hd_p = me.misfit.apply_hessian(p)
        assert_equal(Hd_p.shape, (me.N,))
        Hr_p = me.regularization.apply_hessian(p, me.get_optimization_variable(), me.regularization_parameter)
        assert_equal(Hr_p.shape, (me.N,))
        H_p = Hd_p + Hr_p
        return H_p, Hd_p, Hr_p

    def apply_gauss_newton_hessian_triple(me, p: np.ndarray) -> typ.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert_equal(p.shape, (me.N,))
        Hdgn_p = me.misfit.apply_gauss_newton_hessian(p)
        assert_equal(Hdgn_p.shape, (me.N,))
        Hr_p = me.regularization.apply_hessian(p, me.get_optimization_variable(), me.regularization_parameter)
        assert_equal(Hr_p.shape, (me.N,))
        Hgn_p = Hdgn_p + Hr_p
        return Hgn_p, Hdgn_p, Hr_p

    def cost(me) -> float:
        return me.cost_triple()[0]

    def gradient(me) -> np.ndarray:
        return me.gradient_triple()[0]

    def apply_hessian(me, p: np.ndarray) -> np.ndarray:
        return me.apply_hessian_triple(p)[0]

    def apply_gauss_newton_hessian(me, p: np.ndarray) -> np.ndarray:
        return me.apply_gauss_newton_hessian_triple(p)[0]
