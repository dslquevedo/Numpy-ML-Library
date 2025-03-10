import numpy as np
from attrs import define
from .base_optimizer import Optimizer, _increment_time
from ..utils.array_type_aliases import Tensor


@define
class _Stochastic_Gradient(Optimizer):
    pass


@define
class SGD(_Stochastic_Gradient):
    """ Stochastic Gradient Descent """
    @_increment_time
    def update(self, param: Tensor, grad: Tensor) -> None:
        """ θ ← θ - a ∙ ∇θU(θ) """
        param -= np.multiply(self.lr, grad)
        return

@define
class SGA(_Stochastic_Gradient):
    """ Stochastic Gradient Ascent """
    @_increment_time
    def update(self, param: Tensor, grad: Tensor) -> None:
        """ θ ← θ + a ∙ ∇θU(θ) """
        param += np.multiply(self.lr, grad)
        return