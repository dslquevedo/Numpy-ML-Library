import numpy as np
from attrs import define, field
from typing import ClassVar
from .base_optimizer import Optimizer, _increment_time
from ..utils.array_type_aliases import Tensor
from ..utils.stable_math import ewa


@define  
class _Momentum(Optimizer):
    beta: ClassVar[np.float32] = np.float32(0.9)
    eps: ClassVar[np.float32] = np.float32(1e-8)

    avg_V: Tensor = field(init=False)
    
    def __attrs_post_init__(self) -> None:
        self.avg_V = np.zeros(self._param_shape, dtype=np.float32)


@define
class Momentum(_Momentum):
    @_increment_time
    def update(self, param: Tensor, grad: Tensor) -> None:
        """ θ ← θ - a ∙ ∇θU(θ) """
        self.avg_V = ewa(grad, self.avg_V, self.beta)
        param -= np.multiply(self.lr, self.avg_V)
        return


@define
class Momentum_Ascent(_Momentum):
    @_increment_time
    def update(self, param: Tensor, grad: Tensor) -> None:
        """ θ ← θ + a ∙ ∇θU(θ) """
        self.avg_V = ewa(grad, self.avg_V, self.beta)
        param += np.multiply(self.lr, self.avg_V)
        return


if __name__ == "__main__":
    pass