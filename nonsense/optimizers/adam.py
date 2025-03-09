from attrs import define, field
import numpy as np
from typing import ClassVar
from ..utils.stable_math import ewa
from ..utils.array_types import Tensor
from .base_optimizer import Optimizer, _increment_time


@define
class _Adam(Optimizer):
    beta1: ClassVar[np.float32] = np.float32(0.9)
    beta2: ClassVar[np.float32] = np.float32(0.999)
    eps: ClassVar[np.float32] = np.float32(1e-8)

    avg_V: Tensor = field(init=False)
    avg_S: Tensor = field(init=False)

    def __attrs_post_init__(self) -> None:
        self.avg_V = np.zeros(self._param_shape, dtype=np.float32)
        self.avg_S = np.zeros(self._param_shape, dtype=np.float32)
        return

    def _update_moments(self, grad: Tensor) -> None:
        """ Imporve docstring
        
        V̄ ← [β₁∙V̄ + (1 - β₁) ∙ ∇U(θ)] / (1 - β₁ᵗ)
        S̄ ← [β₂∙S̄ + (1 - β₂) ∙ ∇U(θ)²] / (1 - β₂ᵗ)
        """
        self.avg_V: Tensor = ewa(grad, self.avg_V, self.beta1)
        self.avg_S: Tensor = ewa(np.square(grad, dtype=np.float32), self.avg_S, self.beta2)
        return

    
@define
class Adam(_Adam):
    @_increment_time
    def update(self, param: Tensor, grad: Tensor) -> None:
        """ θ ← θ - a ∙ ∇U(θ) """
        self._update_moments(grad)
        V_corrected = self.avg_V / (1 - self.beta1**self.t)
        S_corrected = self.avg_S / (1 - self.beta2**self.t)
        param -= self.lr*V_corrected / np.sqrt(S_corrected + self.eps)
        return


@define
class Adam_Ascent(_Adam):
    @_increment_time
    def update(self, param: Tensor, grad: Tensor) -> None:
        """ θ ← θ + a ∙ ∇U(θ) """
        self._update_moments(grad)
        V_corrected = self.avg_V / (1 - self.beta1**self.t)
        S_corrected = self.avg_S / (1 - self.beta2**self.t)
        param += self.lr*V_corrected / np.sqrt(S_corrected + self.eps)
        return


if __name__ == "__main__":
    pass