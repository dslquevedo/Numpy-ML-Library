import numpy as np
from attrs import define, field, setters, validators
from typing import Callable
from network import Layer, Parameter
from .initializers import Initialization
from utils.array_type_aliases import Matrix
from utils.validate_dtype import validate_float32


@define
class _DenseND(Layer):
    nodes: int = field(validator=validators.ge(1), on_setattr=setters.frozen)
    bias_enabled: bool = field(default=True, on_setattr=setters.frozen)


@define
class Dense(_DenseND):
    def _initialize(self, X: Matrix) -> None:
        validate_float32(X)

        fan_in: int = X.shape[1]
        fan_out: int = self.nodes

        weights_shape = (fan_in, self.nodes)
        weights = np.empty(weights_shape, dtype=np.float32)
        weights = Parameter(weights, )

        