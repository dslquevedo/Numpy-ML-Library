import numpy as np
from attrs import define, field, setters
from typing import Tuple, Callable, Any
from ..utils.array_type_aliases import Tensor

def _increment_time(func: Callable) -> Callable:
    """ Helper decorator to increment time step by 1"""
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        self.t += np.float32(1.0)
        return func(self, *args, **kwargs)
    return wrapper


@define
class Optimizer:
    _param_shape: Tuple[int, ...] = field(on_setattr=setters.frozen)
    lr: float = field(init=False, on_setattr=setters.frozen)
    t: int = field(init=False, default=0)

    def set_learning_rate(self, lr) -> None:
        object.__setattr__(self, "lr", lr)
        return None

    @_increment_time
    def update(self, param: Tensor, grad: Tensor) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    pass
