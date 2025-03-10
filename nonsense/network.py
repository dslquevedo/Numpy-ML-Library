from __future__ import annotations
import numpy as np
from numpy.random import Generator
from nptyping import InvalidDTypeError, InvalidShapeError, NDArray
from attrs import define, field, setters
from attrs.validators import not_, in_
from typing import Type, Union, Dict, List, Optional, ClassVar, Set, Any
from .optimizers import Optimizer, Adam
from .layers.initializers import Initializer, Initialization
from .utils.array_type_aliases import Tensor, TensorShape

def _mutate_immutable_sneaky(*args):
    for args in args


@define
class Parameter:
    value: Tensor = field()

    name: str = field(on_setattr=setters.frozen)
    optimizable: bool = field(on_setattr=setters.frozen)
    # Should i add a "As is" or "None" optimizer for the type hints?
    optimizer: Optimizer = field(init=False, on_setattr=setters.frozen)
    frozen: bool = field(on_setattr=setters.frozen)
    layer: Layer = field(on_setattr=setters.frozen)
    initialization: Initialization = field(default=Initializer.identity)
    
    

@define
class Layer:
    cache: Dict[str, Union[Tensor, TensorShape]] = field(init=False, factory=dict, repr=False)
    frozen: bool = field(init=False, default=False)
    
    # Sequential Architecture = Basic linked list
    next_layers: None | Layer = field(init=False, default=None)
    prev_layers: None | Layer = field(init=False, default=None)

    params: Dict[str, Parameter] = field(init=False, factory=dict, repr=False)


@define
class _Network:
    _registered_names: ClassVar[Set[str]]  = set()
    name: str = field()
    rng: Generator = field(init=False, on_setattr=setters.frozen)
    lr: float = field(init=False)
    optimizer: Type[Optimizer] = field(init=False, default=Adam)

    @name.validator #type: ignore
    def _validate_name(self, attribute: Any, value: str) -> None:
        if value in self._registered_names:
            raise ValueError(f"Name '{value}' already taken")
    
    def __attrs_post_init__(self) -> None:
        self._registered_names.add(self.name)

    def set_optimizer(self, optimizer: Type[Optimizer]) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    pass
     

