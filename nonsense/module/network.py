from __future__ import annotations
import numpy as np
from numpy.random import Generator
from nptyping import InvalidDTypeError, InvalidShapeError, NDArray
from attrs import define, field, setters
from attrs.validators import not_, in_
from typing import Type, Union, Dict, List, Optional, ClassVar, Set, Any
from nonsense.optimizers import Optimizer, Adam
from nonsense.layers.initializers import Initializer, Initialization
from nonsense.utils.array_type_aliases import Tensor, TensorShape

# def _mutate_immutable_sneaky(*args):
#     for args in args


@define
class Parameter:
    name: str = field(on_setattr=setters.frozen)
    value: Tensor = field()
    optimizable: bool = field(on_setattr=setters.frozen)
    optimizer: Optimizer = field(init=False, on_setattr=setters.frozen)  # Should i add a "As is" or "None" optimizer for the type hints?
    frozen: bool = field(on_setattr=setters.frozen)
    initialization: Initialization = field(default=Initializer.identity)
    layer: Layer = field(init=False, on_setattr=setters.frozen)

    def __attrs_post_init__(self) -> None:
        self.layer.params[self.name] = self 
        return
    

@define
class Layer:
    cache: Dict[str, Union[Tensor, TensorShape]] = field(init=False, factory=dict, repr=False)
    frozen: bool = field(init=False, default=False)
    # Sequential Architecture = Basic linked list
    next_layers: None | Layer = field(init=False, default=None)
    prev_layers: None | Layer = field(init=False, default=None)
    params: Dict[str, Parameter] = field(init=False, factory=dict, repr=False)


@define
class Node:
    cache: Dict[str, Union[Tensor, TensorShape]] = field(init=False, factory=dict, repr=False)
    frozen: bool = field(init=False, default=None)
    fw_node: None | Node = field(init=False, default=None)
    bw_

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
     

