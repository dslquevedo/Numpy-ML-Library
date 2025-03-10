import numpy as np
from attrs import define, field, setters
from utils.array_type_aliases import Tensor
from network import Layer, _Network


@define 
class Sequential(_Network):
    head: Optional[Layer] = field(default=None)
    tail: Optional[Layer] = field(default=None)

    def __attrs_post_init__(self) -> None:
        