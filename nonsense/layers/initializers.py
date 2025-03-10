"""  
REFERENCES
(1) Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training 
    deep feedforward neural networks. Université de Montréal. 
    https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

(2) He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers: 
    Surpassing Human-Level Performance on ImageNet Classification. Microsoft. 
    https://arxiv.org/pdf/1502.01852
"""
import numpy as np
from numpy.random import Generator
from attrs import define, field
from typing import ClassVar, Type, TypeAlias
from collections.abc import Callable
from utils.array_type_aliases import Tensor


@define
class Initializer:
    """ A Collection of initialization methods that will an array with values. 
    
    WARNING -> If mounting pre-optimized weights, information will be lost.
    """
    rng: ClassVar[Generator] = field()

    @classmethod
    def identity(cls, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
         return np.asarray(arr, dtype=np.float32)

    @classmethod
    def zeros(cls, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
        arr.fill(np.float32(0.0))
        return arr

    @classmethod
    def ones(cls, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
        arr.fill(np.float32(1.0))
        return arr

    @classmethod
    def gaussian(cls, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
        cls.rng.standard_normal(dtype=np.float32, out=arr)
        return arr

    @classmethod
    def xavier_uniform(cls, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
        std: np.float32 = np.float32((2.0 / (fan_in + fan_out)) ** 0.5)
        limit: np.float32 = np.float32((3.0 ** 0.5) * std)
        arr[:] = cls.rng.uniform(low=-limit, high=limit)
        return np.asarray(arr, dtype=np.float32)

    @classmethod
    def xavier_normal(cls, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
        std: np.float32 = np.float32((2.0 / (fan_in + fan_out)) ** 0.5)
        cls.rng.standard_normal(dtype=np.float32, out=arr)
        return np.multiply(arr, std, dtype=np.float32)

    @classmethod
    def kaiming_uniform(cls, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
        std: np.float32 = np.float32(fan_in ** (-0.5))
        limit: np.float32 = np.float32((3.0 ** 0.5) * std)
        arr[:] = cls.rng.uniform(low=-limit, high=limit)
        return np.asarray(arr, dtype=np.float32)

    @classmethod
    def kaiming_normal(cls, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
        std: np.float32 = np.float32(fan_in ** (-0.5))
        cls.rng.standard_normal(dtype=np.float32, out=arr)
        return np.multiply(arr, std, dtype=np.float32)


Initialization: TypeAlias = Callable[
    [Type[Initializer], np.ndarray, int, int], 
    Tensor
]


if __name__ == "__main__":
    pass









# @define
# class Identity(_Initializer):
#     """ Returns pre-specified weights, converting to dtype32 a different dtype """
#     def initialization(self, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
#         return np.asarray(arr, dtype=np.float32)


# @define
# class Zeros(_Initializer):
#     def initialization(self, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
#         arr.fill(np.float32(0.0))
#         return arr


# @define
# class Ones(_Initializer):
#     def initialization(self, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
#         arr.fill(np.float32(1.0))
#         return arr


# @define
# class Gaussian(_Initializer):
#     def initialization(self, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
#         self.rng.standard_normal(dtype=np.float32, out=arr)
#         return arr


# @define
# class XavierUniform(_Initializer):
#     def initialization(self, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
#         std: np.float32 = np.float32((2.0 / (fan_in + fan_out)) ** 0.5)
#         limit: np.float32 = np.float32((3.0 ** 0.5) * std)
#         arr[:] = _WeightsInitializers.rng.uniform(low=-limit, high=limit)
#         return np.asarray(arr, dtype=np.float32)


# @define
# class XavierNormal(_Initializer):
#     def intialize(self, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
#         std: np.float32 = np.float32((2.0 / (fan_in + fan_out)) ** 0.5)
#         self.rng.standard_normal(dtype=np.float32, out=arr)
#         return np.multiply(arr, std, dtype=np.float32)


# @define
# class KaimingUniform(_Initializer):
#     def intiialize(self, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
#         std: np.float32 = np.float32(fan_in ** (-0.5))
#         limit: np.float32 = np.float32((3.0 ** 0.5) * std)
#         arr[:] = _Initializer.rng.uniform(low=-limit, high=limit)
#         return np.asarray(arr, dtype=np.float32)


# @define
# class KaimingNormal(_Initializer):
#     def initialization(self, arr: np.ndarray, fan_in: int, fan_out: int) -> Tensor:
#         std: np.float32 = np.float32(fan_in ** (-0.5))
#         _Initializer.rng.standard_normal(dtype=np.float32, out=arr)
#         return np.multiply(arr, std, dtype=np.float32)





        

    


