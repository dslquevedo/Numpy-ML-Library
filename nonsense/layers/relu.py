from attrs import frozen
import numpy as np
from ..utils.array_types import Tensor


@frozen
class ReLU:
    @staticmethod
    def forward(X: Tensor) -> Tensor:
        """ A = max(0, X) """
        return np.where(X <= 0, np.float32(0), X)

    @staticmethod
    def backward(dA: Tensor, X: Tensor) -> Tensor:
        """ dX = dA if X > 0 else 0 """
        return np.where (X <= 0, np.float32(0), dA)

if __name__ == "__main__":
    pass