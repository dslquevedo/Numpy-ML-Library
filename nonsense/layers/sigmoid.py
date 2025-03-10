import numpy as np
from attrs import frozen
from ..utils.array_type_aliases import Matrix
from ..utils.stable_math import exp # WARNING: EXP ALTERS INPUT IN-PLACE

""" Only implemented after Dense1D layers (for this project) """

@frozen
class Sigmoid:
    @staticmethod
    def forward(X: Matrix) -> Matrix:
        """ A = (1 + e^-X)^(-1) """
        exp_negX: Matrix = exp(np.negative(X)) # WARNING: EXP ALTERS INPUT IN-PLACE
        plus_one: Matrix = np.add(np.float32(1), exp_negX, out=exp_negX)
        A: Matrix = np.reciprocal(plus_one, out=plus_one)
        return A

    @staticmethod
    def backward(dA: Matrix, X: Matrix) -> Matrix:
        """ dZ = dA if X > 0 else 0 """
        return np.where(X <= 0, np.float32(0.0), dA) 

if __name__ == "__main__":
    pass