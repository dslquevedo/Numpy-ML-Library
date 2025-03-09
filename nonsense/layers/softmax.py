from attrs import frozen
import numpy as np
from ..utils.array_types import Matrix
from ..utils.stable_math import exp # WARNING: EXP ALTERS INPUT IN-PLACE

""" Only implemented after Dense1D layers (for this project) """

@frozen
class Softmax:
    @staticmethod
    def forward(X: Matrix) -> Matrix:
        """ A = exp(Z - max(Z)) / sum(exp(Z - max(Z))"""
        X_max: Matrix = np.max(X, axis=1, keepdims=True)
        X_norm: Matrix = np.subtract(X, X_max)
        exp_X_norm: Matrix = exp(X_norm) # WARNING: EXP ALTERS INPUT IN-PLACE
        A: Matrix = np.divide(exp_X_norm, np.sum(exp_X_norm, axis=1), out=exp_X_norm)
        return A

    @staticmethod
    def backward(dA: Matrix, X: Matrix) -> Matrix:
        """ dZ = A * (dA - sum(A * dA, axis=-1))"""
        A: Matrix = Softmax.forward(X)
        sum_A_dA: Matrix = np.sum(np.multiply(A, dA), axis=1, keepdims=True)
        dX: Matrix = np.multiply(A, np.subtract(dA, sum_A_dA, out=dA), out=dA)
        if np.isnan(dX).any():
            raise ValueError("NaN in Softamx")
        return dX

if __name__ == "__main__":
    pass
    