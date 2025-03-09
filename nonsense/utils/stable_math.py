import numpy as np
from array_types import Tensor

def exp(X: Tensor) -> Tensor:
    BIG = np.float32(3.4+38)
    numpy_exp = np.exp(X)
    # WARNING THIS ALTERS IN PLACE THE INPUT X TO OBTAIN THE OUTPUT (COPY=FALSE)
    stabler_exp = np.nan_to_num(numpy_exp, posinf=BIG, negin=-BIG, copy=False) #type: ignore
    return stabler_exp

def ewa(X: Tensor, X_bar: Tensor, beta: np.float32) -> Tensor:
    """ Exponential Weighted Average: X̄ = β ∙ X̄ + (1 - β) ∙ X """
    return np.add(beta * X_bar, (np.float32(1.0) - beta) * X, out=X_bar)

if __name__ == "__main__":
    pass