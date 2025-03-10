import numpy as np
from .array_type_aliases import Tensor

def exp(X: Tensor) -> Tensor:
    """ Numpy exponential function: Tradeoff NaN for accuracy by clamping 

    Overflow Errors: https://numpy.org/doc/stable/user/basics.types.html
    IEEE 754 -- by default wraps to inf/-inf for overflow 
    """
    BIG = np.float32(3.4+38)
    numpy_exp = np.exp(X)
    # WARNING THIS ALTERS IN PLACE THE INPUT X TO OBTAIN THE OUTPUT (COPY=FALSE)
    stabler_exp = np.nan_to_num(
        numpy_exp, 
        posinf=BIG, # 
        negin=-BIG, # Clamp extremely large negative values to 3.4+38
        nan=np.float32(0.0), # Clamp extremely small values to 0
        copy=False
    ) #type: ignore
    return stabler_exp

def ewa(X: Tensor, X_bar: Tensor, beta: np.float32) -> Tensor:
    """ Exponential Weighted Average: X̄ = β ∙ X̄ + (1 - β) ∙ X """
    return np.add(beta * X_bar, (np.float32(1.0) - beta) * X, out=X_bar)

if __name__ == "__main__":
    pass