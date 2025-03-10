import numpy as np
from nptyping import InvalidDTypeError

__all__ = [
    "validate_float32",
]

def validate_float32(arr: np.ndarray):
    """ A togglable method (by network classes) for checking float32.
    Mostly for troubleshooting. Validation preferred over coercion. 
    """
    if arr.dtype != np.float32:
        raise InvalidDTypeError(f"Expected array dtype float32, got {arr.dtype}")
    return

if __name__ == "__main__":
    pass