from typing import TypeAlias, Any, Tuple
from nptyping import NDArray, Float32, Shape

""" nptyping is not numpy.typing
https://github.com/ramonhagenaars/ing ping/tree/master
NDArray[Shape["..."], dtype]

A scalar is a type of rank 0 tensor
Rank 0 tensors currently not type checkable 

##Expect in MOST cases except where specified:
# Vector
Axis 0: batch
Axis 1: features

# Matrix
Axis 0: batch
Axis 1: features

# Rank3Tensor
Axis 0: batch
Axis 1: depth
Axis 2: length

# Rank4Tensor
Axis 0: batch
Axis 1: depth
Axis 2: width
Axis 3: height

# Rank5Tensor Only present in 2D CONV
Axis 0: batch
Axis 1: Kernel broadcasting "port" 
Axis 2: depth
Axis 3: width
Axis 4: height
"""

__all__ = [
    "Tensor",
    "Vector",
    "Matrix",
    "Rank3Tensor",
    "Rank4Tensor",
    "Rank5Tensor",
    "TensorShape",
    "VectorShape",
    "MatrixShape",
    "Rank3Shape",
    "Rank4Shape",
    "Rank5Shape",
]

# n-Rank Tensor (including scalar)
TensorShape: TypeAlias = Tuple[int, ...]
VectorShape: TypeAlias = Tuple[int]
MatrixShape: TypeAlias = Tuple[int, int]
Rank3Shape: TypeAlias = Tuple[int, int, int]
Rank4Shape: TypeAlias = Tuple[int, int, int, int]
Rank5Shape: TypeAlias = Tuple[int, int, int, int, int]

Tensor: TypeAlias = NDArray[Any, Float32]
Vector: TypeAlias = NDArray[Shape["*"], Float32]
Matrix: TypeAlias = NDArray[Shape["*, *"], Float32]
Rank3Tensor: TypeAlias = NDArray[Shape["*, *, *"], Float32]
Rank4Tensor: TypeAlias = NDArray[Shape["*, *, *, *"], Float32]
Rank5Tensor: TypeAlias = NDArray[Shape["*, *, *, *, *"], Float32]


if __name__ == "__main__":
    pass
