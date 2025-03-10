import numpy as np
from nptyping import InvalidShapeError

def validate_rank(arr: np.ndarray, rank: int) -> None:
    if len(arr.shape) != rank:
        raise InvalidShapeError(f"Expected a rank {rank} array, received: rank {len(arr.shape)}")
    return

def validate_rank_1(arr: np.ndarray) -> None:
    if len(arr.shape) != 1:
        raise InvalidShapeError(f"Expected a rank 1 array, received: rank {len(arr.shape)}")
    return

def validate_rank_2(arr: np.ndarray) -> None:
    if len(arr.shape) != 2:
        raise InvalidShapeError(f"Expected a rank 2 array, received: rank {len(arr.shape)}")
    return

def validate_rank_3(arr: np.ndarray) -> None:
    if len(arr.shape) != 3:
        raise InvalidShapeError(f"Expected a rank 3 array, received: rank {len(arr.shape)}")
    return

def validate_rank_4(arr: np.ndarray) -> None:
    if len(arr.shape) != 4:
        raise InvalidShapeError(f"Expected a rank 4 array, received: rank {len(arr.shape)}")
    return

def validate_rank_5(arr: np.ndarray) -> None:
    if len(arr.shape) != 5:
        raise InvalidShapeError(f"Expected a rank 5 array, received: rank {len(arr.shape)}")
    return


if __name__ == "__main__":
    arr1 = np.array([1, 2, 3])              # Rank 1
    arr2 = np.array([[1, 2], [3, 4]])       # Rank 2
    arr3 = np.array([[[1, 2], [3, 4]]])     # Rank 3

    validate_rank_1(arr1)
    print("Rank 1 validated")

    validate_rank_2(arr2)
    print("Rank 2 validated")

    try:
        validate_rank_1(arr2)
    except InvalidShapeError as e:
        print("Error demonstration:", e)

    try:
        validate_rank_3(arr1)
    except InvalidShapeError as e:
        print("Error demonstration:", e)