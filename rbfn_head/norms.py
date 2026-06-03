from torch import Tensor
from torch import linalg as LA


def manhattan(x: Tensor) -> Tensor:
    return LA.norm(x, ord=1, dim=-1)


def euclidian(x: Tensor) -> Tensor:
    return LA.norm(x, ord=2, dim=-1)
