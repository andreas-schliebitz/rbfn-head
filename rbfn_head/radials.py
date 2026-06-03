from torch import Tensor


def linear(r: Tensor) -> Tensor:
    return r


# Strictly positive definite functions


def gaussian(r: Tensor) -> Tensor:
    return (-r.pow(2)).exp()


def inverse_quadratic(r: Tensor) -> Tensor:
    return 1 / (1 + r.pow(2))


def inverse_multiquadric(r: Tensor) -> Tensor:
    return 1 / (1 + r.pow(2)).sqrt()


# Not strictly positive definite functions


def multiquadric(r: Tensor) -> Tensor:
    return (1 + r.pow(2)).sqrt()


def rth(r: Tensor) -> Tensor:
    return r * r.tanh()


def tps(r: Tensor) -> Tensor:
    return r.pow(2) * r.log()
