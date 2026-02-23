"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    """$f(x, y) = x * y$"""
    return x * y
# TODO fill in the remaining mathematical operators. Use mul provided above as an example.
# - id
def id(x: float) -> float:
    """$f(x) = x$"""
    return x
# - add
def add(x: float, y: float) -> float:
    """$f(x, y) = x + y$"""
    return x + y
# - neg
def neg(x: float) -> float:
    """$f(x) = -x$"""
    return -x
# - lt
def lt(x: float, y: float) -> float:
    """$f(x, y) = 1.0 if x < y else 0.0$"""
    return 1.0 if x < y else 0.0
# - eq
def eq(x: float, y: float) -> float:
    """$f(x, y) = 1.0 if x == y else 0.0$"""
    return 1.0 if x == y else 0.0
# - max
def max(x: float, y: float) -> float:
    """$f(x, y) = x if x > y else y$"""
    return x if x > y else y
# - is_close
def is_close(x: float, y: float) -> float:
    """$f(x, y) = |x - y| < 1e-2$"""
    return 1.0 if abs(x - y) < 1e-2 else 0.0
# - sigmoid
def sigmoid(x: float) -> float:
    """$f(x) = \frac{1.0}{1.0 + e^{-x}}$"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))
# - relu
def relu(x: float) -> float:
    """ $f(x) = x$ if x > 0 else 0.0$"""
    return x if x > 0 else 0.0
# - log
def log(x: float) -> float:
    """$f(x) = log(x)$"""
    return math.log(x)
# - exp
def exp(x: float) -> float:
    """$f(x) = e^x$"""
    return math.exp(x)
# - log_back
def log_back(grad: float, x: float) -> float:
    """Gradient of logarithm function."""
    return grad / x
# - inv
def inv(x: float) -> float:
    """$f(x) = \frac{1.0}{x}$"""
    return 1.0 / x
# - inv_back
def inv_back(grad: float, x: float) -> float:
    """Gradient of inverse function."""
    return -grad / (x * x)
# - relu_back
def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0.0
    



# TODO: Implement for Task 0.1.


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
def map(fn: Callable[[float], float], ls: Iterable[float]) -> list[float]:
    return [fn(x) for x in ls]
# - zipWith 
def zipWith(fn: Callable[[float, float], float], 
            ls1: Iterable[float], 
            ls2: Iterable[float]) -> list[float]:
    return [fn(x, y) for x, y in zip(ls1, ls2)]
# - reduce
def reduce(fn: Callable[[float, float], float], 
           ls: Iterable[float]) -> float:
    ls = list(ls)
    if len(ls) == 0:
        return 0.0

    result = ls[0]
    for x in ls[1:]:
        result = fn(result, x)

    return result
# Use these to implement
# - negList : negate all elemnts in a list using map
def negList(ls: Iterable[float]) -> list[float]:
    return map(neg, ls)
# - addLists : add corresponding elements from two lists using zipWith
def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> list[float]:
    return zipWith(add, ls1, ls2)
# - sum: sum all elements in a list using reduce
def sum(ls: Iterable[float]) -> float:
    return reduce(add, ls)
# - prod: calculate the product of all elements in a list using reduce
def prod(ls: Iterable[float]) -> float:
    return reduce(mul, ls)


# TODO: Implement for Task 0.3.
