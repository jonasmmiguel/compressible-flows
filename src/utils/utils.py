import numpy as np


def linint(x: float, point1: tuple, point2: tuple) -> float:
    """
    linear interpolation
    """
    x1 = point1[0]
    y1 = point1[1]

    x2 = point2[0]
    y2 = point2[1]

    return y1 + (x-x1) * (y2-y1) / (x2-x1)


def get_vapor_property(x, qf, qg):
    return linint(x, (0.0, qf), (1.0, qg))


def get_quality(q, qf, qg):
    return linint(q, (qf, 0.0), (qg, 1.0))