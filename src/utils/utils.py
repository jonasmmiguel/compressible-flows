import numpy as np


def linint(x: float, p1: tuple, p2: tuple) -> float:
    """
    linear interpolation
    """
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]

    return y1 + (x-x1) * (y2-y1) / (x2-x1)