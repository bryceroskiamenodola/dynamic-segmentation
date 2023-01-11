from __future__ import annotations
from typing import Any
import math
def join(s1: str, s2: str):
    if s1[-1] == '"':
        return s1[:-1]+s2+'"'
    else:
        return s1+s2
def decay(iteration:int,time_constant:int) -> float:
    return math.exp(-iteration/time_constant)
def step(iteration:int,time_constant:int) -> float:
    return (1.0 - (float(iteration)/time_constant))
def constant(iteration:int,time_constant:int) -> float:
    return float(1.0)