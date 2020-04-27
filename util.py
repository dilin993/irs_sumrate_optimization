import numpy as np
from nodes import Node


def db2lin(a):
    return np.power(10,a/10)


def distance(node1, node2):
    return np.linalg.norm(node1.pos - node2.pos)


def lin2db(a):
    return 10 * np.log10(a)


def lin2dbm(a):
    return lin2db(a) + 30
