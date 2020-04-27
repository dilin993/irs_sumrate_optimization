from rician_channel import RicianChannel
import util
import numpy as np

class Node:

    def __init__(self, antnum, pos):
        (p1, p2, p3) = pos
        self.pos = np.array([p1,p2,p3]).T
        self.antnum = antnum


class BS(Node):

    def __init__(self, antnum, pos):
        Node.__init__(self, antnum, pos)


class UE(Node):

    def __init__(self, antnum, pos):
        Node.__init__(self, antnum, pos)


class IRS(Node):

    def __init__(self, antnum, pos):
        Node.__init__(self, antnum, pos)