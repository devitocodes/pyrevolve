import pyrevolve as pr
import numpy as np
from collections.abc import Mapping


class Symbol(object):
    def __init__(self, data_dim):
        self.__storage = np.zeros(data_dim)
        self.__data_dim = data_dim

    @property
    def data(self):
        return self.__storage

    @data.setter
    def data(self, value):
        if(isinstance(value, np.ndarray)):
            self.__storage = value
        else:
            raise Exception("Symbol data must be a numpy array.")

    @property
    def size(self):
        return self.__storage.size


class ForwardOperator(pr.Operator):
    def __init__(self, u, m):
        self.u = u
        self.m = m

    def apply(self, t_start, t_end):
        # print((">"*(t_end-t_start)).rjust(t_end))
        for i in range(t_start, t_end):
            u.data = u.data + m.data


class ReverseOperator(pr.Operator):
    def __init__(self, u, m, v):
        self.u = u
        self.v = v
        self.m = m

    def apply(self, t_start, t_end):
        # print(("<"*(t_end-t_start)).rjust(t_end))
        for i in range(t_end, t_start, -1):
            v.data = v.data + m.data


class MyCheckpoint(pr.Checkpoint):
    """Holds a list of symbol objects that hold data."""

    def __init__(self, symbols):
        """Intialise a checkpoint object. Upon initialisation, a checkpoint
        stores only a reference to the symbols that are passed into it.
        The symbols must be passed as a mapping symbolname->symbolobject."""

        if(isinstance(symbols, Mapping)):
            self.symbols = symbols
        else:
            raise Exception("Symbols must be a Mapping, for example a \
                              dictionary.")

    def get_data_location(self, timestep):
        return [x.data for x in list(self.symbols.values())]

    def get_data(self, timestep):
        return [x.data for x in self.symbols.values()]

    @property
    def size(self):
        """The memory consumption of the data contained in this checkpoint."""
        size = 0
        for i in self.symbols:
            size = size+self.symbols[i].size
        return size

    @property
    def dtype(self):
        return np.float32


nSteps = 30
u = Symbol((4))
m = Symbol((4))
m.data = np.ones((4))
checkpoint = MyCheckpoint({"u": u, "m": m})
v = Symbol((1))
fwdo = ForwardOperator(u, m)
revo = ReverseOperator(u, m, v)
wrp = pr.Revolver(checkpoint, fwdo, revo, None, nSteps)
wrp.apply_forward()
print("u=%s" % u.data)
wrp.apply_reverse()
print("v=%s" % v.data)
