from pyrevolve import Operator, Checkpoint
import numpy as np
from operator import mul
from functools import reduce


def np_ref_address(ptr):
    return ptr.__array_interface__['data'][0]


class SimpleCheckpoint(Checkpoint):
    def __init__(self):
        self.save_counter = 0
        self.load_counter = 0

    def get_data(self, timestep):
        self.save_counter += 1
        return bytearray()

    def get_data_location(self, timestep):
        self.load_counter += 1
        return bytearray()

    @property
    def dtype(self):
        return np.float32

    @property
    def size(self):
        return 1

    @property
    def nbytes(self):
        return 1


class SimpleOperator(Operator):
    def __init__(self):
        self.counter = 0

    def apply(self, *args, **kwargs):
        t_start = kwargs['t_start']
        t_end = kwargs['t_end']
        assert(t_start <= t_end)
        self.counter += abs(t_end - t_start)


class IncrementOperator(Operator):
    def __init__(self, direction, field):
        assert(direction in (-1, 1))
        self.direction = direction
        self.field = field

    def apply(self, **kwargs):
        t_start = kwargs['t_start']
        t_end = kwargs['t_end']
        assert(t_start <= t_end)
        self.field[:] = self.field[:] + self.direction * abs(t_start - t_end)


class YoCheckpoint(Checkpoint):
    def __init__(self, field):
        self.field = field

    def get_data(self, timestep):
        return [self.field]

    def get_data_location(self, timestep):
        return [self.field]

    @property
    def dtype(self):
        return self.field.dtype

    @property
    def size(self):
        return reduce(mul, self.field.shape)

    @property
    def nbytes(self):
        return self.field.nbytes


class IncrementCheckpoint(Checkpoint):
    def __init__(self, _objects):
        self.objects = _objects
        dtypes = set([o.dtype for o in _objects])
        assert(len(dtypes) == 1)
        self._dtype = dtypes.pop()
        self._size = 0
        for o in _objects:
            self._size += o.size

    def get_data(self, timestep):
        return self.objects

    def get_data_location(self, timestep):
        return self.get_data(timestep)

    @property
    def dtype(self):
        return self._dtype

    @property
    def size(self):
        return self._size

    @property
    def nbytes(self):
        return self._size*(np.dtype(self._dtype).itemsize)


class IncOperator(Operator):
    def __init__(self, direction, u, v=None):
        assert(direction in (-1, 1))
        self.direction = direction
        self.u = u
        self.v = v
        self.counter = 0

    def apply(self, **kwargs):
        t_start = kwargs['t_start']
        t_end = kwargs['t_end']
        assert(t_start <= t_end)
        if self.direction == 1:
            self.u[:] = self.u[:] + self.direction * abs(t_start - t_end)
        else:
            self.v[:] = (self.u[:]*(-1) + 1)
        self.counter += abs(t_end - t_start)
