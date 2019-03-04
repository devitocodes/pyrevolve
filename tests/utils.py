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
        print("Appyling from %d to %d" % (t_start, t_end))
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
        print("bytes", self.field.nbytes)
        return reduce(mul, self.field.shape)
