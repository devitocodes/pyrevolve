from pyrevolve import Operator, Checkpoint
import numpy as np


def np_ref_address(ptr):
    return ptr.__array_interface__['data'][0]


class SimpleCheckpoint(Checkpoint):
    def __init__(self):
        self.save_counter = 0
        self.load_counter = 0
        self.save_pointers = set()
        self.load_pointers = set()

    def save(self, ptr, compressor):
        self.save_counter += 1
        self.save_pointers.add(np_ref_address(ptr))

    def load(self, ptr, decompressor):
        self.load_counter += 1
        self.load_pointers.add(np_ref_address(ptr))

    @property
    def dtype(self):
        return np.float32

    @property
    def size(self):
        return 10


class SimpleOperator(Operator):
    def __init__(self):
        self.counter = 0

    def apply(self, *args, **kwargs):
        t_start = kwargs['t_start']
        t_end = kwargs['t_end']
        self.counter += abs(t_end - t_start)
