from pyrevolve import Operator, Checkpoint, Revolver

import numpy as np
import pytest


def np_ref_address(ptr):
    return ptr.__array_interface__['data'][0]


class SimpleCheckpoint(Checkpoint):
    def __init__(self):
        self.save_counter = 0
        self.load_counter = 0
        self.save_pointers = set()
        self.load_pointers = set()

    def save(self, ptr):
        self.save_counter += 1
        self.save_pointers.add(np_ref_address(ptr))

    def load(self, ptr):
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


@pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
                                     (10, 9), (10, 10), (10, 11), (10, 12)])
def test_forward_nt(nt, ncp):
    cp = SimpleCheckpoint()
    f = SimpleOperator()
    b = SimpleOperator()

    rev = Revolver(cp, f, b, ncp, nt)
    assert(f.counter == 0)
    rev.apply_forward()

    assert(f.counter == nt)


@pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
                                     (10, 9), (10, 10), (10, 11), (10, 12)])
def test_reverse_nt(nt, ncp):
    cp = SimpleCheckpoint()
    f = SimpleOperator()
    b = SimpleOperator()

    rev = Revolver(cp, f, b, ncp, nt)
    rev.apply_forward()
    assert(b.counter == 0)
    rev.apply_reverse()

    assert(b.counter == nt)


@pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
                                     (10, 9), (10, 10), (10, 11), (10, 12)])
def test_number_of_saves_in_forward(nt, ncp):
    cp = SimpleCheckpoint()
    f = SimpleOperator()
    b = SimpleOperator()

    rev = Revolver(cp, f, b, ncp, nt)
    assert(cp.save_counter == 0)
    rev.apply_forward()
    assert(cp.save_counter == min(ncp, nt-1))


@pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
                                     (10, 9), (10, 10), (10, 11), (10, 12)])
def test_num_loads_and_saves(nt, ncp):
    cp = SimpleCheckpoint()
    f = SimpleOperator()
    b = SimpleOperator()

    rev = Revolver(cp, f, b, ncp, nt)
    rev.apply_forward()
    assert(cp.load_counter == 0)
    rev.apply_reverse()

    assert(cp.load_counter >= cp.save_counter)


@pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
                                     (10, 9), (10, 10), (10, 11), (10, 12)])
def test_ptr_loads_and_saves(nt, ncp):
    cp = SimpleCheckpoint()
    f = SimpleOperator()
    b = SimpleOperator()

    rev = Revolver(cp, f, b, ncp, nt)
    rev.apply_forward()
    rev.apply_reverse()
    assert(cp.save_pointers == cp.load_pointers)
    assert(len(cp.save_pointers) == min(ncp, nt - 1))
