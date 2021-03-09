from utils import SimpleOperator, SimpleCheckpoint
from pyrevolve import Revolver

import pytest


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


# The following test had to be disabled since, with the new API, the
# user's code never sees the pointers to the locations where data is
# written. This test needs to be conceptually rewritten
# TODO: Rewrite test so it implements a custom storage object to test

# @pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
#                                    (10, 9), (10, 10), (10, 11), (10, 12)])
# def test_ptr_loads_and_saves(nt, ncp):
#    cp = SimpleCheckpoint()
#    f = SimpleOperator()
#    b = SimpleOperator()

#    rev = Revolver(cp, f, b, ncp, nt)
#    rev.apply_forward()
#    rev.apply_reverse()
#    assert(cp.save_pointers == cp.load_pointers)
#    assert(len(cp.save_pointers) == min(ncp, nt - 1))
