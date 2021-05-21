from utils import SimpleOperator, SimpleCheckpoint
from utils import IncrementCheckpoint, IncOperator
from pyrevolve import MultiLevelRevolver
from pyrevolve import NumpyStorage, DiskStorage
import numpy as np
import pytest


@pytest.mark.parametrize("nt", [1, 5, 10, 20, 100])
@pytest.mark.parametrize("singlefile", [True, False])
@pytest.mark.parametrize("mwd", [0, 2])
@pytest.mark.parametrize("mrd", [0, 2])
@pytest.mark.parametrize("dwd", [0, 2])
@pytest.mark.parametrize("drd", [0, 2])
@pytest.mark.parametrize("uf", [1, 2])
@pytest.mark.parametrize("ub", [1, 2])
def test_forward_nt(nt, mwd, mrd, dwd, drd, uf, ub, singlefile):
    nx = 10
    ny = 10
    df = np.zeros([nx, ny])
    db = np.zeros([nx, ny])
    cp = IncrementCheckpoint([df, db])
    f = IncOperator(1, df)
    b = IncOperator(-1, db)

    npStorage = NumpyStorage(cp.size, nt, cp.dtype,
                             wd=mwd, rd=mrd)
    dkStorage = DiskStorage(cp.size, nt, cp.dtype,
                            filedir="./",
                            singlefile=singlefile,
                            wd=dwd, rd=drd)
    st_list = [npStorage, dkStorage]
    rev = MultiLevelRevolver(cp, f, b, nt,
                             storage_list=st_list,
                             uf=uf, ub=ub)
    assert(f.counter == 0)
    rev.apply_forward()
    assert(f.counter == nt)


@pytest.mark.parametrize("nt", [1, 5, 10, 20])
@pytest.mark.parametrize("singlefile", [True, False])
@pytest.mark.parametrize("mwd", [0, 2])
@pytest.mark.parametrize("mrd", [0, 2])
@pytest.mark.parametrize("dwd", [0, 2])
@pytest.mark.parametrize("drd", [0, 2])
@pytest.mark.parametrize("uf", [1, 2])
@pytest.mark.parametrize("ub", [1, 2])
def test_reverse_nt(nt, mwd, mrd, dwd, drd, uf, ub, singlefile):
    nx = 10
    ny = 10
    df = np.zeros([nx, ny])
    db = np.zeros([nx, ny])
    cp = IncrementCheckpoint([df])
    f = IncOperator(1, df)
    b = IncOperator(-1, df, db)

    npStorage = NumpyStorage(cp.size, nt, cp.dtype,
                             wd=mwd, rd=mrd)
    dkStorage = DiskStorage(cp.size, nt, cp.dtype,
                            filedir="./",
                            singlefile=singlefile,
                            wd=dwd, rd=drd)
    st_list = [npStorage, dkStorage]
    rev = MultiLevelRevolver(cp, f, b, nt,
                             storage_list=st_list,
                             uf=uf, ub=ub)

    rev.apply_forward()
    assert(f.counter == nt)
    assert(b.counter == 0)
    rev.apply_reverse()
    assert(np.count_nonzero(db) == 0)


@pytest.mark.parametrize("nt", [1, 5, 10, 20])
@pytest.mark.parametrize("singlefile", [True, False])
@pytest.mark.parametrize("mwd", [0, 2])
@pytest.mark.parametrize("mrd", [0, 2])
@pytest.mark.parametrize("dwd", [0, 2])
@pytest.mark.parametrize("drd", [0, 2])
@pytest.mark.parametrize("uf", [1, 2])
@pytest.mark.parametrize("ub", [1, 2])
def test_num_loads_and_saves(nt, mwd, mrd, dwd, drd, uf, ub, singlefile):
    cp = SimpleCheckpoint()
    f = SimpleOperator()
    b = SimpleOperator()

    npStorage = NumpyStorage(cp.size, nt, cp.dtype,
                             wd=mwd, rd=mrd)
    dkStorage = DiskStorage(cp.size, nt, cp.dtype,
                            filedir="./",
                            singlefile=singlefile,
                            wd=dwd, rd=drd)
    st_list = [npStorage, dkStorage]
    rev = MultiLevelRevolver(cp, f, b, nt,
                             storage_list=st_list,
                             uf=uf, ub=ub)

    rev.apply_forward()
    assert(cp.load_counter == 0)
    rev.apply_reverse()

    assert(cp.load_counter >= cp.save_counter)
