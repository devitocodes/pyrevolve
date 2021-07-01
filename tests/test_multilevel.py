from utils import SimpleOperator, SimpleCheckpoint
from utils import IncrementCheckpoint, IncOperator
from pyrevolve import MultiLevelRevolver, MemoryRevolver
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

    npStorage = NumpyStorage(cp.size, nt, cp.dtype, wd=mwd, rd=mrd)
    dkStorage = DiskStorage(
        cp.size, nt, cp.dtype, filedir="./", singlefile=singlefile, wd=dwd, rd=drd
    )
    st_list = [npStorage, dkStorage]
    rev = MultiLevelRevolver(cp, f, b, nt, storage_list=st_list, uf=uf, ub=ub)
    assert f.counter == 0
    rev.apply_forward()
    assert f.counter == nt


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

    npStorage = NumpyStorage(cp.size, nt, cp.dtype, wd=mwd, rd=mrd)
    dkStorage = DiskStorage(
        cp.size, nt, cp.dtype, filedir="./", singlefile=singlefile, wd=dwd, rd=drd
    )
    st_list = [npStorage, dkStorage]
    rev = MultiLevelRevolver(cp, f, b, nt, storage_list=st_list, uf=uf, ub=ub)

    rev.apply_forward()
    assert f.counter == nt
    assert b.counter == 0
    rev.apply_reverse()
    assert np.count_nonzero(db) == 0


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

    npStorage = NumpyStorage(cp.size, nt, cp.dtype, wd=mwd, rd=mrd)
    dkStorage = DiskStorage(
        cp.size, nt, cp.dtype, filedir="./", singlefile=singlefile, wd=dwd, rd=drd
    )
    st_list = [npStorage, dkStorage]
    rev = MultiLevelRevolver(cp, f, b, nt, storage_list=st_list, uf=uf, ub=ub)

    rev.apply_forward()
    assert cp.load_counter == 0
    rev.apply_reverse()

    assert cp.load_counter >= cp.save_counter


@pytest.mark.parametrize("nt", [1, 5, 10, 20])
@pytest.mark.parametrize("mwd", [0, 2])
@pytest.mark.parametrize("mrd", [0, 2])
@pytest.mark.parametrize("dwd", [0, 2])
@pytest.mark.parametrize("drd", [0, 2])
@pytest.mark.parametrize("uf", [1, 2])
@pytest.mark.parametrize("ub", [1, 2])
def test_multi_and_single_outputs(nt, mwd, mrd, dwd, drd, uf, ub):
    """
    Tests whether SingleLevelRevolver and MultilevelRevolver are producing
    the same outputs
    """
    nx = 10
    ny = 10
    const = 1
    m_df = np.zeros([nx, ny])
    m_db = np.zeros([nx, ny])
    m_cp = IncrementCheckpoint([m_df])
    m_fwd = IncOperator(const, m_df)
    m_rev = IncOperator((-1) * const, m_df, m_db)
    s_df = np.zeros([nx, ny])
    s_db = np.zeros([nx, ny])
    s_cp = IncrementCheckpoint([s_df])
    s_fwd = IncOperator(const, s_df)
    s_rev = IncOperator((-1) * const, s_df, s_db)

    m_npStorage = NumpyStorage(m_cp.size, nt, m_cp.dtype, wd=mwd, rd=mrd)
    m_dkStorage = DiskStorage(
        m_cp.size, nt, m_cp.dtype, filedir="./", singlefile=False, wd=dwd, rd=drd
    )
    st_list = [m_npStorage, m_dkStorage]
    m_wrp = MultiLevelRevolver(
        m_cp, m_fwd, m_rev, nt, storage_list=st_list, uf=uf, ub=ub
    )

    s_wrp = MemoryRevolver(s_cp, s_fwd, s_rev, nt, nt)

    m_wrp.apply_forward()
    s_wrp.apply_forward()
    assert m_fwd.counter == nt
    assert s_fwd.counter == nt
    assert m_rev.counter == 0
    assert s_rev.counter == 0
    assert (m_df == s_df).all()

    m_wrp.apply_reverse()
    s_wrp.apply_reverse()
    assert np.count_nonzero(m_db) == 0
    assert np.count_nonzero(s_db) == 0
    assert (m_db == s_db).all()
