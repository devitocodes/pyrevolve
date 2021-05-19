from pyrevolve.compression import init_compression, compressors_available
from pyrevolve.storage import BytesStorage
from utils import SimpleOperator, SimpleCheckpoint
from utils import IncrementCheckpoint, IncOperator
from pyrevolve import SingleLevelRevolver
import numpy as np
import pytest


@pytest.mark.parametrize("scheme", compressors_available)
def test_save_and_restore_with_compression(scheme):
    dtype = np.float32
    ncp = 5

    compression = init_compression({'scheme': scheme})
    store = BytesStorage(100, ncp, dtype, compression, False)
    a = np.zeros((5, 5), dtype=dtype)
    b = np.ones((3, 3, ), dtype=dtype)

    a1 = np.empty_like(a)
    b1 = np.empty_like(b)
    for i in range(ncp):
        store.save(0, [a+i, b+i])
        store.load(0, [a1, b1])
        assert(np.allclose(a+i, a1))
        assert(np.allclose(b+i, b1))


@pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
                                     (10, 9), (10, 10), (10, 11), (10, 12)])
@pytest.mark.parametrize("singlefile", [True, False])
@pytest.mark.parametrize("diskckp", [True, False])
def test_forward_nt(nt, ncp, singlefile, diskckp):
    df = np.zeros([nt, ncp])
    db = np.zeros([nt, ncp])
    cp = IncrementCheckpoint([df, db])
    f = IncOperator(1, df)
    b = IncOperator(-1, db)

    rev = SingleLevelRevolver(cp, f, b, ncp, nt,
                              diskstorage=diskckp,
                              filedir="./",
                              singlefile=singlefile)
    assert(f.counter == 0)
    rev.apply_forward()
    assert(f.counter == nt)


@pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
                                     (10, 9), (10, 10), (10, 11), (10, 12)])
@pytest.mark.parametrize("singlefile", [True, False])
@pytest.mark.parametrize("diskckp", [True, False])
def test_reverse_nt(nt, ncp, singlefile, diskckp):
    df = np.zeros([nt, ncp])
    db = np.zeros([nt, ncp])
    cp = IncrementCheckpoint([df])
    f = IncOperator(1, df)
    b = IncOperator(-1, df, db)

    rev = SingleLevelRevolver(cp, f, b, ncp, nt,
                              diskstorage=diskckp,
                              filedir="./",
                              singlefile=singlefile)

    rev.apply_forward()
    assert(b.counter == 0)
    rev.apply_reverse()
    assert(b.counter == nt)
    assert(np.count_nonzero(db) == 0)


@pytest.mark.parametrize("nt, ncp", [(10, 2), (10, 4), (10, 6), (10, 8),
                                     (10, 9), (10, 10), (10, 11), (10, 12)])
@pytest.mark.parametrize("singlefile", [True, False])
@pytest.mark.parametrize("diskckp", [True, False])
def test_num_loads_and_saves(nt, ncp, singlefile, diskckp):
    cp = SimpleCheckpoint()
    f = SimpleOperator()
    b = SimpleOperator()

    rev = SingleLevelRevolver(cp, f, b, ncp, nt,
                              diskstorage=diskckp,
                              filedir="./",
                              singlefile=singlefile)

    rev.apply_forward()
    assert(cp.load_counter == 0)
    rev.apply_reverse()

    assert(cp.load_counter >= cp.save_counter)
