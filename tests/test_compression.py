import numpy as np

from pyrevolve.compression import (compressors, decompressors, allowed_names,
                                   init_compression)
from pyrevolve import Revolver
from utils import IncrementOperator, YoCheckpoint

def test_all_defined():
    for scheme in allowed_names:
        assert(scheme in compressors)
        assert(scheme in decompressors)


def test_all_reversible():
    a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))
    for scheme in allowed_names:
        compressor, decompressor = init_compression({'scheme': scheme})
        compressed = compressor(a)
        decompressed = decompressor(compressed)
        assert(a.shape == decompressed.shape)
        assert(np.all(np.isclose(a, decompressed)))


def test_complete():
    nt = 100
    ncp = 10
    shape = (10, 10)
    a = np.zeros(shape)
    
    fwd = IncrementOperator(1, a)
    rev = IncrementOperator(-1, a)
    cp = YoCheckpoint(a)
    compression_params = {'scheme': None}
    revolver = Revolver(cp, fwd, rev, ncp, nt, compression_params=compression_params)
    revolver.apply_forward()
    assert(np.all(np.isclose(a, np.zeros(shape) + nt)))
    revolver.apply_reverse()
    assert(np.all(np.isclose(a, np.zeros(shape))))
