import numpy as np
import pytest

from pyrevolve.compression import (compressors, decompressors,
                                   init_compression, compressors_available)
from pyrevolve import Revolver
from utils import IncrementOperator, YoCheckpoint


@pytest.mark.skip(reason="TODO multiple builds for compression")
def test_all_defined():
    for scheme in compressors_available:
        assert(scheme in compressors)
        assert(scheme in decompressors)


@pytest.mark.skip(reason="TODO multiple builds for compression")
def test_all_reversible():
    a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))
    for scheme in compressors_available:
        compressor, decompressor = init_compression({'scheme': scheme})
        compressed = compressor(a)
        decompressed = decompressor(compressed)
        assert(a.shape == decompressed.shape)
        assert(np.all(np.isclose(a, decompressed)))


@pytest.mark.parametrize("scheme", compressors_available)
def test_complete(scheme):
    nt = 100
    ncp = 10
    shape = (10, 10)
    a = np.zeros(shape)

    fwd = IncrementOperator(1, a)
    rev = IncrementOperator(-1, a)
    cp = YoCheckpoint(a)
    compression_params = {'scheme': scheme}
    revolver = Revolver(cp, fwd, rev, ncp, nt,
                        compression_params=compression_params)
    revolver.apply_forward()
    assert(np.all(np.isclose(a, np.zeros(shape) + nt)))
    revolver.apply_reverse()
    assert(np.all(np.isclose(a, np.zeros(shape))))


def test_compression_is_used():
    nt = 100
    ncp = 10
    shape = (10, 10)
    a = np.zeros(shape)

    fwd = IncrementOperator(1, a)
    rev = IncrementOperator(-1, a)
    cp = YoCheckpoint(a)
    counters = [0, 0]

    compressor, decompressor = compressors[None], decompressors[None]

    def this_compressor(params, data):
        counters[0] += 1
        return compressor(params, data)

    def this_decompressor(params, data):
        counters[1] += 1
        return decompressor(params, data)

    compression_params = {'scheme': 'custom', 'compressor': this_compressor,
                          'decompressor': this_decompressor}
    revolver = Revolver(cp, fwd, rev, ncp, nt,
                        compression_params=compression_params)
    revolver.apply_forward()
    assert(counters[0] >= ncp)
    revolver.apply_reverse()
    assert(counters[1] >= counters[0])
