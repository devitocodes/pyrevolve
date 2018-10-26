import numpy as np

from pyrevolve.compression import compressors, decompressors, allowed_names


def test_all_defined():
    for scheme in allowed_names:
        assert(scheme in compressors)
        assert(scheme in decompressors)


def test_all_reversible():
    a = np.linspace(0, 100, num=1000000).reshape((100, 100, 100))
    for scheme in allowed_names:
        compressor = compressors[scheme]
        decompressor = decompressors[scheme]
        compressed = compressor(a)
        decompressed = decompressor(compressed)
        assert(a.shape == decompressed.shape)
        assert(np.all(np.isclose(a, decompressed)))
