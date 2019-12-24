from pyrevolve.compression import init_compression, compressors_available
from pyrevolve.storage import BytesStorage
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
