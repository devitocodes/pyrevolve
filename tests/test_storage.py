from pyrevolve.compression import init_compression
from pyrevolve.storage import BytesStorage
import numpy as np


def test_save_and_restore_no_compression():
    dtype = np.float32
    compression_scheme = 'zfp'
    compression = init_compression({'scheme': compression_scheme})
    store = BytesStorage(100, 5, dtype, compression, False)
    a = np.zeros((5, 5), dtype=dtype)
    b = np.ones((3, 3, ), dtype=dtype)
    store.save(0, [a, b])

    a1 = np.empty_like(a)
    b1 = np.empty_like(b)

    store.load(0, [a1, b1])

    assert(np.allclose(a, a1))
    assert(np.allclose(b, b1))
