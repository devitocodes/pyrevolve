import numpy as np
from functools import reduce
from operator import mul


class NumpyStorage(object):
    """Holds a chunk of memory large enough to store all checkpoints. The
    []-operator is overloaded to return a pointer to the memory reserved for a
    given checkpoint number. Revolve will typically use this as LIFO, but the
    storage also supports random access."""

    """Allocates memory on initialisation. Requires number of checkpoints and
    size of one checkpoint. Memory is allocated in C-contiguous style."""
    def __init__(self, size_ckp, n_ckp, dtype, profiler):
        self.storage = np.zeros((n_ckp, size_ckp), order='C', dtype=dtype)
        self.shapes = {}
        self.profiler = profiler

    """Returns a pointer to the contiguous chunk of memory reserved for the
    checkpoint with number `key`."""
    def __getitem__(self, key):
        return self.storage[key, :]

    def save(self, key, data_pointers):
        slot = self[key]
        offset = 0
        shapes = []
        for ptr in data_pointers:
            assert(ptr.flags.c_contiguous)
            with self.profiler.get_timer('storage', 'flatten'):
                data = ptr.ravel()
            with self.profiler.get_timer('storage', 'copy_save'):
                np.copyto(slot[offset:len(data)+offset], data)
            offset += len(data)
            shapes.append(ptr.shape)
        self.shapes[key] = shapes

    def load(self, key, locations):
        slot = self[key]
        offset = 0
        for shape, ptr in zip(self.shapes[key], locations):
            size = reduce(mul, ptr.shape)
            with self.profiler.get_timer('storage', 'copy_load'):
                np.copyto(ptr, slot[offset:offset+size].reshape(ptr.shape))
            offset += size


class BytesStorage(object):
    """Holds a chunk of memory large enough to store all checkpoints. The
    []-operator is overloaded to return a pointer to the memory reserved for a
    given checkpoint number. Revolve will typically use this as LIFO, but the
    storage also supports random access."""

    """Allocates memory on initialisation. Requires number of checkpoints and
    size of one checkpoint. Memory is allocated in C-contiguous style."""
    def __init__(self, size_ckp, n_ckp, dtype, compression, auto_pickle=False):
        size = size_ckp * n_ckp
        self.size_ckp = size_ckp
        self.n_ckp = n_ckp
        self.dtype = dtype
        self.storage = bytearray(size)
        self.auto_pickle = auto_pickle
        self.compressor, self.decompressor = compression
        self.lengths = {}

    """Returns a pointer to the contiguous chunk of memory reserved for the
    checkpoint with number `key`. May be a copy."""
    def __getitem__(self, key):
        ptr, start, end = self.get_location(key)
        return ptr[start:end]

    def get_location(self, key):
        assert(key < self.n_ckp)
        start = self.size_ckp * key
        end = start + self.size_ckp * np.dtype(self.dtype).itemsize
        return (self.storage, start, end)

    def save(self, key, data):
        logger.debug("ByteStorage: Saving to location %d" % key)
        logger.debug(np.linalg.norm(data))
        data = self.compressor(data)
        if not (isinstance(data, bytes) or isinstance(data, bytearray)):
            if not self.auto_pickle:
                raise TypeError("Expecting data to be bytes/bytearray, " +
                                "found %s" % type(data))
            else:
                data = pickle.dumps(data)
        
        ptr, start, end = self.get_location(key)
        logger.debug("Start: %d, End: %d" % (start, end))
        allowed_size = end - start
        actual_size = len(data)
        logger.debug("Actual size: %d" % actual_size)
        logger.debug(hashlib.md5(data).hexdigest())
        assert(actual_size <= allowed_size)
        self.storage[start:(start+actual_size)] = data
        self.lengths[key] = actual_size
        logger.debug(hashlib.md5(self.storage[start:(start+actual_size)]).hexdigest())
        logger.debug(hashlib.md5(self.storage[start:end]).hexdigest())
        logger.debug("Saved")

    def load(self, key, location):
        logger.debug("ByteStorage: Loading from location %d" % key)
        ptr, start, end = self.get_location(key)
        actual_size = self.lengths[key]
        logger.debug("Start: %d, End: %d" % (start, end))
        if not (isinstance(location, bytes) or
           isinstance(location, bytearray)) and self.auto_pickle:
            logger.debug(hashlib.md5(self.storage[start:(start+actual_size)]).hexdigest())
            logger.debug(str(self.storage[start:(start+actual_size)]))
            data = pickle.loads(self.storage[start:(start+actual_size)])
        else:
            data = self.storage[start:(start+actual_size)]
        
        logger.debug(np.linalg.norm(data))
        location[:] = self.decompressor(data)
        logger.debug("ByteStorage: Load complete")
