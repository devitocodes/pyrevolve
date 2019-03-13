import numpy as np
from functools import reduce
from operator import mul

from .logger import logger
from .compression import CompressedObject
import pickle


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
            assert(ptr.strides[-1] == ptr.itemsize)
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
        self.storage = memoryview(bytearray(size))
        self.auto_pickle = auto_pickle
        self.compressor, self.decompressor = compression
        self.lengths = {}
        self.metadata = {}

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
        logger.debug("ByteStorage: Saving to location %d/%d" % (key,
                                                                self.n_ckp))
        dataset = [self.compressor(x) for x in data]
        logger.debug("ByteStorage: Compression complete")
        offset = 0
        sizes = []
        metadatas = []
        ptr, start, end = self.get_location(key)
        for compressed_object in dataset:
            if not (isinstance(compressed_object, CompressedObject)):
                if not self.auto_pickle:
                    raise TypeError("Expecting data to be bytes/bytearray, " +
                                    "found %s" % type(compressed_object))
                else:
                    assert(isinstance(data, tuple) and len(data) == 2)
                    data, metadata = data
                    data = pickle.dumps(metadata)
            start += offset
            compressed_data = compressed_object.data
            metadata = compressed_object.metadata
            logger.debug("Start: %d, End: %d" % (start, end))
            allowed_size = end - start
            actual_size = len(compressed_data)
            logger.debug("Actual size: %d" % actual_size)

            assert(actual_size <= allowed_size)
            logger.debug(type(compressed_data))
            self.storage[start:(start+actual_size)] = compressed_data
            sizes.append(actual_size)
            offset += actual_size
            metadatas.append(metadata)

        self.lengths[key] = sizes
        self.metadata[key] = metadatas

    def load(self, key, locations):
        logger.debug("ByteStorage: Loading from location %d" % key)
        ptr, start, end = self.get_location(key)
        sizes = self.lengths[key]
        metadatas = self.metadata[key]

        assert(len(locations) == len(sizes) == len(metadatas))

        offset = 0
        for actual_size, metadata, location in zip(sizes, metadatas,
                                                   locations):
            logger.debug("Start: %d, End: %d" % (start, end))
            start += offset
            compressed_data = self.storage[start:(start+actual_size)]
            compressed_object = CompressedObject(compressed_data,
                                                 metadata=metadata)

            decompressed = self.decompressor(compressed_object)
            location[:] = decompressed
            offset += actual_size
        logger.debug("ByteStorage: Load complete")
