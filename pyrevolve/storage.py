from abc import ABCMeta, abstractmethod
import numpy as np
import datetime
from functools import reduce
from operator import mul
from .logger import logger
from .compression import CompressedObject
import pickle
import os
import shutil


class Storage(object):
    """
    Abstract class for storage implementations.
    The user can access the storage object either by Vector
    interface ('load/save'), or by Stack interface
    ('push/peek/pop'). However, using both Vector and
    Stack interface at the same time is not advised, as it
    can mess up internal pointers.
    CRevolve Scheduler uses Vector interface while H-Revolve
    uses Stack.

    Attributes:
        size_ckp:   size of one checkpoint (in bytes)
        n_ckp:      number of checkpoints
        dtype:      C-like data type
        profiler:   profiler
        wd:         write cost
        rd:         rd cost
        name:       storage name
    """

    __metaclass__ = ABCMeta

    def __init__(self, size_ckp, n_ckp, dtype, profiler, wd=0, rd=0, name="storage"):
        """
        @attributes:
            size_ckp:   size of one checkpoint (in bytes)
            n_ckp:      number of checkpoints
            dtype:      C-like data type
            profiler:   profiler
            wd:         write cost
            rd:         rd cost
            name:       storage name
        """
        super().__init__()
        self.size_ckp = size_ckp
        self.n_ckp = n_ckp
        self.dtype = dtype
        self.profiler = profiler
        self.wd = wd
        self.rd = rd
        self.name = name
        self.__maxsize = self.size_ckp * n_ckp
        self.__current_size = 0
        self.__itemsize = np.dtype(self.dtype).itemsize

        # stack interface controls
        self.__stack_ptr = -1

    @abstractmethod
    def save(self, key, data_pointers):
        return NotImplemented

    @abstractmethod
    def load(self, key, locations):
        return NotImplemented

    def push(self, data_pointers):
        if self.isFull():
            ValueError(
                "StorageError: Trying to push into an \
                already full stack."
            )
        else:
            self.__stack_ptr += 1
            self.save(self.__stack_ptr, data_pointers)

    def peek(self, locations):
        if self.isEmpty():
            return
        else:
            self.load(self.__stack_ptr, locations)

    def pop(self, locations):
        if self.isEmpty():
            raise ValueError("StorageError: Trying to pop from an empty stack.")
        else:
            self.load(self.__stack_ptr, locations)
            self.__stack_ptr -= 1

    def isFull(self):
        return self.__stack_ptr >= (self.n_ckp - 1)

    def isEmpty(self):
        return self.__stack_ptr == -1

    @property
    def maxsize(self):
        """Returns the maximum storage size in words of 'dtype'"""
        return self.__maxsize

    @property
    def maxsize_in_bytes(self):
        """Returns the maximum storage size in bytes"""
        return self.__maxsize*self.__itemsize

    @property
    def size(self):
        """Returns the current storage size in words of 'dtype'"""
        return self.__current_size

    @property
    def nckp(self):
        """Returns the storage size in 'number_of_chkp'.
        This property is used by H-Revolve implementation
        """
        return self.n_ckp


class DiskStorage(Storage):
    """
    Stores all checkpoints on one or multiple .dat binary filess,
    depending of 'singlefile' flag. The []-operator is overloaded to
    return a file object with its file-pointer placed according to
    the checkpoint key.
    Revolve will typically use this as LIFO, but the storage also supports
    random access. By default, the .dat file is removed when the object
    is destroyed.
    """

    """
    Requires number of checkpoints and size of one checkpoint.
    'filedir': base directory where a dat/ folder is created.
    All .dat binary files are stored into 'fildir/dat/' folder.
    'singlefile': lets the user decide whether to use one or
    multiple files to store checkpoints.
    """

    def __init__(
        self,
        size_ckp,
        n_ckp,
        dtype,
        profiler=None,
        filedir="./",
        singlefile=True,
        wd=0,
        rd=0,
        name="DiskStorage",
        keepfiles=False
    ):
        super().__init__(size_ckp, n_ckp, dtype, profiler, wd=wd, rd=rd, name=name)
        self.singlefile = singlefile
        self.keepfiles = keepfiles
        self.filedir = None
        self.__myPID = os.getpid()
        self.__myTimestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.__datDirName = "dat_D{}_PID{}/".format(self.__myTimestamp, self.__myPID)
        self.__datFileTemplate = "CKP_D{}_PID{}.dat".format(
            self.__myTimestamp, self.__myPID
        )
        if filedir is None:
            self.filedir = "./" + self.__datDirName
        else:
            self.filedir = filedir + self.__datDirName

        if not os.path.exists(self.filedir):
            os.makedirs(self.filedir)

        """ create unique file names"""
        self.datFileName = self.filedir + self.__datFileTemplate

        self.__current_size = 0
        self.shapes = {}
        if self.singlefile is True:
            self.storage_w = open(self.datFileName, "bw+")
            self.storage_r = open(self.datFileName, "br+")
            self.default_storage = self.storage_w

    def __del__(self):
        """ Removes .dat file by default """
        if self.keepfiles is False:
            self.removeDatdir()

    def removeDatdir(self):
        """ Removes dat file directory """
        shutil.rmtree(self.filedir, ignore_errors=True)

    def __setW(self):
        """Sets WRITE stream as default"""
        self.default_storage = self.storage_w

    def __setR(self):
        """Sets READ stream as default"""
        self.default_storage = self.storage_r

    def checkFilesDir(self):
        if not os.path.exists(self.filedir):
            os.makedirs(self.filedir)

    def __getitem__(self, key):
        """Returns a pointer to the contiguous chunk of memory reserved for the
        checkpoint with number `key`."""
        assert key < self.n_ckp
        noffset = key * self.size_ckp
        if self.__current_size > 0:
            assert noffset <= self.__current_size
        foffset = noffset * (np.dtype(self.dtype).itemsize)
        """Moves the file-pointer to position determined
        by the 'key' parameter"""
        self.default_storage.seek(foffset, os.SEEK_SET)
        return self.default_storage

    def save(self, key, data_pointers):
        with self.profiler.get_timer("storage", "copy_save"):
            shapes = []
            if self.singlefile is True:
                self.__setW()
                slot = self[key]
            else:
                ckpfile = self.datFileName + (".k%d" % (key))
                self.checkFilesDir()
                slot = open(ckpfile, "bw")

            for ptr in data_pointers:
                assert ptr.strides[-1] == ptr.itemsize
                with self.profiler.get_timer("storage", "flatten"):
                    data = ptr.ravel()
                    data = data.astype(self.dtype)
                data.tofile(slot)
                slot.flush()
                self.__current_size += self.size_ckp
                shapes.append(ptr.shape)
            self.shapes[key] = shapes
            if self.singlefile is False:
                slot.close()

    def load(self, key, locations):
        with self.profiler.get_timer("storage", "copy_load"):
            if self.singlefile is True:
                self.__setR()
                slot = self[key]
            else:
                ckpfile = self.datFileName + (".k%d" % (key))
                self.checkFilesDir()
                slot = open(ckpfile, "br")

            offset = 0
            for shape, ptr in zip(self.shapes[key], locations):
                size = reduce(mul, ptr.shape)
                ckp = np.fromfile(slot, dtype=self.dtype, count=size)
                np.copyto(ptr, ckp.reshape(ptr.shape))
                offset += size
            if self.singlefile is False:
                slot.close()


class NumpyStorage(Storage):
    """Holds a chunk of memory large enough to store all checkpoints. The
    []-operator is overloaded to return a pointer to the memory reserved for a
    given checkpoint number. Revolve will typically use this as LIFO, but the
    storage also supports random access."""

    """Allocates memory on initialisation. Requires number of checkpoints and
    size of one checkpoint. Memory is allocated in C-contiguous style."""

    def __init__(
        self, size_ckp, n_ckp, dtype, profiler=None, wd=0, rd=0, name="MemoryStorage"
    ):
        super().__init__(size_ckp, n_ckp, dtype, profiler, wd=wd, rd=rd, name=name)
        self.storage = np.zeros((n_ckp, size_ckp), order="C", dtype=dtype)
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
            assert ptr.strides[-1] == ptr.itemsize
            with self.profiler.get_timer("storage", "flatten"):
                data = ptr.ravel()
            with self.profiler.get_timer("storage", "copy_save"):
                np.copyto(slot[offset:(len(data) + offset)], data)

            offset += len(data)
            shapes.append(ptr.shape)
        self.shapes[key] = shapes

    def load(self, key, locations):
        slot = self[key]
        offset = 0
        for shape, ptr in zip(self.shapes[key], locations):
            size = reduce(mul, ptr.shape)
            with self.profiler.get_timer("storage", "copy_load"):
                np.copyto(ptr, slot[offset:offset + size].reshape(ptr.shape))
            offset += size


class BytesStorage(Storage):
    """Holds a chunk of memory large enough to store all checkpoints. The
    []-operator is overloaded to return a pointer to the memory reserved for a
    given checkpoint number. Revolve will typically use this as LIFO, but the
    storage also supports random access."""

    """Allocates memory on initialisation. Requires number of checkpoints and
    size of one checkpoint. Memory is allocated in C-contiguous style."""

    def __init__(
        self,
        size_ckp,
        n_ckp,
        dtype,
        compression,
        auto_pickle=False,
        profiler=None,
        wd=0,
        rd=0,
        name="ByteStorage",
    ):
        super().__init__(size_ckp, n_ckp, dtype, profiler, wd=wd, rd=rd, name=name)
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
        assert key < self.n_ckp
        start = self.size_ckp * key
        end = start + self.size_ckp * np.dtype(self.dtype).itemsize
        return (self.storage, start, end)

    def save(self, key, data):
        logger.debug("ByteStorage: Saving to location %d/%d" % (key, self.n_ckp))
        dataset = [self.compressor(x) for x in data]
        logger.debug("ByteStorage: Compression complete")
        offset = 0
        sizes = []
        metadatas = []
        ptr, start, end = self.get_location(key)
        for compressed_object in dataset:
            if not (isinstance(compressed_object, CompressedObject)):
                if not self.auto_pickle:
                    raise TypeError(
                        "Expecting data to be bytes/bytearray, "
                        + "found %s" % type(compressed_object)
                    )
                else:
                    assert isinstance(data, tuple) and len(data) == 2
                    data, metadata = data
                    data = pickle.dumps(metadata)
            start += offset
            compressed_data = compressed_object.data
            metadata = compressed_object.metadata
            logger.debug("Start: %d, End: %d" % (start, end))
            allowed_size = end - start
            actual_size = len(compressed_data)
            logger.debug("Actual size: %d" % actual_size)

            assert actual_size <= allowed_size
            logger.debug(type(compressed_data))
            self.storage[start:(start + actual_size)] = compressed_data
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

        assert len(locations) == len(sizes) == len(metadatas)

        offset = 0
        for actual_size, metadata, location in zip(sizes, metadatas, locations):
            logger.debug("Start: %d, End: %d" % (start, end))
            start += offset
            compressed_data = self.storage[start:(start + actual_size)]
            compressed_object = CompressedObject(compressed_data, metadata=metadata)

            decompressed = self.decompressor(compressed_object)
            location[:] = decompressed
            offset += actual_size
        logger.debug("ByteStorage: Load complete")
