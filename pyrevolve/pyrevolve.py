from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np

try:
    import pyrevolve.crevolve as cr
except ImportError:
    import crevolve as cr
from .compression import init_compression as init
from .schedulers import Revolve, Action
from .logger import logger
from . import custom_pickle as pickle
import hashlib
from timeit import default_timer
from functools import reduce
from operator import mul


class Timer(object):
    def __init__(self, profiler, section, action):
        self.timer = default_timer
        self.profiler = profiler
        self.section = section
        self.action = action
        
    def __enter__(self):
        self.start = self.timer()
        return self
        
    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        self.profiler.increment(self.section, self.action, self.elapsed)

        
class Profiler(object):
    def __init__(self):
        self.timings = {}
        self.counts = {}

    def get_timer(self, section, action):
        return Timer(self, section, action)

    def increment(self, section, action, elapsed):
        # Warning: Not thread safe
        section_timings = self.timings.get(section, {})
        section_timings[action] = section_timings.get(action, 0) + elapsed
        self.timings[section] = section_timings

        section_counts = self.counts.get(section, {})
        section_counts[action] = section_counts.get(action, 0) + 1
        self.counts[section] = section_counts

    def summary(self):
        summary = '****************'
        for section, section_timings in self.timings.items():
            summary += '\nIn section %s:' % section
            for action, action_time in section_timings.items():
                summary += '\n\tAction %s: %f (%d)' % (action, action_time, self.counts[section][action])
        summary += '\n****************'
        return summary

    def get_dict(self):
        results = {}
        for s_n, s_dict in self.timings.items():
            for a_n, a_time in s_dict.items():
                results['%s_%s_timing' % (s_n, a_n)] = a_time

        for s_n, s_dict in self.counts.items():
            for a_n, a_time in s_dict.items():
                results['%s_%s_counts' % (s_n, a_n)] = a_time

        return results


class Operator(object):
    """ Abstract base class for an Operator that may be used with pyRevolve."""
    __metaclass__ = ABCMeta

    def apply(self, **kwargs):
        pass


class Checkpoint(object):
    """Abstract base class, containing the methods and properties that any
    user-given Checkpoint class must have."""
    __metaclass__ = ABCMeta

    @abstractproperty
    def dtype(self):
        """Return the numpy-compatible dtype of the checkpoint data
           (float32/float64)
        """
        return NotImplemented

    @abstractproperty
    def size(self):
        """Return the size of a single checkpoint, in number of entries."""
        return NotImplemented

    @abstractmethod
    def get_data(self, timestep):
        """Deep-copy live data into the numpy array `ptr`."""
        return NotImplemented

    @property
    def nbytes(self):
        return self.size * np.dtype(self.dtype).itemsize


class NumpyStorage(object):
    """Holds a chunk of memory large enough to store all checkpoints. The
    []-operator is overloaded to return a pointer to the memory reserved for a
    given checkpoint number. Revolve will typically use this as LIFO, but the
    storage also supports random access."""

    """Allocates memory on initialisation. Requires number of checkpoints and
    size of one checkpoint. Memory is allocated in C-contiguous style."""
    def __init__(self, size_ckp, n_ckp, dtype):
        self.storage = np.zeros((n_ckp, size_ckp), order='C', dtype=dtype)
        self.shapes = {}

    """Returns a pointer to the contiguous chunk of memory reserved for the
    checkpoint with number `key`."""
    def __getitem__(self, key):
        return self.storage[key, :]

    def save(self, key, data_pointers):
        slot = self[key]
        offset = 0
        shapes = []
        for ptr in data_pointers:
            data = ptr.flatten()
            np.copyto(slot[offset:len(data)+offset], ptr.flatten())
            offset += len(data)
            shapes.append(ptr.shape)
        self.shapes[key] = shapes

    def load(self, key, locations):
        slot = self[key]
        offset = 0
        for shape, ptr in zip(self.shapes[key], locations):
            size = reduce(mul, ptr.shape)
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


class Revolver(object):
    """
    This should be the only user-facing class in here. It manages the
    interaction between the operators passed by the user, and the data storage.

    TODO:
        * Reverse operator is always called for a single step. Change this.
        * Avoid redundant data stores if higher-order stencils save multiple
          time steps, and checkpoints are close together.
        * Only offline single-stage is supported at the moment.
        * Give users a good handle on verbosity.
        * Find a better name than `checkpoint`, as the object with that name
          stores live data rather than one of the checkpoints.
    """

    def __init__(self, checkpoint, fwd_operator, rev_operator,
                 n_checkpoints=None, n_timesteps=None, timings=None, compression_params=None):
        """Initialise checkpointer for a given forward- and reverse operator, a
        given number of time steps, and a given storage strategy. The number of
        time steps must currently be provided explicitly, and the storage must
        be the single-staged memory storage."""
        if(n_timesteps is None):
            raise Exception("Online checkpointing not yet supported. Specify \
                              number of time steps!")
        if(n_checkpoints is None):
            n_checkpoints = cr.adjust(n_timesteps)
        if compression_params is None:
            compression_params = {}
        self.timings = timings
        self.fwd_operator = fwd_operator
        self.rev_operator = rev_operator
        self.checkpoint = checkpoint
        compressor, decompressor = init(compression_params)
        self.storage = NumpyStorage(checkpoint.size, n_checkpoints, checkpoint.dtype)
        #self.storage = BytesStorage(checkpoint.nbytes, n_checkpoints,
        #                            checkpoint.dtype, auto_pickle=True,
        #                            compression=(compressor, decompressor))
        self.n_timesteps = n_timesteps

        self.scheduler = Revolve(n_checkpoints, n_timesteps)
        self.profiler = Profiler()

    def apply_forward(self):
        """Executes only the forward computation while storing checkpoints,
        then returns."""

        while(True):
            # ask Revolve what to do next.
            action = self.scheduler.next()
            if(action.type == Action.ADVANCE):
                # advance forward computation
                with self.profiler.get_timer('forward', 'advance'):
                    self.fwd_operator.apply(t_start=self.scheduler.old_capo,
                                            t_end=self.scheduler.capo)
            elif(action.type == Action.TAKESHOT):
                # take a snapshot: copy from workspace into storage
                with self.profiler.get_timer('forward', 'takeshot'):
                    self.save_checkpoint()
            elif(action.type == Action.LASTFW):
                # final step in the forward computation
                with self.profiler.get_timer('forward', 'lastfw'):
                    self.fwd_operator.apply(t_start=self.scheduler.old_capo,
                                            t_end=self.n_timesteps)
                break
            else:
                raise ValueError("Unknown action %s" % str(action))

    def apply_reverse(self):
        """Executes only the backward computation while loading checkpoints,
        then returns. The forward operator will be called as needed to
        recompute sections of the trajectory that have not been stored in the
        forward run."""

        with self.profiler.get_timer('reverse', 'reverse'):
            self.rev_operator.apply(t_start=self.scheduler.capo,
                                    t_end=self.scheduler.capo+1)

        while(True):
            # ask Revolve what to do next.
            action = self.scheduler.next()
            if(action.type == Action.ADVANCE):
                # advance forward computation
                with self.profiler.get_timer('reverse', 'advance'):
                    self.fwd_operator.apply(t_start=self.scheduler.old_capo,
                                            t_end=self.scheduler.capo)
            elif(action.type == Action.TAKESHOT):
                # take a snapshot: copy from workspace into storage
                with self.profiler.get_timer('reverse', 'takeshot'):
                    self.save_checkpoint()
            elif(action.type == Action.RESTORE):
                # restore a snapshot: copy from storage into workspace
                with self.profiler.get_timer('reverse', 'restore'):
                    self.load_checkpoint()
            elif(action.type == Action.REVERSE):
                # advance adjoint computation by a single step
                with self.profiler.get_timer('reverse', 'reverse'):
                    self.fwd_operator.apply(t_start=self.scheduler.capo,
                                            t_end=self.scheduler.capo+1)
                    self.rev_operator.apply(t_start=self.scheduler.capo,
                                            t_end=self.scheduler.capo+1)
            elif(action.type == Action.TERMINATE):
                break
            else:
                raise ValueError("Unknown action %s" % str(action))

    def save_checkpoint(self):
        data_pointers = self.checkpoint.get_data(self.scheduler.capo)
        self.storage.save(self.scheduler.cp_pointer, data_pointers)

    def load_checkpoint(self):
        locations = self.checkpoint.get_data_location(self.scheduler.capo)
        self.storage.load(self.scheduler.cp_pointer, locations)
