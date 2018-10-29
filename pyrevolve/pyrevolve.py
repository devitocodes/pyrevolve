from abc import ABCMeta, abstractproperty, abstractmethod
import logging
import pickle

import numpy as np

try:
    import pyrevolve.crevolve as cr
except ImportError:
    import crevolve as cr
from .compression import init_compression as init
from .schedulers import Revolve, Action


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

    @abstractproperty
    def bytes(self):
        """Return the size of a single checkpoint, in number of entries."""
        return NotImplemented


class NumpyStorage(object):
    """Holds a chunk of memory large enough to store all checkpoints. The
    []-operator is overloaded to return a pointer to the memory reserved for a
    given checkpoint number. Revolve will typically use this as LIFO, but the
    storage also supports random access."""

    """Allocates memory on initialisation. Requires number of checkpoints and
    size of one checkpoint. Memory is allocated in C-contiguous style."""
    def __init__(self, size_ckp, n_ckp, dtype):
        self.storage = np.zeros((n_ckp, size_ckp), order='C', dtype=dtype)

    """Returns a pointer to the contiguous chunk of memory reserved for the
    checkpoint with number `key`."""
    def __getitem__(self, key):
        return self.storage[key, :]


class BytesStorage(object):
    """Holds a chunk of memory large enough to store all checkpoints. The
    []-operator is overloaded to return a pointer to the memory reserved for a
    given checkpoint number. Revolve will typically use this as LIFO, but the
    storage also supports random access."""

    """Allocates memory on initialisation. Requires number of checkpoints and
    size of one checkpoint. Memory is allocated in C-contiguous style."""
    def __init__(self, size_ckp, n_ckp, dtype, auto_pickle=False):
        size = size_ckp * n_ckp
        self.size_ckp = size_ckp
        self.n_ckp = n_ckp
        self.dtype = dtype
        self.storage = bytearray(size)
        self.auto_pickle = auto_pickle

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
        if not (isinstance(data, bytes) or isinstance(data, bytearray)):
            if not self.auto_pickle:
                raise TypeError("Expecting data to be bytes/bytearray but found %s" % type(data))
            else:
                data = pickle.dumps(data)

        ptr, start, end = self.get_location(key)
        allowed_size = end - start
        actual_size = len(data)
        assert(actual_size <= allowed_size)
        ptr[start:(start+actual_size)] = data

    def load(self, key, location):
        ptr, start, end = self.get_location(key)
        if self.auto_pickle:
            data = pickle.loads(ptr[start:end])
        else:
            data = ptr[start:end]

        location[:] = data[:]
        
        

class Revolver(object):
    """
    This should be the only user-facing class in here. It manages the
    interaction between the operators passed by the user, and the data storage.

    Todo:
        * Reverse operator is always called for a single step. Change this.
        * Avoid redundant data stores if higher-order stencils save multiple
          time steps, and checkpoints are close together.
        * Only offline single-stage is supported at the moment.
        * Give users a good handle on verbosity.
        * Find a better name than `checkpoint`, as the object with that name
          stores live data rather than one of the checkpoints.
    """

    def __init__(self, checkpoint, fwd_operator, rev_operator,
                 n_checkpoints=None, n_timesteps=None, compression_params={}):
        """Initialise checkpointer for a given forward- and reverse operator, a
        given number of time steps, and a given storage strategy. The number of
        time steps must currently be provided explicitly, and the storage must
        be the single-staged memory storage."""
        if(n_timesteps is None):
            raise Exception("Online checkpointing not yet supported. Specify \
                              number of time steps!")
        if(n_checkpoints is None):
            n_checkpoints = cr.adjust(n_timesteps)
        self.fwd_operator = fwd_operator
        self.rev_operator = rev_operator
        self.checkpoint = checkpoint
        self.storage = BytesStorage(checkpoint.bytes, n_checkpoints,
                                         checkpoint.dtype, auto_pickle=True)
        self.n_timesteps = n_timesteps

        self.scheduler = Revolve(n_checkpoints, n_timesteps)
        # cr.CRevolve(n_checkpoints, n_timesteps, storage_disk)
        self.compressor, self.decompressor = init(compression_params)

    def apply_forward(self):
        """Executes only the forward computation while storing checkpoints,
        then returns."""

        while(True):
            # ask Revolve what to do next.
            action = self.scheduler.next()
            if(action.type == Action.ADVANCE):
                # advance forward computation
                self.fwd_operator.apply(t_start=self.scheduler.old_capo,
                                        t_end=self.scheduler.capo)
            elif(action.type == Action.TAKESHOT):
                # take a snapshot: copy from workspace into storage
                self.save_checkpoint()
            elif(action.type == Action.RESTORE):
                # restore a snapshot: copy from storage into workspace
                self.load_checkpoint()
            elif(action.type == Action.LASTFW):
                # final step in the forward computation
                self.fwd_operator.apply(t_start=self.scheduler.old_capo,
                                        t_end=self.n_timesteps)
                break
            else:
                raise ValueError("Unknown action %d" % action)

    def apply_reverse(self):
        """Executes only the backward computation while loading checkpoints,
        then returns. The forward operator will be called as needed to
        recompute sections of the trajectory that have not been stored in the
        forward run."""

        self.rev_operator.apply(t_start=self.scheduler.capo,
                                t_end=self.scheduler.capo+1)

        while(True):
            # ask Revolve what to do next.
            action = self.scheduler.next()
            if(action.type == Action.ADVANCE):
                # advance forward computation
                self.fwd_operator.apply(t_start=self.scheduler.old_capo,
                                        t_end=self.scheduler.capo)
            elif(action.type == Action.TAKESHOT):
                # take a snapshot: copy from workspace into storage
                self.save_checkpoint()
            elif(action.type == Action.RESTORE):
                # restore a snapshot: copy from storage into workspace
                self.load_checkpoint()
            elif(action.type == Action.REVERSE):
                # advance adjoint computation by a single step
                self.fwd_operator.apply(t_start=self.scheduler.capo,
                                        t_end=self.scheduler.capo+1)
                self.rev_operator.apply(t_start=self.scheduler.capo,
                                        t_end=self.scheduler.capo+1)
            elif(action.type == Action.TERMINATE):
                break
            else:
                raise ValueError("Unknown action %s" % str(action))

    def save_checkpoint(self):
        data = self.checkpoint.get_data(self.scheduler.capo)
        self.storage.save(self.scheduler.cp_pointer, data)

    def load_checkpoint(self):
        location = self.checkpoint.get_data(self.scheduler.capo)
        self.storage.load(self.scheduler.cp_pointer, location)
