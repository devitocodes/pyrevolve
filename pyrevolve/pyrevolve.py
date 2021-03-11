from abc import ABCMeta, abstractproperty, abstractmethod

import numpy as np
from . import crevolve as cr
from .compression import init_compression as init
from .schedulers import Revolve, Action
from .profiling import Profiler
from .storage import NumpyStorage, BytesStorage, DiskStorage


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


class BaseRevolver(object):
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
    __metaclass__ = ABCMeta

    def __init__(self, checkpoint, fwd_operator, rev_operator,
                 n_checkpoints=None, n_timesteps=None, timings=None,
                 profiler=None):
        """Initialise checkpointer for a given forward- and reverse operator, a
        given number of time steps, and a given storage strategy. The number of
        time steps must currently be provided explicitly, and the storage must
        be the single-staged memory storage."""
        if(n_timesteps is None):
            raise Exception("Online checkpointing not yet supported. Specify \
                              number of time steps!")
        if(n_checkpoints is None):
            self.n_checkpoints = cr.adjust(n_timesteps)
        else:
            self.n_checkpoints = n_checkpoints

        if profiler is None:
            self.profiler = Profiler()
        else:
            self.profiler = profiler

        self.timings = timings
        self.fwd_operator = fwd_operator
        self.rev_operator = rev_operator
        self.checkpoint = checkpoint
        self.n_timesteps = n_timesteps
        self.scheduler = Revolve(self.n_checkpoints, n_timesteps)

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

    @abstractmethod
    def createStorage(self):
        return NotImplemented


class DiskRevolver(BaseRevolver):
    """
        This class is an specialization of the Revolver class
        that uses a DiskStorage as its default storage method.
        When no 'filename' is provided, the storage
        .dat file is created inside the working directory.
        The storage file is removed by default when storage
        object is destroyed. DiskRevolver uses a single
        checkpoint file by default.
    """
    def __init__(self, checkpoint, fwd_operator, rev_operator,
                 n_checkpoints=None, n_timesteps=None, timings=None,
                 filedir="./", singlefile=True):
        super().__init__(checkpoint, fwd_operator, rev_operator, n_checkpoints,
                         n_timesteps, timings)
        self.filedir = filedir
        self.singlefile = singlefile
        self.createStorage()

    def createStorage(self):
        self.storage = DiskStorage(self.checkpoint.size, self.n_checkpoints,
                                   self.checkpoint.dtype, self.profiler,
                                   filedir=self.filedir,
                                   singlefile=self.singlefile)


class MemoryRevolver(BaseRevolver):
    """
        This class is an specialization of the Revolver class
        that uses a NumpyStorage as its default storage method.
    """
    def __init__(self, checkpoint, fwd_operator, rev_operator,
                 n_checkpoints=None, n_timesteps=None, timings=None,
                 compression_params=None):
        super().__init__(checkpoint,
                         fwd_operator,
                         rev_operator,
                         n_checkpoints,
                         n_timesteps,
                         timings)

        if compression_params is None:
            compression_params = {'scheme': None}

        self.compression_params = compression_params
        self.createStorage()

    def createStorage(self):
        if self.compression_params['scheme'] is None:
            self.storage = NumpyStorage(self.checkpoint.size, self.n_checkpoints,
                                        self.checkpoint.dtype,
                                        profiler=self.profiler)
        else:
            compressor, decompressor = init(self.compression_params)
            self.storage = BytesStorage(self.checkpoint.nbytes, self.n_checkpoints,
                                        self.checkpoint.dtype, auto_pickle=True,
                                        compression=(compressor, decompressor))


""" To keep backward compatibility with previous testcases
and all previous codes that use the name Revolver """
Revolver = MemoryRevolver
