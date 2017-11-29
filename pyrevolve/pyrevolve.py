try:
    import pyrevolve.crevolve as cr
except ImportError:
    import crevolve as cr
import numpy as np
from abc import ABCMeta, abstractproperty, abstractmethod


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
    def save(self, ptr):
        """Deep-copy live data into the numpy array `ptr`."""
        return NotImplemented

    @abstractmethod
    def load(self, ptr):
        """Deep-copy from the numpy array `ptr` into the live data."""
        return NotImplemented


class CheckpointStorage(object):
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

    def __init__(self, checkpoint,
                 fwd_operator, rev_operator,
                 n_checkpoints=None, n_timesteps=None):
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
        self.storage = CheckpointStorage(checkpoint.size, n_checkpoints,
                                         checkpoint.dtype)
        self.n_timesteps = n_timesteps
        storage_disk = None  # this is not yet supported
        # We use the crevolve wrapper around the C++ Revolve library.
        self.ckp = cr.CRevolve(n_checkpoints, n_timesteps, storage_disk)

    def apply_forward(self):
        """Executes only the forward computation while storing checkpoints,
        then returns."""

        while(True):
            # ask Revolve what to do next.
            action = self.ckp.revolve()
            if(action == cr.Action.advance):
                # advance forward computation
                self.fwd_operator.apply(t_start=self.ckp.oldcapo,
                                        t_end=self.ckp.capo)
            elif(action == cr.Action.takeshot):
                # take a snapshot: copy from workspace into storage
                self.checkpoint.save(self.storage[self.ckp.check])
            elif(action == cr.Action.restore):
                # restore a snapshot: copy from storage into workspace
                self.checkpoint.load(self.storage[self.ckp.check])
            elif(action == cr.Action.firstrun):
                # final step in the forward computation
                self.fwd_operator.apply(t_start=self.ckp.oldcapo,
                                        t_end=self.n_timesteps)
                break

    def apply_reverse(self):
        """Executes only the backward computation while loading checkpoints,
        then returns. The forward operator will be called as needed to
        recompute sections of the trajectory that have not been stored in the
        forward run."""

        self.rev_operator.apply(t_start=self.ckp.capo,
                                t_end=self.ckp.capo+1)

        while(True):
            # ask Revolve what to do next.
            action = self.ckp.revolve()
            if(action == cr.Action.advance):
                # advance forward computation
                self.fwd_operator.apply(t_start=self.ckp.oldcapo,
                                        t_end=self.ckp.capo)
            elif(action == cr.Action.takeshot):
                # take a snapshot: copy from workspace into storage
                self.checkpoint.save(self.storage[self.ckp.check])
            elif(action == cr.Action.restore):
                # restore a snapshot: copy from storage into workspace
                self.checkpoint.load(self.storage[self.ckp.check])
            elif(action == cr.Action.youturn):
                # advance adjoint computation by a single step
                self.fwd_operator.apply(t_start=self.ckp.capo,
                                        t_end=self.ckp.capo+1)
                self.rev_operator.apply(t_start=self.ckp.capo,
                                        t_end=self.ckp.capo+1)
            elif(action == cr.Action.terminate):
                break
