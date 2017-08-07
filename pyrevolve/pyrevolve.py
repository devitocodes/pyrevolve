import pyrevolve.crevolve as cr
import numpy as np
from abc import ABCMeta, abstractproperty, abstractmethod


class Checkpoint:
    """Abstract base class, containing the methods and properties that any
    user-given Checkpoint class must have."""
    __metaclass__ = ABCMeta

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


class CheckpointStorage:
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

    arg_names = {'t_start': 't_s', 't_end': 't_e'}
    
    def __init__(self, checkpoint,
                 fwd_operator, rev_operator, n_timesteps, n_checkpoints=None):
        """Initialise checkpointer for a given forward- and reverse operator, a
        given number of time steps, and a given storage strategy. The number of
        time steps must currently be provided explicitly, and the storage must
        be the single-staged memory storage."""
        if(n_timesteps is None):
            raise Exception("Online checkpointing not yet supported. Specify \
                              number of time steps!")
        n_timesteps = n_timesteps - 2
        if(n_checkpoints is None):
            n_checkpoints = cr.adjust(n_timesteps)
        self.fwd_operator = fwd_operator
        self.rev_operator = rev_operator
        self.checkpoint = checkpoint
        checkpoint.revolver = self
        self.storage = CheckpointStorage(checkpoint.size, n_checkpoints, checkpoint.dtype)
        self.n_timesteps = n_timesteps
        self.fwd_args = {}
        self.rev_args = {}
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
                self.call_fw(t_start=self.ckp.oldcapo, t_end=self.ckp.capo)
            elif(action == cr.Action.takeshot):
                # take a snapshot: copy from workspace into storage
                print("Taking snapshot number: %d"%self.ckp.check)
                self.checkpoint.save(self.storage[self.ckp.check])
            elif(action == cr.Action.restore):
                # restore a snapshot: copy from storage into workspace
                print("Restoring snapshot number: %d"%self.ckp.check)
                self.checkpoint.load(self.storage[self.ckp.check])
            elif(action == cr.Action.firstrun):
                # final step in the forward computation
                self.call_fw(t_start=self.ckp.oldcapo, t_end=self.n_timesteps)
                break

    def apply_reverse(self):
        """Executes only the backward computation while loading checkpoints,
        then returns. The forward operator will be called as needed to
        recompute sections of the trajectory that have not been stored in the
        forward run."""

        self.call_r(t_start=self.ckp.capo, t_end=self.ckp.capo+1)

        while(True):
            # ask Revolve what to do next.
            action = self.ckp.revolve()
            if(action == cr.Action.advance):
                # advance forward computation
                self.call_fw(t_start=self.ckp.oldcapo, t_end=self.ckp.capo)
            elif(action == cr.Action.takeshot):
                # take a snapshot: copy from workspace into storage
                print("Taking snapshot number: %d"%self.ckp.check)
                self.checkpoint.save(self.storage[self.ckp.check])
            elif(action == cr.Action.restore):
                # restore a snapshot: copy from storage into workspace
                print("Restoring snapshot number: %d"%self.ckp.check)
                self.checkpoint.load(self.storage[self.ckp.check])
            elif(action == cr.Action.youturn):
                # advance adjoint computation by a single step
                print("t=%d, L2(u(t))=%d"%(self.ckp.capo+1, np.linalg.norm(self.checkpoint.symbols[0].data[(self.ckp.capo+1)%3, :, :])))
                self.call_r(t_start=self.ckp.capo, t_end=self.ckp.capo+1)
            elif(action == cr.Action.terminate):
                break


    def call(self, t_start, t_end, args, op):
        args = args.copy()
        args[self.arg_names['t_start']] = t_start
        args[self.arg_names['t_end']] = t_end
        op.apply(**args)

    def call_fw(self, t_start, t_end):
        print("Forward from %d to %d, but actually: (%d, %d)"%(t_start, t_end, t_start, t_end+2))
        self.call(t_start, t_end+2, self.fwd_args, self.fwd_operator)

    def call_r(self, t_start, t_end):
        print("Reverse from %d to %d, but actually: (%d, %d)"%(t_end, t_start, t_start, t_end+2))
        self.call(t_start, t_end+2, self.rev_args, self.rev_operator)
