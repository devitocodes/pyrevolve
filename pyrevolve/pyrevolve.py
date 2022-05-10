from abc import ABCMeta, abstractproperty, abstractmethod
import numpy as np
from . import crevolve as cr
from .compression import init_compression as init
from .schedulers import CRevolve, HRevolve, Action, Architecture
from .profiling import Profiler
from .storage import NumpyStorage, BytesStorage, DiskStorage


class Operator(object):
    """Abstract base class for an Operator that may be used with pyRevolve."""

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
    interaction between the operators passed by the user, and the data
    storages.
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

    storage_list = []

    def __init__(
        self,
        checkpoint,
        fwd_operator,
        rev_operator,
        n_checkpoints,
        n_timesteps,
        storage_list=None,
        scheduler=None,
        timings=None,
        profiler=None,
    ):
        """
        Initialises checkpointer for a given forward- and reverse operator, a
        given number of time steps, and a set of storage strategies. The number
        of time steps must currently be provided explicitly. A list of storage
        methods and a scheduler object must be provided as well. Otherwise
        NumpyStorage and CRevolve are used as default
        """
        if n_timesteps is None:
            raise Exception(
                "Online checkpointing not yet supported. Specify \
                              number of time steps!"
            )

        if profiler is None:
            self.profiler = Profiler()
        else:
            self.profiler = profiler

        if storage_list is None:
            self.storage_list = []
        else:
            self.storage_list = storage_list

        self.checkpoint = checkpoint
        self.n_checkpoints = n_checkpoints
        self.n_timesteps = n_timesteps
        self.timings = timings
        self.fwd_operator = fwd_operator
        self.rev_operator = rev_operator
        self.scheduler = scheduler

    def addStorage(self, new_storage):
        self.storage_list.append(new_storage)

    def removeStorage(self, st_idx):
        if st_idx < len(self.storage_list):
            del self.storage_list[st_idx]

    def resetStorageList(self):
        self.storage_list.clear()

    def addDiskStorage(self, filedir="./", singlefile=False, wd=0, rd=0):
        diskSt = DiskStorage(
            self.checkpoint.size,
            self.n_checkpoints,
            self.checkpoint.dtype,
            profiler=self.profiler,
            filedir=filedir,
            singlefile=singlefile,
            wd=wd,
            rd=rd,
        )
        self.addStorage(diskSt)
        return len(self.storage_list) - 1  # st index

    def addNumpyStorage(self, compression_params=None):
        npSt = None
        if compression_params is None:
            compression_params = {"scheme": None}
        if compression_params["scheme"] is None:
            npSt = NumpyStorage(
                self.checkpoint.size,
                self.n_checkpoints,
                self.checkpoint.dtype,
                profiler=self.profiler,
            )
        else:
            compressor, decompressor = init(self.compression_params)
            npSt = BytesStorage(
                self.checkpoint.nbytes,
                self.n_checkpoints,
                self.checkpoint.dtype,
                auto_pickle=True,
                compression=(compressor, decompressor),
                profiler=self.profiler,
            )
        self.addStorage(npSt)
        return len(self.storage_list) - 1  # st index

    def addByteStorage(self, compression_params):
        compressor, decompressor = init(compression_params)
        npSt = BytesStorage(
            self.checkpoint.nbytes,
            self.n_checkpoints,
            self.checkpoint.dtype,
            auto_pickle=True,
            compression=(compressor, decompressor),
            profiler=self.profiler,
        )
        self.addStorage(npSt)
        return len(self.storage_list) - 1  # st index

    @property
    def makespan(self):
        return 0

    @property
    def ratio(self):
        return 0

    def apply_forward(self):
        """Executes only the forward computation while storing checkpoints,
        then returns."""
        while True:
            # ask Revolve what to do next.
            action = self.scheduler.next()
            if action.type == Action.ADVANCE:
                # advance forward computation
                with self.profiler.get_timer("forward", "advance"):
                    self.fwd_operator.apply(
                        t_start=self.scheduler.old_capo, t_end=self.scheduler.capo
                    )
            elif action.type == Action.TAKESHOT:
                # take a snapshot: copy from workspace into storage
                with self.profiler.get_timer("forward", "takeshot"):
                    self.save_checkpoint(action.storageIndex())
            elif action.type == Action.CPDEL:
                # remove a snapshot from the storage stack
                with self.profiler.get_timer("forward", "remove"):
                    self.remove_checkpoint(action.storageIndex())
            elif action.type == Action.LASTFW:
                # final step in the forward computation
                with self.profiler.get_timer("forward", "lastfw"):
                    self.fwd_operator.apply(
                        t_start=self.scheduler.old_capo, t_end=self.n_timesteps
                    )
                break
            elif action.type == Action.REVERSE:
                """HRevolve scheduler doesn't have an explicit LASTFW operation.
                Because of that, aplly_forward ends when the first REVERSE
                action is reached.
                """
                break
            else:
                raise ValueError("Unknown action %s" % str(action))

    def apply_reverse(self):
        """Executes only the backward computation while loading checkpoints,
        then returns. The forward operator will be called as needed to
        recompute sections of the trajectory that have not been stored in the
        forward run."""
        action = None
        while True:
            # ask Revolve what to do next.
            action = self.scheduler.next()
            if action.type == Action.REVERSE:
                # advance adjoint computation by a single step
                with self.profiler.get_timer("reverse", "reverse"):
                    self.fwd_operator.apply(
                        t_start=self.scheduler.capo, t_end=self.scheduler.capo + 1
                    )
                    self.rev_operator.apply(
                        t_start=self.scheduler.capo, t_end=self.scheduler.capo + 1
                    )
            elif action.type == Action.REVSTART:
                """Sets the rev_operator to 'nt' only if its not already there.
                This condition happens when using CRevolve shceduler, but not
                when using HRevolve.
                """
                with self.profiler.get_timer("reverse", "reverse"):
                    self.rev_operator.apply(
                        t_start=self.scheduler.capo, t_end=self.scheduler.capo + 1
                    )
            elif action.type == Action.TAKESHOT:
                # take a snapshot: copy from workspace into storage
                with self.profiler.get_timer("reverse", "takeshot"):
                    self.save_checkpoint(action.storageIndex())
            elif action.type == Action.ADVANCE:
                # advance forward computation
                with self.profiler.get_timer("reverse", "advance"):
                    self.fwd_operator.apply(
                        t_start=self.scheduler.old_capo, t_end=self.scheduler.capo
                    )
            elif action.type == Action.RESTORE:
                # restore a snapshot: copy from storage into workspace
                with self.profiler.get_timer("reverse", "restore"):
                    self.load_checkpoint(action.storageIndex())
            elif action.type == Action.CPDEL:
                # remove a snapshot from the storage stack
                with self.profiler.get_timer("reverse", "remove"):
                    self.remove_checkpoint(action.storageIndex())
            elif action.type == Action.TERMINATE:
                break
            else:
                raise ValueError("Unknown action %s" % str(action))

    def save_checkpoint(self, st_idx=0):
        data_pointers = self.checkpoint.get_data(self.scheduler.capo)
        self.storage_list[st_idx].save(self.scheduler.cp_pointer, data_pointers)

    def load_checkpoint(self, st_idx=0):
        locations = self.checkpoint.get_data_location(self.scheduler.capo)
        self.storage_list[st_idx].load(self.scheduler.cp_pointer, locations)

    def remove_checkpoint(self, st_idx=0):
        return NotImplemented

    def storage_ckps(self, k=0):
        """Returns a list of all checkpoint keys stored at the k-th
        storage level"""
        return NotImplemented


class SingleLevelRevolver(BaseRevolver):
    """
    This class is an specialization of the BaseRevolver class
    that uses either DiskStorage or NumpyStorage as single level
    storage method. SingleLevelRevolver uses CRevolve scheduler which is
    based on tradional single-level Revolve algorithm.

    If DiskStorage is used:
        - When no 'filename' is provided, the storage
          .dat file is created inside 'filedir' directory.
        - 'singlefile' specifies whether multiple or a single
          storage file is used. (default = single file).
        - The storage file is removed by default when storage
          object is destroyed.

    If NumpyStorage is used:
        - No compression scheme is set as default.
    """

    def __init__(
        self,
        checkpoint,
        fwd_operator,
        rev_operator,
        n_checkpoints,
        n_timesteps,
        timings=None,
        profiler=None,
        compression_params=None,
        diskstorage=False,
        filedir="./",
        singlefile=True,
    ):
        """
        Initializes a single-level Revolver
        @params:
            checkpoint:         checkpoint object
            fwd_operator:       forward operator
            rev_operator:       backward operator
            n_checkpoints:      number of checkpoints
            n_timesteps:        number of timesteps
            timings:            timings
            profiler:           Profiler
            compression_params: compression scheme
            diskstorage:        True for using disk storage
            filedir:            disk storage directory
            singlefile:         True for single-file disk storage
        """
        super().__init__(
            checkpoint,
            fwd_operator,
            rev_operator,
            n_checkpoints,
            n_timesteps,
            timings=timings,
            profiler=profiler,
        )

        self.filedir = filedir
        self.singlefile = singlefile

        if n_checkpoints is None:
            self.n_checkpoints = cr.adjust(n_timesteps)
        else:
            self.n_checkpoints = n_checkpoints

        self.scheduler = CRevolve(self.n_checkpoints, self.n_timesteps)

        # remove storage list to avoid memory overflow
        self.resetStorageList()
        if diskstorage is True:
            self.addDiskStorage(filedir=self.filedir, singlefile=self.singlefile)
        else:
            self.compression_params = compression_params
            self.addNumpyStorage(compression_params)

    @property
    def ratio(self):
        return self.scheduler.ratio

    def storage_ckps(self, k=0):
        """Returns a list of all checkpoint keys stored at the k-th
        storage level"""
        # single level always uses first storage object on storage_list
        return self.scheduler.storage(0)


class MultiLevelRevolver(BaseRevolver):
    """
    This class is an specialization of the Revolver class
    that can use both NumpyStorage and Diskstorage as
    storage methods. The HRevolve scheduler is used
    to manage checkpointing with multiple storages.
    It implements H-revolve algorithm as proposed in
    the paper "H-Revolve: A Framework for Adjoint
    Computationon Synchronous Hierarchical Platforms"
    by Herrmann and Pallez [1]'

    Ref:
    [1] Herrmann, Pallez, "H-Revolve: A Framework for
    Adjoint Computation on Synchronous Hierarchical
    Platforms", ACM Transactions on Mathematical
    Software  46(2), 2020.
    """

    def __init__(
        self,
        checkpoint,
        fwd_operator,
        rev_operator,
        n_timesteps,
        storage_list,
        timings=None,
        profiler=None,
        uf=1,
        ub=1,
        up=1,
    ):
        """
        Initializes a multi-level Revolver using HRevolve
        scheduler. The user MUST provide a list of storage
        objects. If an empty storage list is provided,
        the scheduler attribute will not be created. In this
        case, the user can add storages after the Revover
        creation and reload the scheduler attribute.

        @params:
            checkpoint:         checkpoint object
            fwd_operator:       forward operator
            rev_operator:       backward operator
            n_timesteps:        number of timesteps
            timings:            timings
            profiler:           profiler
            storage_list:       list of storage objects
            uf:                 forward operation cost
            ud:                 backward operation cost
            up:                 turn operation cost
        """
        super().__init__(
            checkpoint,
            fwd_operator,
            rev_operator,
            n_timesteps,
            n_timesteps,
            storage_list=storage_list,
            timings=timings,
            profiler=profiler,
        )
        self.uf = uf  # forward cost (default=1)
        self.ub = ub  # backward cost (default=1)
        self.up = up  # turn cost (default=1)
        self.arch = None
        if storage_list is not None:
            for st in storage_list:
                # add Revolver profiler to each storage
                st.profiler = self.profiler
            self.reload_scheduler()
        else:
            raise ValueError("Parameter 'storage_list'can not be None.")

    def reload_scheduler(self, uf=1, ub=1, up=1):
        """
        Reloads the scheduler object based on
        the current storage list.
        """
        if len(self.storage_list) > 0:
            self.uf = uf
            self.ub = ub
            self.up = up
            self.arch = Architecture(self.storage_list)
            self.scheduler = HRevolve(
                self.n_checkpoints, self.n_timesteps, self.arch, self.uf, self.ub, self.up
            )
        else:
            raise ValueError(
                "Empty 'storage_list'. Storage list \
                must contain at least one storage method."
            )

    @property
    def makespan(self):
        return self.scheduler.makespan

    @property
    def ratio(self):
        return self.scheduler.ratio

    def storage_ckps(self, k=0):
        """Returns a list of all checkpoint keys stored at the k-th
        storage level"""
        return self.scheduler.storage(k)

    def save_checkpoint(self, st_idx=0):
        data_pointers = self.checkpoint.get_data(self.scheduler.capo)
        self.storage_list[st_idx].push(data_pointers)

    def load_checkpoint(self, st_idx=0):
        locations = self.checkpoint.get_data_location(self.scheduler.capo)
        self.storage_list[st_idx].peek(locations)

    def remove_checkpoint(self, st_idx=0):
        locations = self.checkpoint.get_data_location(self.scheduler.capo)
        self.storage_list[st_idx].pop(locations)


class MemoryRevolver(SingleLevelRevolver):
    """
    This class is an specialization of the SingleLevelRevolver class
    that uses a NumpyStorage as its single-level storage method.
    SingleLevelRevolver uses CRevolve scheduler which is
    based on the classic single-level Revolve algorithm.
    By default, no compression scheme is set to the NumpyStorage.
    """

    def __init__(
        self,
        checkpoint,
        fwd_operator,
        rev_operator,
        n_checkpoints,
        n_timesteps,
        timings=None,
        profiler=None,
        compression_params=None,
    ):
        super().__init__(
            checkpoint,
            fwd_operator,
            rev_operator,
            n_checkpoints,
            n_timesteps,
            timings=timings,
            profiler=profiler,
            compression_params=compression_params,
            diskstorage=False,
        )


class DiskRevolver(SingleLevelRevolver):
    """
    This class is an specialization of the SingleLevelRevolver class
    that uses a DiskStorage as its single-level storage method.
    SingleLevelRevolver uses CRevolve scheduler which is
    based on the classic single-level Revolve algorithm.
    About DiskStorage:
        - When no 'filename' is provided, the storage
          .dat file is created inside 'filedir' directory.
        - 'singlefile' specifies whether multiple or a single
          storage file is used. (default = single file).
        - The storage file is removed by default when storage
          object is destroyed.
    """

    def __init__(
        self,
        checkpoint,
        fwd_operator,
        rev_operator,
        n_checkpoints,
        n_timesteps,
        timings=None,
        profiler=None,
        filedir="./",
        singlefile=True,
    ):
        super().__init__(
            checkpoint,
            fwd_operator,
            rev_operator,
            n_checkpoints,
            n_timesteps,
            timings=timings,
            profiler=profiler,
            diskstorage=True,
            filedir=filedir,
            singlefile=singlefile,
        )


""" To keep backward compatibility with previous testcases
and all previous codes that use the name Revolver """
Revolver = MemoryRevolver
