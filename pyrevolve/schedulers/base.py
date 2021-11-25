from abc import ABCMeta, abstractmethod
import json


class Architecture:
    """
    This class describes a multilevel memory
    architecture, with read/write access costs
    and memory sizes for each level.
    @attributes:
        nblevels:    number of memory levels
        wd:         list of write costs
        rd:         list of read costs
        sizes:      number of checkpoins per storage
    """

    def __init__(self, storage_list, wd=[1], rd=[1], sizes=[1]):
        """
        Creates an instance of Architecture for a
        given storage list
        If no storage list is provided, creates a
        default object with a single memory
        level with wd=rd=1
        @attributes:
            nbleves:    number of memory levels
            wd:         list of write costs
            rd:         list of read costs
            sizes:      number of checkpoins per storage
        """
        if storage_list is None:
            self.nblevels = 1
            self.wd = wd
            self.rd = rd
            self.sizes = sizes
        else:
            self.wd = []
            self.rd = []
            self.sizes = []
            self.nblevels = len(storage_list)
            for st in storage_list:
                self.wd.append(st.wd)
                self.rd.append(st.rd)
                self.sizes.append(st.nckp)
        self.__check_arch_values__()

    def __repr__(self):
        l = []
        for i in range(self.nblevels):
            l.append((self.sizes[i], self.wd[i], self.rd[i]))
        return l.__repr__()

    def __check_arch_values__(self):
        if (
            (len(self.sizes) != self.nblevels)
            or (len(self.wd) != self.nblevels)
            or (len(self.rd) != self.nblevels)
        ):
            raise ImportError(
                "The number level in the architecture does \
                not correspond to the number of costs."
            )
        if (sorted(self.wd) != self.wd) or (sorted(self.rd) != self.rd):
            print(
                "WARNING!!! This code is optimal only if the \
                 costs of writing and reading of the architecture \
                 are in the increasing order for the levels."
            )


class Action(object):
    __metaclass__ = ABCMeta
    ADVANCE = 0
    TAKESHOT = 1
    RESTORE = 2
    LASTFW = 3
    REVERSE = 4
    CPDEL = 5
    TERMINATE = 6

    # Extra action for CRevolve scheduler
    REVSTART = 15

    type_names = {
        ADVANCE: "ADVANCE",
        TAKESHOT: "TAKESHOT",
        RESTORE: "RESTORE",
        LASTFW: "LASTFW",
        REVERSE: "REVERSE",
        CPDEL: "CPDEL",
        TERMINATE: "TERMINATE",
        REVSTART: "REVSTART",
    }

    def __init__(self, action_type, capo, old_capo, ckp):
        self.type = action_type
        self.capo = capo
        self.old_capo = old_capo
        self.ckp = ckp

    def __repr__(self):
        return json.dumps(
            dict(
                {
                    "type": self.type_names[self.type],
                    "from": self.old_capo,
                    "to": self.capo,
                    "ckp": self.ckp,
                }
            )
        )

    @abstractmethod
    def storageIndex(self):
        return NotImplemented


class Scheduler(object):
    """
    Abstract base class for scheduler implementations.
    This class defines the scheduler interface used by
    pyrevolve FSM.
    """

    __metaclass__ = ABCMeta

    def __init__(self, n_checkpoints, n_timesteps):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_checkpoints = n_checkpoints

    @abstractmethod
    def next(self):
        return NotImplemented

    @property
    def capo(self):
        return 0

    @property
    def old_capo(self):
        return 0

    @property
    def cp_pointer(self):
        return 0

    @property
    def oplist(self):
        return None
