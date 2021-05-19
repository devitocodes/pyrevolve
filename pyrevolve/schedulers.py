from abc import ABCMeta, abstractmethod

try:
    import pyrevolve.crevolve as cr
except ImportError:
    import crevolve as cr

from .hrevolve import Function, argmin
from .hrevolve import Sequence, Operation, get_hopt_table
import json


class Architecture:
    """
    This class describes a multilevel memory
    architecture, with read/write access costs
    and memory sizes for each level.
    @attributes:
        nbleves:    number of memory levels
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

    type_names = {
        ADVANCE: "ADVANCE",
        TAKESHOT: "TAKESHOT",
        RESTORE: "RESTORE",
        LASTFW: "LASTFW",
        REVERSE: "REVERSE",
        CPDEL: "CPDEL",
        TERMINATE: "TERMINATE",
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
                    "capo": self.capo,
                    "old_capo": self.old_capo,
                    "ckp": self.ckp,
                }
            )
        )

    @abstractmethod
    def storageIndex(self):
        return NotImplemented


class CAction(Action):
    """
    This class is an specialization of the Action
    base class for CRevolve scheduler
    """

    def __init__(self, action_type, capo, old_capo, ckp):
        super().__init__(action_type, capo, old_capo, ckp)

    def storageIndex(self):
        return 0


class HAction(Action):
    """
    This class is an specialization of the Action
    base class for HRevolve scheduler
    """

    h_operations = {
        "Forward": Action.ADVANCE,
        "Forwards": Action.ADVANCE,
        "Backward": Action.REVERSE,
        "Checkpoint": Action.TAKESHOT,
        "Read": Action.RESTORE,
        "Write": Action.TAKESHOT,
        "Discard": Action.CPDEL,
        "Terminate": Action.TERMINATE,
    }

    def __init__(
        self, action_type=Action.TERMINATE, h_op=None, capo=0, old_capo=0, ckp=0
    ):
        if h_op is None:
            super().__init__(action_type, capo, old_capo, ckp)
        else:
            super().__init__(self.h_operations[h_op.type], capo, old_capo, ckp)
            if self.type == Action.ADVANCE:
                if h_op.type == "Forwards":
                    self.capo = h_op.index[1] + 1  # to
                    self.old_capo = h_op.index[0]  # from
                else:
                    self.capo = h_op.index + 1  # to
                    self.old_capo = h_op.index  # from
            elif self.type == Action.REVERSE:
                self.capo = h_op.index  # to
                self.old_capo = h_op.index  # from
            elif (
                (self.type == Action.TAKESHOT)
                or (self.type == Action.RESTORE)
                or (self.type == Action.CPDEL)
            ):
                self.capo = h_op.index[1]
                self.old_capo = h_op.index[1]

            self.index = h_op.index

    def storageIndex(self):
        if (
            (self.type == Action.TAKESHOT)
            or (self.type == Action.RESTORE)
            or (self.type == Action.CPDEL)
        ):
            return self.index[0]
        else:
            return 0

    def __repr__(self):
        return json.dumps(
            dict(
                {
                    "type": self.type_names[self.type],
                    "capo": self.capo,
                    "old_capo": self.old_capo,
                    "ckp": self.ckp,
                    "st_idx": self.storageIndex(),
                }
            )
        )


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


class CRevolve(Scheduler):
    """
    Scheduler class based on the CPP implementation of
    the traditional Revolve Algorithm
    """

    translations = {
        cr.Action.advance: Action.ADVANCE,
        cr.Action.takeshot: Action.TAKESHOT,
        cr.Action.restore: Action.RESTORE,
        cr.Action.firstrun: Action.LASTFW,
        cr.Action.youturn: Action.REVERSE,
        cr.Action.terminate: Action.TERMINATE,
    }

    def __init__(self, n_checkpoints, n_timesteps):
        super().__init__(n_checkpoints, n_timesteps)
        self.revolve = cr.CRevolve(n_checkpoints, n_timesteps, None)

    def next(self):
        return CAction(
            self.translations[self.revolve.revolve()],
            self.capo,
            self.old_capo,
            self.cp_pointer,
        )

    @property
    def capo(self):
        return self.revolve.capo

    @property
    def old_capo(self):
        return self.revolve.oldcapo

    @property
    def cp_pointer(self):
        return self.revolve.check


class HRevolve(Scheduler):
    """
    Scheduler class based on the Python implementation of
    the multilevel H-Revolve Algorithm described in the
    paper "H-Revolve: A Framework for Adjoint Computation
    on Synchronous Hierarchical Platforms"
    by Herrmann and Pallez [1]'

    Ref:
    [1] Herrmann, Pallez, "H-Revolve: A Framework for
    Adjoint Computation on Synchronous Hierarchical
    Platforms", ACM Transactions on Mathematical
    Software  46(2), 2020.
    """

    def __init__(self, n_checkpoints, n_timesteps, architecture=None, uf=1, ub=1, up=1):
        super().__init__(n_checkpoints, n_timesteps)

        if n_checkpoints != n_timesteps:
            raise ValueError(
                "HRevolveError: the number of checkpoints \
                must be equal to the number of timesteps"
            )

        self.hsequence = None
        if architecture is None:
            self.architecture = Architecture()  # loads default arch
        else:
            self.architecture = architecture
        self.uf = uf  # forward cost (default=1)
        self.ub = ub  # backward cost (default=1)
        self.up = up  # turn cost (default=1)
        self.__capo = 0
        self.__old_capo = 0
        self.__copindex = 0  # current operation index
        # compute h-revolve sequence
        self.__sequence = self.hrevolve(
            self.n_timesteps,
            self.architecture.nblevels - 1,
            self.architecture.sizes[-1],
        )
        self.__oplist = self.__sequence.get_flat_op_list()
        self.n_ops = len(self.__oplist)

    def resetSequence(self):
        self.__copindex = 0

    def next(self):
        if self.__copindex >= self.n_ops:
            return HAction(action_type=Action.TERMINATE)
        op = self.__oplist[self.__copindex]
        self.__copindex += 1
        ha = HAction(h_op=op)
        self.__capo = ha.capo
        self.__old_capo = ha.old_capo

        # H-revolve computes the forward steps Fi
        # with i ∈ {0, . . . , l − 1}).
        if ha.type is Action.ADVANCE and self.__capo >= (self.n_timesteps):
            ha.type = Action.LASTFW
        return ha

    @property
    def capo(self):
        return self.__capo

    @property
    def old_capo(self):
        return self.__old_capo

    @property
    def cp_pointer(self):
        return self.__capo

    @property
    def makespan(self):
        return self.__sequence.makespan

    def hrevolve_aux(self, l, K, cmem, hoptp=None, hopt=None):
        """
            This function is a copy of the orginal HRevolve_Aux
            function that composes the python H-Revolve implementation
            published by Herrmann and Pallez [1] in the following
            Gitlab repository:

            https://gitlab.inria.fr/adjoint-computation/H-Revolve/tree/master

            Some minor adaptations were made in order to allow is use in
            PyRevolve.

            @parameters:
                l : number of forward step to execute in the AC graph
                K: the level of memory
                cmem: number of available slots in the K-th level of memory
                Return the optimal sequence of \
                makespan HOpt(l, architecture)
        """
        cvect = self.architecture.sizes
        wvect = self.architecture.wd
        rvect = self.architecture.rd
        if (hoptp is None) or (hopt is None):
            (hoptp, hopt) = get_hopt_table(l, self.architecture)
        sequence = Sequence(
            Function("HRevolve_aux", l, [K, cmem]),
            self.architecture,
            self.uf,
            self.ub,
            self.up,
        )
        if cmem == 0:
            raise KeyError(
                "HRevolve_aux should not be call with cmem = 0.\
                 Contact developers."
            )
        if l == 0:
            sequence.insert(Operation("Backward", 0))
            return sequence
        if l == 1:
            if wvect[0] + rvect[0] < rvect[K]:
                sequence.insert(Operation("Write", [0, 0]))
            sequence.insert(Operation("Forward", 0))
            sequence.insert(Operation("Backward", 1))
            if wvect[0] + rvect[0] < rvect[K]:
                sequence.insert(Operation("Read", [0, 0]))
            else:
                sequence.insert(Operation("Read", [K, 0]))
            sequence.insert(Operation("Backward", 0))
            sequence.insert(Operation("Discard", [0, 0]))
            return sequence
        if K == 0 and cmem == 1:
            for index in range(l - 1, -1, -1):
                if index != l - 1:
                    sequence.insert(Operation("Read", [0, 0]))
                sequence.insert(Operation("Forwards", [0, index]))
                sequence.insert(Operation("Backward", index + 1))
            sequence.insert(Operation("Read", [0, 0]))
            sequence.insert(Operation("Backward", 0))
            sequence.insert(Operation("Discard", [0, 0]))
            return sequence
        if K == 0:
            list_mem = [
                j * self.uf
                + hopt[0][l - j][cmem - 1]
                + rvect[0]
                + hoptp[0][j - 1][cmem]
                for j in range(1, l)
            ]
            if min(list_mem) < hoptp[0][l][1]:
                jmin = argmin(list_mem)
                sequence.insert(Operation("Forwards", [0, jmin - 1]))
                sequence.insert_sequence(
                    self.hrevolve(l - jmin, 0, cmem - 1, hoptp=hoptp, hopt=hopt).shift(
                        jmin
                    )
                )
                sequence.insert(Operation("Read", [0, 0]))
                sequence.insert_sequence(
                    self.hrevolve_aux(jmin - 1, 0, cmem, hoptp=hoptp, hopt=hopt)
                )
                return sequence
            else:
                sequence.insert_sequence(
                    self.hrevolve_aux(l, 0, 1, hoptp=hoptp, hopt=hopt)
                )
                return sequence
        list_mem = [
            j * self.uf + hopt[K][l - j][cmem - 1] + rvect[K] + hoptp[K][j - 1][cmem]
            for j in range(1, l)
        ]
        if min(list_mem) < hopt[K - 1][l][cvect[K - 1]]:
            jmin = argmin(list_mem)
            sequence.insert(Operation("Forwards", [0, jmin - 1]))
            sequence.insert_sequence(
                self.hrevolve(l - jmin, K, cmem - 1, hoptp=hoptp, hopt=hopt).shift(jmin)
            )
            sequence.insert(Operation("Read", [K, 0]))
            sequence.insert_sequence(
                self.hrevolve_aux(jmin - 1, K, cmem, hoptp=hoptp, hopt=hopt)
            )
            return sequence
        else:
            sequence.insert_sequence(
                self.hrevolve(l, K - 1, cvect[K - 1], hoptp=hoptp, hopt=hopt)
            )
            return sequence

    def hrevolve(self, l, K, cmem, hoptp=None, hopt=None):
        """
        This function is a copy of the orginal HRevolve
        function that composes the python H-Revolve implementation
        published by Herrmann and Pallez [1] in the following
        Gitlab repository:

        https://gitlab.inria.fr/adjoint-computation/H-Revolve/tree/master

        Some minor adaptations were made in order to allow is use in
        PyRevolve.

        @parameters:
        l : number of forward step to execute in the AC graph
        K: the level of memory
        cmem: number of available slots in the K-th level of
        memory.
        Return the optimal sequence of makespan
        HOpt(l, architecture)
        """
        cvect = self.architecture.sizes
        wvect = self.architecture.wd
        if (hoptp is None) or (hopt is None):
            (hoptp, hopt) = get_hopt_table(l, self.architecture)
        sequence = Sequence(
            Function("HRevolve", l, [K, cmem]),
            self.architecture,
            self.uf,
            self.ub,
            self.up,
        )
        if l == 0:
            sequence.insert(Operation("Backward", 0))
            return sequence
        if K == 0 and cmem == 0:
            raise KeyError(
                "It's impossible to execute an AC \
                    graph of size > 0 with no memory."
            )
        if l == 1:
            sequence.insert(Operation("Write", [0, 0]))
            sequence.insert(Operation("Forward", 0))
            sequence.insert(Operation("Backward", 1))
            sequence.insert(Operation("Read", [0, 0]))
            sequence.insert(Operation("Backward", 0))
            sequence.insert(Operation("Discard", [0, 0]))
            return sequence
        if K == 0:
            sequence.insert(Operation("Write", [0, 0]))
            sequence.insert_sequence(
                self.hrevolve_aux(l, 0, cmem, hoptp=hoptp, hopt=hopt)
            )
            return sequence
        if wvect[K] + hoptp[K][l][cmem] < hopt[K - 1][l][cvect[K - 1]]:
            sequence.insert(Operation("Write", [K, 0]))
            sequence.insert_sequence(
                self.hrevolve_aux(l, K, cmem, hoptp=hoptp, hopt=hopt)
            )
            return sequence
        else:
            sequence.insert_sequence(
                self.hrevolve(l, K - 1, cvect[K - 1], hoptp=hoptp, hopt=hopt)
            )
            return sequence


Revolve = CRevolve
