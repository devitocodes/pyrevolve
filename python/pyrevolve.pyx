cimport declarations

import numpy as np
from enum import Enum
from warnings import warn

class RevolveError(Exception):
    pass
    # TODO: the hardcoded limits really should be removed in a future version. This should be as easy as replacing the arrays in the C++ code with an std::vector.
    errCodes = {
        10: "Number of checkpoints stored exceeds hardcoded limit. Modify 'checkup' and recompile pyrevolve.",
        11: "Number of checkpoints stored exceeds snaps. This is a bug in pyrevolve, please report it.",
        12: "Internal error in numforw, please report a pyrevolve bug.",
        13: "Enhancement of 'fine', increase 'snaps'. Please rerun and allow for more checkpoints to be stored.",
        14: "Number of snaps exceeds hardcoded limit. Modify 'snapsup' and recompile pyrevolve.",
        15: "Number of reps exceeds hardcoded limit. Modify 'repsup' and recompile pyrevolve."
    }

class Action(Enum):
    advance   = 1
    takeshot  = 2
    restore   = 3
    firstrun  = 4
    youturn   = 5
    terminate = 6
    error     = 7

def adjust(timesteps):
    cdef int c_st = timesteps
    return declarations.revolve_adjust(c_st)

def maxrange(snapshots, timefactor):
    cdef int c_ss = snapshots
    cdef int c_tt = timefactor
    return declarations.revolve_maxrange(c_ss, c_tt)

def numforw(timesteps, snapshots):
    cdef int c_st = timesteps
    cdef int c_sn = snapshots
    return declarations.revolve_numforw(c_st, c_sn)

def expense(timesteps, snapshots):
    cdef int c_st = timesteps
    cdef int c_sn = snapshots
    return declarations.revolve_expense(c_st, c_sn)

cdef class checkpointer(object):
    cdef declarations.CRevolve __r

    def __init__(self, snapshots, timesteps=None, snapshots_disk=None):
        cdef int c_sn
        cdef int c_st
        cdef int c_sr
        
        # if no number of steps is given, we need an online strategy
        if(timesteps == None):
            if(snapshots_disk != None):
                warn("Multi-stage online checkpointing is not implemented.")
                warn("Using single-stage online checkpointing instead.")
            c_sn = snapshots
            self.__r = declarations.revolve_create_online(c_sn)
        # if number of steps is given and no multi-stage strategy is requested,
        # use standard offline Revolve
        elif(snapshots_disk == None):
            c_sn = snapshots
            c_st = timesteps
            self.__r = declarations.revolve_create_offline(c_st, c_sn)
        # number of steps is known and multi-stage is requested,
        # use offline multistage strategy
        else:
            c_sn = snapshots
            c_st = timesteps
            c_sr = snapshots_disk
            self.__r = declarations.revolve_create_multistage(c_st, c_sr, c_sn)

    @property
    def info(self):
        return declarations.revolve_getinfo(self.__r)

    def revolve(self):
        cdef declarations.CACTION action
        action = declarations.revolve(self.__r)
        if(action == declarations.CACTION_ADVANCE):
            retAction = Action.advance
        elif(action == declarations.CACTION_TAKESHOT):
            retAction = Action.takeshot
        elif(action == declarations.CACTION_RESTORE):
            retAction = Action.restore
        elif(action == declarations.CACTION_FIRSTRUN):
            retAction = Action.firstrun
        elif(action == declarations.CACTION_YOUTURN):
            retAction = Action.youturn
        elif(action == declarations.CACTION_TERMINATE):
            retAction = Action.terminate
        else:
            # in this case, action must be "error"
            retAction = Action.error
            raise(RevolveError(RevolveError.errCodes[self.info]))
        return retAction

    @property
    def advances(self):
        return declarations.revolve_getadvances(self.__r)

    @property
    def check(self):
        return declarations.revolve_getcheck(self.__r)

    @property
    def checkram(self):
        return declarations.revolve_getcheckram(self.__r)

    @property
    def checkrom(self):
        return declarations.revolve_getcheckrom(self.__r)

    @property
    def capo(self):
        return declarations.revolve_getcapo(self.__r)

    @property
    def fine(self):
        return declarations.revolve_getfine(self.__r)

    @property
    def oldcapo(self):
        return declarations.revolve_getoldcapo(self.__r)

    @property
    def where(self):
        return declarations.revolve_getwhere(self.__r)

    #@info.setter
    #def info(self, value):
    #    cdef int c_value = value
    #    declarations.revolve_setinfo(self.__r, c_value)

    def turn(self, final):
        cdef int c_final = final
        declarations.revolve_turn(self.__r, c_final)

    def __del__(self):
        declarations.revolve_destroy(self.__r)

cdef class memoryStorage(object):
    cdef __container
    cdef __snapshots
    cdef __snapshots_disk
    cdef __data_dim
    cdef __head_fwd
    cdef __rslt_fwd
    cdef __rslt_rev

    def __init__(self,snapshots,data_dim,snapshots_disk = None):
        try:
            container_dim = [snapshots]+list(data_dim)
        except TypeError, te:
            container_dim = [snapshots,data_dim]
        self.__container = np.zeros(container_dim)
        self.__snapshots = snapshots
        self.__snapshots_disk = snapshots_disk
        self.__data_dim = data_dim
        self.__head_fwd = np.zeros(data_dim)
        self.__rslt_fwd = np.zeros(data_dim)
        self.__rslt_rev = np.zeros(data_dim)
        if(snapshots_disk != None):
            warn("Multi-stage checkpointing not yet supported.")
        
    def load(self,idx):
        self.__head_fwd = self.__container[idx,:]

    def store(self,idx):
        self.__container[idx,:] = self.__head_fwd

    @property
    def forward_head(self):
        return self.__head_fwd

    @forward_head.setter
    def forward_head(self, value):
        self.__head_fwd = value

    @property
    def forward_result(self):
        return self.__rslt_fwd

    @forward_result.setter
    def forward_result(self, value):
        self.__rslt_fwd = value

    @property
    def reverse_result(self):
        return self.__rslt_rev

    @reverse_result.setter
    def reverse_result(self, value):
        self.__rslt_rev = value

    @property
    def snapshots(self):
        return self.__snapshots

    @property
    def snapshots_disk(self):
        return self.__snapshots_disk

    @property
    def dimensions(self):
        return self.__data_dimensions

cdef class checkpointExecutor(object):
    cdef storage
    cdef ckp
    cdef fwd_operator
    cdef rev_operator

    def __init__(self, fwd_operator, rev_operator, storage, timesteps = None):
        self.storage = storage
        if(storage.snapshots_disk != None):
            warn("Multi-stage checkpointing not yet supported.")
        if(timesteps == None):
            raise Exception("Online checkpointing not yet supported.")
        self.ckp = checkpointer(storage.snapshots, timesteps, storage.snapshots_disk)
        self.fwd_operator = fwd_operator
        self.rev_operator = rev_operator

    def apply(self):
        while(True):
            action = self.ckp.revolve()
            if(action == Action.advance):
                self.storage.forward_head = self.fwd_operator.apply(self.storage.forward_head, self.ckp.capo-self.ckp.oldcapo)
            elif(action == Action.takeshot):
                self.storage.store(self.ckp.check)
            elif(action == Action.restore):
                self.storage.load(self.ckp.check)
            elif(action == Action.firstrun):
                self.storage.forward_result = self.fwd_operator.apply(self.storage.forward_head, 1)
                self.storage.reverse_result = self.rev_operator.apply(self.storage.forward_head, self.storage.reverse_result, 1)
            elif(action == Action.youturn):
                self.storage.reverse_result = self.rev_operator.apply(self.storage.forward_head, self.storage.reverse_result, 1)
            elif(action == Action.terminate):
                return self.storage.forward_result, self.storage.reverse_result

