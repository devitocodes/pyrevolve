cimport declarations

import numpy as np
from enum import Enum
from warnings import warn
import copy

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

