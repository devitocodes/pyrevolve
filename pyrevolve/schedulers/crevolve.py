try:
    import pyrevolve.crevolve as cr
except ImportError:
    import crevolve as cr
from .base import Action, Scheduler


class CAction(Action):
    """
    This class is an specialization of the Action
    base class for CRevolve scheduler
    """

    def __init__(self, action_type, capo, old_capo, ckp):
        super().__init__(action_type, capo, old_capo, ckp)

    def storageIndex(self):
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
        self.__revstart_action = None
        self.revolve = cr.CRevolve(n_checkpoints, n_timesteps, None)
        self.__oplist = []
        self.__stored_ckps = []
        self.__ratio = self.__calc_ratio()
        self.revolve = cr.CRevolve(n_checkpoints, n_timesteps, None)

    def __calc_ratio(self):
        fcomp = 0
        ca = self.next()
        self.__oplist.append(ca)
        while ca.type != Action.TERMINATE:
            if (ca.type == Action.ADVANCE) or (ca.type == Action.LASTFW):
                st = ca.old_capo
                end = ca.capo
                fcomp += (end-st)
            ca = self.next()
            self.__oplist.append(ca)

        return (fcomp/self.n_timesteps)

    @property
    def oplist(self):
        return self.__oplist

    def next(self):
        if self.__revstart_action is None:
            ca = CAction(
                action_type=self.translations[self.revolve.revolve()],
                capo=self.capo,
                old_capo=self.old_capo,
                ckp=self.cp_pointer,
            )
            if ca.type is Action.LASTFW:
                self.__revstart_action = CAction(
                    action_type=Action.REVSTART,
                    capo=self.capo,
                    old_capo=self.old_capo,
                    ckp=self.cp_pointer,
                )
            if ca.type is Action.TAKESHOT:
                self.__stored_ckps.append(ca.capo)
        else:
            ca = self.__revstart_action
            self.__revstart_action = None
        return ca

    @property
    def capo(self):
        return self.revolve.capo

    @property
    def old_capo(self):
        return self.revolve.oldcapo

    @property
    def cp_pointer(self):
        return self.revolve.check

    @property
    def ratio(self):
        return self.__ratio

    def storage(self, k):
        """Returns a list of all checkpoint keys stored at the k-th
        storage level. For CRevolve, k is always 0 """
        return self.__stored_ckps
