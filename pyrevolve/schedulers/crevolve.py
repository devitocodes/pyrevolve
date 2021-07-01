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
        self.revolve = cr.CRevolve(n_checkpoints, n_timesteps, None)
        self.__revstart_action = None

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
