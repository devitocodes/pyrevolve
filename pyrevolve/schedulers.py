try:
    import pyrevolve.crevolve as cr
except ImportError:
    import crevolve as cr
import json


class Action(object):
    ADVANCE = 0
    TAKESHOT = 1
    RESTORE = 2
    LASTFW = 3
    REVERSE = 4
    CPDEL = 5
    TERMINATE = 6

    type_names = {ADVANCE: 'ADVANCE', TAKESHOT: 'TAKESHOT', RESTORE: 'RESTORE',
                  LASTFW: 'LASTFW', REVERSE: 'REVERSE', CPDEL: 'CPDEL',
                  TERMINATE: 'TERMINATE'}

    def __init__(self, action_type, capo, oldcapo, ckp):
        self.type = action_type
        self.capo = capo
        self.oldcapo = oldcapo
        self.ckp = ckp

    def __repr__(self):
        return json.dumps(dict({'type': self.type_names[self.type],
                                'capo': self.capo,
                                'oldcapo': self.oldcapo, 'ckp': self.ckp}))


class CRevolve(object):
    translations = {cr.Action.advance: Action.ADVANCE,
                    cr.Action.takeshot: Action.TAKESHOT,
                    cr.Action.restore: Action.RESTORE,
                    cr.Action.firstrun: Action.LASTFW,
                    cr.Action.youturn: Action.REVERSE,
                    cr.Action.terminate: Action.TERMINATE}

    def __init__(self, number_checkpoints, number_timesteps):
        self.revolve = cr.CRevolve(number_checkpoints, number_timesteps, None)

    def next(self):
        return Action(self.translations[self.revolve.revolve()], self.capo,
                      self.old_capo, self.cp_pointer)

    @property
    def capo(self):
        return self.revolve.capo

    @property
    def old_capo(self):
        return self.revolve.oldcapo

    @property
    def cp_pointer(self):
        return self.revolve.check


Revolve = CRevolve
