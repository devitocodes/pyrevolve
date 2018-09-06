from checkpointer import schedule_nodes, Chain
import util 
try:
    import pyrevolve.crevolve as cr
except ImportError:
    import crevolve as cr

class Action(object):
    ADVANCE = 0
    TAKESHOT = 1
    RESTORE = 2
    LASTFW = 3
    REVERSE = 4
    CPDEL = 5
    TERMINATE = 6

    def __init__(self, action_type):
        self.type = action_type



class CRevolve(object):
    translations = {
        cr.Action.advance: Action.ADVANCE,
        cr.Action.takeshot: Action.TAKESHOT,
        cr.Action.restore: Action.RESTORE,
        cr.Action.firstrun: Action.LASTFW,
        cr.Action.youturn: Action.REVERSE,
        cr.Action.terminate: Action.TERMINATE
        }
    def __init__(self, number_checkpoints, number_timesteps):
        self.revolve = cr.CRevolve(number_checkpoints, number_timesteps, None)

    def next(self):
        return Action(self.translations[self.revolve.revolve()])

    @property
    def capo(self):
        return self.revolve.capo

    @property
    def old_capo(self):
        return self.revolve.oldcapo

    @property
    def cp_pointer(self):
        return self.revolve.check

        

class PythonRevolve(object):
    translations = { util.Action.FW: Action.ADVANCE,
                        util.Action.BW: Action.REVERSE,
                        util.Action.CPSAVE: Action.TAKESHOT,
                        util.Action.CPLOAD: Action.RESTORE,
                        util.Action.CPDEL: Action.CPDEL,
                        util.Action.LASTFW: Action.LASTFW}
    c = 1
    d = 10
    def __init__(self, number_checkpoints, number_timesteps):
        nodes = [{'input_size': self.d, 'output_size': self.d, 'compute_cost': self.c}
                     for i in range(number_timesteps)]
        memory = self.d * number_timesteps
        self.schedule = PythonRevolve.translate(schedule_nodes(nodes, memory))
        print(self.schedule)
        self.action_pointer = -1
        self.next_switch = -1
        self.cp_pointer = -1

    def next(self):
        self.action_pointer += 1
        self.next_switch = self.action_pointer
        next_action = self.schedule[self.action_pointer]
        while self.schedule[self.next_switch].type == next_action.type:
            self.next_switch += 1
        if next_action.type == Action.TAKESHOT:
            self.cp_pointer += 1
        return next_action

    @property
    def capo(self):
        return self.schedule[self.next_switch].index

    @property
    def old_capo(self):
        return self.schedule[self.action_pointer].index

    @staticmethod
    def translate(schedule):
        for a in schedule.actions:
            a.type == PythonRevolve.translations[a.type]
        return schedule

Revolve = CRevolve


class CompressionRevolve(CRevolve):
    def __init__(self, number_checkpoints, number_timesteps):
        self.revolve = cr.CRevolve(number_checkpoints, number_timesteps, None)

    def next(self):
        return Action(self.translations[self.revolve.revolve()])

    @property
    def capo(self):
        return self.revolve.capo

    @property
    def old_capo(self):
        return self.revolve.oldcapo

    @property
    def cp_pointer(self):
        return self.revolve.check
