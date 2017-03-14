import pyrevolve as pr

# questions and todos
#  - the operators currently do not have the absolute time. Should this be provided? Or
#    should this be part of the data that is stored in the checkpoint?
#  - reverse operator is currently always called for a single step
#  - if multiple time steps are needed to resume forward sweep, they need to be packed
#    into the single 1D vector by the operator, and the Checkpointer object would not be
#    aware of overlapping data if steps are close together.
#  - only offline single-stage is supported at the moment
#  - how to create an abstract base class that forces apply() to follow the below
#    signatures?
#  - give user a good handle on verbosity

class ForwardOperator(object):
    def apply(self, val, nIter):
        for i in range(nIter):
            val = val + 1
            print("  pri %s"%val)
        return val

class ReverseOperator(object):
    def apply(self, val, valb, nIter):
        for i in range(nIter):
            valb = valb + 1
            print("  adj %s %s"%(valb,val+valb))
        return valb


nSteps = 30
nSnaps = pr.adjust(nSteps)
fwdo = ForwardOperator()
revo = ReverseOperator()
#TODO: need to be more general here. several fields. how to get their structure?
data_dim = (2,3)
storage = pr.memoryStorage(nSnaps,data_dim)
wrp = pr.checkpointExecutor(fwdo, revo, storage, nSteps)
print(wrp.apply())

