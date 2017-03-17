import pyrevolve as pr
import numpy as np

# questions and todos
#  - reverse operator is currently always called for a single step
#  - if multiple time steps are needed to resume forward sweep, they need to be packed
#    into the single 1D vector by the operator, and the Checkpointer object would not be
#    aware of overlapping data if steps are close together.
#  - only offline single-stage is supported at the moment
#  - give user a good handle on verbosity

#access descriptor
# some magic (AD??) to determine which variables need to be recorded in the forward sweep
# active: variable depends on independents and influences dependents in differentiable way
#         do we need this at all as long as the user defines the adjoint operator?
# in: input
# out: output
# TBR: as a first approximation: in && out
# if in/out: needs to be stored in each iteration
# if in: needs to be stored at start
# if out: needs to be stored in fwd_rslt at end
# if active: needs
# in AD, TBR (to-be-recorded) is a more precise concept and removes variables that have no
#    non-linear influence on the result, and hence no influence on the adjoint. This
#    requires an analysis of the forward operator, and often a custom forward operator that
#    computes only those variables that are actually needed for the reverse sweep. Maybe
#    we'll have that in the future.

class Checkpoint(object):
    data = None

    def __init__(self,dataObjects):
        self.data = dataObjects

    def copy(self):
        cp_data = {}
        for i in self.data:
            cp_data[i] = self.data[i].copy()
        cp = Checkpoint(cp_data)
        return cp

    def copyFrom(self,other):
        for i in other.data:
            self.data[i].data = other.data[i].data.copy()

    @property
    def nbytes(self):
        size = 0
        for i in self.data:
            size = size+self.data[i].nbytes
        return size


class memoryStorage(object):
    __container = None
    __snapshots = None
    __snapshots_disk = None
    __head_fwd = None
    __rslt_fwd = None
    __rslt_rev = None

    def __init__(self,snapshots):
        self.__snapshots = snapshots
        self.__snapshots_disk = None
        self.__container = snapshots*[None]
        if(self.__snapshots_disk != None):
            warn("Multi-stage checkpointing not yet supported.")
    
    def init(self,ivals):
        self.__head_fwd = Checkpoint(ivals)
 
    def load(self,idx):
        self.__head_fwd.copyFrom(self.__container[idx])

    def store(self,idx):
        self.__container[idx] = Checkpoint(self.__head_fwd.data).copy()

    def turn(self):
        self.__rslt_fwd = self.__head_fwd.copy()

    def finalise(self):
        self.__head_fwd.copyFrom(self.__rslt_fwd)

    def prints(self):
        for i in range(self.__snapshots):
            try:
                allData = self.__container[i].data
                for var in allData:
                    print(allData[var].data),
                    print(self.__head_fwd.data[var].data)
            except:
                pass
 
    @property
    def snapshots(self):
        return self.__snapshots

    @property
    def snapshots_disk(self):
        return self.__snapshots_disk

class checkpointExecutor(object):
    storage = None
    ckp = None
    fwd_operator_onestep = None
    fwd_operator_nsteps = None
    rev_operator = None
    needs_ckp = None
    needs_arg = None
    adj_needs_arg = None

    def __init__(self, fwd_operator_onestep, fwd_operator_nsteps, rev_operator, storage, timesteps = None):
        self.storage = storage
        if(storage.snapshots_disk != None):
            warn("Multi-stage checkpointing not yet supported.")
        if(timesteps == None):
            raise Exception("Online checkpointing not yet supported.")
        self.ckp = pr.checkpointer(storage.snapshots, timesteps, storage.snapshots_disk)
        self.fwd_operator_onestep = fwd_operator_onestep
        self.fwd_operator_nsteps = fwd_operator_nsteps
        fwd_operator = fwd_operator_onestep
        self.rev_operator = rev_operator
        self.needs_ckp = set(fwd_operator.input_params).intersection(fwd_operator.output_params)
        self.needs_arg = set(fwd_operator.input_params).union(fwd_operator.output_params)
        self.adj_needs_arg = set(rev_operator.input_params).union(rev_operator.output_params)

    def apply(self,**kwargs):
        working_data = {}
        adj_working_data = {}
        checkpoint_data = {}
        for arg in self.needs_arg:
            working_data[arg] = kwargs[arg]
        for arg in self.adj_needs_arg:
            adj_working_data[arg] = kwargs[arg]
        for arg in self.needs_ckp:
            checkpoint_data[arg] = kwargs[arg]
        self.storage.init(checkpoint_data)

        while(True):
            action = self.ckp.revolve()
            if(action == pr.Action.advance):
                nSteps = self.ckp.capo-self.ckp.oldcapo
                if(nSteps < self.fwd_operator_nsteps.min_steps):
                    self.fwd_operator_onestep.apply(nSteps, **working_data)
                else:
                    self.fwd_operator_nsteps.apply(nSteps, **working_data)
            elif(action == pr.Action.takeshot):
                self.storage.store(self.ckp.check)
            elif(action == pr.Action.restore):
                self.storage.load(self.ckp.check)
            elif(action == pr.Action.firstrun):
                self.fwd_operator_onestep.apply(1, **working_data)
                self.storage.turn()
                self.rev_operator.apply(1, **adj_working_data)
            elif(action == pr.Action.youturn):
                self.rev_operator.apply(1, **adj_working_data)
            elif(action == pr.Action.terminate):
                self.storage.finalise()
                break


class Data(object):
    __storage = None
    __data_dim = None
    
    def __init__(self, data_dim):
        self.__storage = np.zeros(data_dim)
        self.__data_dim = data_dim

    @property
    def nbytes(self):
        return self.__storage.nbytes

    @property
    def data(self):
        return self.__storage

    @data.setter
    def data(self,value):
        self.__storage = value

    def copy(self):
        d = Data(self.__data_dim)
        d.data = self.data.copy()
        return d

class ForwardOperator(object):
    min_steps = 1

    def apply(self, nIter, u, m):
        for i in range(nIter):
            u.data = u.data + m.data
            print("  pri %s %s %d"%(u.data,m.data,self.min_steps))
        return u

    @property
    def input_params(self):
        return ("u","m")

    @property
    def output_params(self):
        return ("u")
        

class ReverseOperator(object):
    def apply(self, nIter, u, m, v):
        for i in range(nIter):
            v.data = v.data + m.data
            print("  adj %s %s"%(v.data,u.data+v.data))
        return v

    @property
    def input_params(self):
        return ("u","m","v")

    @property
    def output_params(self):
        return ("v")



#class ForwardBuffer(object):
#    # this would be like checkpoint, but with extra space for working copies
#class ReverseBuffer(object):
#    # this would be like checkpoint, but with adjoint data

nSteps = 30
nSnaps = pr.adjust(nSteps)
fwdo = ForwardOperator()
fwdn = ForwardOperator()
fwdn.min_steps = 5
revo = ReverseOperator()
u = Data((1,1))
m = Data((1,1))
m.data = 1
v = Data((1,1))
storage = memoryStorage(nSnaps)
wrp = checkpointExecutor(fwdo, fwdn, revo, storage, nSteps)
wrp.apply(u=u,m=m,v=v)
print("u=%s"%u.data)
print("v=%s"%v.data)

