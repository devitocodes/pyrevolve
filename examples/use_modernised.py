import pyrevolve as pr
import numpy as np

class Symbol(object):
    def __init__(self, data_dim):
        self.__storage = np.zeros(data_dim)
        self.__data_dim = data_dim

    #@property
    #def nbytes(self):
    #    return self.__storage.nbytes

    @property
    def data(self):
        return self.__storage

    @data.setter
    def data(self,value):
        self.__storage = value

    def copy(self):
        d = Symbol(self.__data_dim)
        d.data = self.data.copy()
        return d

class ForwardOperator(object):
    def apply(self, nIter, u, m):
        for i in range(nIter):
            u.data = u.data + m.data
            print("  pri %s %d"%(u.data,nIter))
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
            print("  adj %s"%(v.data))
        return v

    @property
    def input_params(self):
        return ("u","m","v")

    @property
    def output_params(self):
        return ("v")





nSteps = 30
fwdo = ForwardOperator()
revo = ReverseOperator()
u = Symbol((1,1))
m = Symbol((1,1))
m.data = 1
v = Symbol((1,1))
wrp = pr.Revolver(fwdo, revo, nSteps)
wrp.apply(u=u,m=m,v=v)
print("u=%s"%u.data)
print("v=%s"%v.data)
    
