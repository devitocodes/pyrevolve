import pyrevolve.crevolve as pr

nSteps = 30
nSnaps = pr.adjust(nSteps)
c = pr.CRevolve(nSnaps, nSteps)

def forward(i):
    return i

def reverse(i,ib):
    return 1

snapStack = [None]*nSnaps

val = 2
valb = 1
while(True):
    action = c.revolve()
    if(action == pr.Action.advance):
        print("advance from t=%d to t=%d"%(c.oldcapo,c.capo))
        for i in range(c.oldcapo,c.capo):
            val = forward(val)
    elif(action == pr.Action.takeshot):
        print("store timestep %d in slot %d"%(c.capo,c.check))
        snapStack[c.check] = val
    elif(action == pr.Action.restore):
        print("load timestep %d from slot %d"%(c.capo,c.check))
        val = snapStack[c.check]
    elif(action == pr.Action.firstrun):
        print("start adjoint at time %d"%(c.capo))
        valb = valb + reverse(val,valb)
    elif(action == pr.Action.youturn):
        print("continue adjoint at time %d"%(c.capo))
        valb = reverse(val,valb)
    print(val,valb)
    if(action == pr.Action.terminate):
        break
