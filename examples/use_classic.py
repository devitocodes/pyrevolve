import pyrevolve.crevolve as pr

nSteps = 30
nSnaps = pr.adjust(nSteps)
c = pr.CRevolve(nSnaps, nSteps)


def forward(nFrom, nTo, i):
    print((">"*(nTo-nFrom)).rjust(nTo))
    for it in range(nFrom, nTo):
        i = i+1
    return i


def reverse(i, ib):
    print("<".rjust(i+1))
    return ib+1


snapStack = [None]*nSnaps

val = 0
valb = 0
valF = None
while(True):
    action = c.revolve()
    if(action == pr.Action.advance):
        # print("advance from t=%d to t=%d" % (c.oldcapo, c.capo))
        val = forward(c.oldcapo, c.capo, val)
    elif(action == pr.Action.takeshot):
        # print("store timestep %d in slot %d" % (c.capo, c.check))
        snapStack[c.check] = val
    elif(action == pr.Action.restore):
        # print("load timestep %d from slot %d" % (c.capo, c.check))
        val = snapStack[c.check]
    elif(action == pr.Action.firstrun):
        # print("start adjoint at time %d" % (c.capo))
        valF = forward(nSteps-1, nSteps, val)
        print("u=%s" % valF)
        valb = valb + reverse(val, valb)
    elif(action == pr.Action.youturn):
        # print("continue adjoint from t=%d to t=%d" % (c.capo+1, c.capo))
        valb = reverse(val, valb)
    if(action == pr.Action.terminate):
        break

print("v=%s" % valb)
