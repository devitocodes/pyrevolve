cimport declarations

def revolve(check,capo,fine,snaps,info):
    cdef int c_check = check
    cdef int c_capo  = capo
    cdef int c_fine  = fine
    cdef int c_snaps = snaps
    cdef int c_info  = info
    cdef declarations.action whatodo
    whatodo = declarations.revolve(&c_check, &c_capo, &c_fine, c_snaps, &c_info)
    return whatodo, c_check, c_capo, c_fine, c_info

def maxrange(ss, tt):
    cdef int c_ss = ss
    cdef int c_tt = tt
    return declarations.maxrange(c_ss, c_tt)

def adjust(steps):
    cdef int c_steps = steps
    cdef int adjsize
    adjsize = declarations.adjust(c_steps)
    return adjsize

def numforw(steps, snaps):
    cdef int c_steps = steps
    cdef int c_snaps = snaps
    return declarations.numforw(c_steps, c_snaps)

def expense(steps, snaps):
    cdef int c_steps = steps
    cdef int c_snaps = snaps
    return declarations.expense(c_steps, c_snaps)

def driver(steps, snaps, info):
    cdef int c_snaps = snaps
    cdef int c_steps = steps
    cdef int c_check = -1
    cdef int c_capo = 0
    cdef int c_fine = c_steps + c_capo
    cdef int c_info  = info
    cdef declarations.action whatodo

    while True:
        whatodo = declarations.revolve(&c_check, &c_capo, &c_fine, c_snaps, &c_info)
        if ((whatodo == declarations.takeshot) and (c_info > 1)):
            print(" takeshot at %6d"%c_capo)
        if ((whatodo == declarations.advance) and (c_info > 2)):
            print(" advance to %7d"%c_capo)
        if ((whatodo == declarations.firsturn) and (c_info > 2)):
            print(" firsturn at %6d"%c_capo)
        if ((whatodo == declarations.youturn) and (c_info > 2)):
            print(" youturn at %7d"%c_capo)
        if ((whatodo == declarations.restore) and (c_info > 2)):
            print(" restore at %7d"%c_capo)
        if (whatodo == declarations.error):
            print(" irregular termination of revolve")
            break
        if (whatodo == declarations.terminate):
            if(info == 10):
                print(" number of checkpoints stored exceeds checkup,")
                print(" increase constant 'checkup' and recompile")
            if(info == 11):
                print(" number of checkpoints stored = %d exceeds snaps = %d,"%(c_check+1,c_snaps))
                print(" ensure 'snaps' > 0 and increase initial 'fine'")
            if(info == 12):
                print(" error occurs in numforw")
            if(info == 13):
                print(" enhancement of 'fine', 'snaps' checkpoints stored,")
                print(" increase 'snaps'")
            if(info == 14):
                print(" number of snaps exceeds snapsup, ")
                print(" increase constant 'snapsup' and recompile")
            if(info == 15):
                print(" number of reps exceeds repsup, ")
                print(" increase constant 'repsup' and recompile")
            break
    return c_info
