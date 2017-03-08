cdef extern from "../c/revolve.h":
    cdef enum action:
        advance,
        takeshot,
        restore,
        firsturn,
        youturn,
        terminate,
        error

    cdef action revolve(int *check, int *capo, int *fine, int snaps, int *info)
    cdef int maxrange(int ss, int tt)
    cdef int adjust(int steps)
    cdef int numforw(int steps, int snaps)
    cdef double expense(int steps, int snaps)
