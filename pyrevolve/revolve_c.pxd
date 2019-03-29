# cython: language_level=3
cdef extern from "../include/revolve_c.h":
    ctypedef enum CACTION:
        CACTION_ADVANCE,
        CACTION_TAKESHOT,
        CACTION_RESTORE,
        CACTION_FIRSTRUN,
        CACTION_YOUTURN,
        CACTION_TERMINATE,
        CACTION_ERROR

    ctypedef struct CRevolve:
        void *ptr

    cdef CRevolve revolve_create_offline(int st, int sn)
    cdef CRevolve revolve_create_online(int sn)
    cdef CRevolve revolve_create_multistage(int st, int sn, int sn_ram)
    cdef CACTION revolve(CRevolve r)
    cdef void revolve_destroy(CRevolve r)

    cdef int revolve_adjust(int steps)
    cdef int revolve_maxrange(int st, int sn)
    cdef int revolve_numforw(int steps, int snaps)
    cdef double revolve_expense(int steps, int snaps)
    cdef int revolve_getadvances(CRevolve r)
    cdef int revolve_getcheck(CRevolve r)
    cdef int revolve_getcheckram(CRevolve r)
    cdef int revolve_getcheckrom(CRevolve r)
    cdef int revolve_getcapo(CRevolve r)
    cdef int revolve_getfine(CRevolve r)
    cdef int revolve_getinfo(CRevolve r)
    cdef int revolve_getoldcapo(CRevolve r)
    cdef int revolve_getwhere(CRevolve r)
    #cdef void revolve_setinfo(CRevolve r, int inf)
    cdef void revolve_turn(CRevolve r, int final)
