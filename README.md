# Revolve
Revolve generates a optimal checkpointing schedules for adjoint computations. The algorithm is described in the excellent paper of Griewank & Walther [^1].

[^1]: Algorithm 799: Revolve: An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation

# pyrevolve
The pyrevolve library is a thin Python wrapper around the reference implementation in C that comes with the original paper. All fuction and argument names are identical to the original implementation.

# Installation and usage
To install, clone the repo and call

    python setup.py build_ext
    
To run, try something like

    import pyrevolve as pr
    nsteps = 30
    nsnaps = 5
    c = pr.checkpointer(5,30)
    c.revolve()

