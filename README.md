# Revolve
Revolve generates a optimal checkpointing schedules for adjoint computations. The algorithm is described in the excellent paper of Griewank & Walther [^1].

[^1]: Algorithm 799: Revolve: An Implementation of Checkpointing for the Reverse or Adjoint Mode of Computational Differentiation

# pyrevolve
The pyrevolve library is a thin Python wrapper around the reference implementation in C that comes with the original paper. All fuction and argument names are identical to the original implementation.

# Installation and usage
To install, clone the repo and call

    python setup.py build_ext --inplace
    
To run, try something like

    import pyrevolve as pr
    nsteps = 30
    nsnaps = pr.adjust(nsteps)
    pr.driver(nsteps, nsnaps, 3)

# Contents

The pyrevolve package contains the functions from the reference implementation:

 - revolve (the main revolve algorithm)
 - maxrange (the maximum number of time steps achievable with a given number of snapshots and a given runtime overhead factor)
 - numforw (estimate the total number of forward evaluations that will be necessary for the given number of steps and snapshots)
 - expense (compute the overhead factor for the given number of steps and snapshots)
 - adjust (find a cost-effective number of checkpoints, which roughly minimises the product of memory usage and compute time. This makes sense if you pay for compute resources per time and per memory size.)
 - driver (example code that uses revolve and prints its actions for illustration)
 
A detailed documentation of these functions is in the original paper, or in the `c/src.c` file. 

