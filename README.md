# Checkpointing

The adjoint computation of an unsteady nonlinear primal function requires the
full primal trajectory in reverse temporal order. Storing this can exceed the
available memory. In that case, Checkpointing can be used to store the state
only at carefully selected points in time. From there, the forward computation
can be restarted to recompute lost sections of the trajectory when they are
needed during the adjoint computation. This is always a tradeoff between memory
and runtime. The classic and provably optimal way to do this for a known number
of time steps is Revolve[^1], and there are other algorithms for optimal online
checkpointing if the number of steps is unknown a priori, or for multistage
checkpointing if there are multiple layers of storage, e.g. memory and hard
drive.

[^1]: Algorithm 799: Revolve: An Implementation of Checkpointing for the Reverse
or Adjoint Mode of Computational Differentiation

# pyrevolve

The pyrevolve library contains two parts: crevolve, which is a thin Python
wrapper around a previously published C++ implementation[^2], and pyrevolve
itself, which sits on top of crevolve and manages data and computation
management for the user.

The C++ files in this package are slightly modified to play more nicely with
Python, but the original is available from the link below.

[^2]: Revolve.cpp: http://www2.math.uni-paderborn.de/index.php?id=12067&L=1

# Installation

The crevolve wrapper requires cython, and the compilation of the C++ files
require that a C++ compiler is installed. To install pyrevolve, clone the repo
and call

    python setup.py build_ext --inplace
    
