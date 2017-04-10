import crevolve as cr


class Checkpoint(object):
    """Holds a list of symbol objects that hold data. Each checkpoint object
    represents a state. This is usually only used internally by pyrevolve."""

    def __init__(self, symbols):
        """Intialise a checkpoint object. Upon initialisation, a checkpoint
        stores only a reference to the symbols that are passed into it."""
        self.symbols = symbols

    def copy(self):
        """Return a new checkpoint that holds new symbols with a deep-copy of
        the data contained in the current checkpoint."""
        cp_data = {}
        for i in self.symbols:
            cp_data[i] = self.symbols[i].copy()
        cp = Checkpoint(cp_data)
        return cp

    def restore_from(self, other):
        """Deep-copy the data contained in the symbols of the other checkpoint
        into the Symbols held by the current checkpoint."""
        for i in other.symbols:
            self.symbols[i].data = other.symbols[i].data.copy()

    @property
    def nbytes(self):
        """The memory consumption of the data contained in this checkpoint."""
        size = 0
        for i in self.symbols:
            size = size+self.symbols[i].nbytes
        return size


class MemoryStorage(object):
    """In the classic Revolve case, this holds a stack of Checkpoint objects.
    This is usually only used internally by pyrevolve.
    Future work: For more advanced cases, this may have to hold several stacks
    (one for memory, one for HDD, one for...) or a vector of Checkpoint objects
    to support random access for online checkpointing.
    """

    def __init__(self, n_snapshots):
        """Initialise storage space for the given number of snapshots."""
        self.__n_snapshots = n_snapshots
        self.__container = n_snapshots*[None]

    def register_symbols(self, ivals):
        """Pass in references to all symbols that need to be checkpointed."""
        self.__head_fwd = Checkpoint(ivals)

    def store(self, idx):
        """Store the data contained in all symbols into slot idx."""
        self.__container[idx] = Checkpoint(self.__head_fwd.symbols).copy()

    def load(self, idx):
        """Load the data from slot idx back into the symbols."""
        self.__head_fwd.restore_from(self.__container[idx])

    def turn(self):
        """Store the data contained in all symbols into the special slot for
        the final result."""
        self.__rslt_fwd = self.__head_fwd.copy()

    def finalise(self):
        """Load the data from the special final result slot back into the
        symbols."""
        self.__head_fwd.restore_from(self.__rslt_fwd)

    def __str__(self):
        for i in range(self.__n_snapshots):
            try:
                allData = self.__container[i].data
                for var in allData:
                    print(allData[var].data),
                    print(self.__head_fwd.data[var].data)
            except:
                pass

    @property
    def n_snapshots(self):
        return self.__n_snapshots


class Revolver(object):
    """
    This should be the only user-facing class in here. It manages the
    interaction between the operators passed by the user, and the data storage.

    What needs to be checkpointed? At the moment, pyrevolve requests from the
    forward and reverse operators a list of their inputs and outputs. The
    variables that are checkpointed are then given by the intersection of
    forward_operator.input and forward_operator.output.
    The rationale is that the forward computation needs to have all its input
    variables to resume the computation, but of those, only the ones that are
    modified over time need to be checkpointed.

    Todo:
        * Reverse operator is always called for a single step. Change this.
        * Avoid redundant data stores if higher-order stencils save multiple
          time steps, and checkpoints are close together.
        * Only offline single-stage is supported at the moment.
        * Give users a good handle on verbosity.

    Nice to have, but too far away for a Todo:
        In Algorithmic Differentiation, TBR (to-be-recorded) is a better way
        to decide which variables need to be stored. Only variables that have
        a nonlinear influence on the result are important, and only those parts
        of the forward operator that compute such variables need to be
        repeated. This would require a full analysis of the forward operator,
        and the creation of a simplified forward operator for checkpointing,
        that has all computations with no effect on the adjoint gradients
        removed.
    """

    def __init__(self, fwd_operator, rev_operator, timesteps=None):
        """Initialise checkpointer for a given forward- and reverse operator, a
        given number of time steps, and a given storage strategy. The number of
        time steps must currently be provided explicitly, and the storage must
        be the single-staged memory storage."""
        if(timesteps is None):
            raise Exception("Online checkpointing not yet supported. Specify \
                              number of time steps!")
        self.storage = MemoryStorage(cr.adjust(timesteps))
        storage_disk = None  # this is not yet supported
        # We use the crevolve wrapper around the C++ Revolve library.
        self.ckp = cr.CRevolve(self.storage.n_snapshots,
                               timesteps, storage_disk)

        self.fwd_operator = fwd_operator
        self.rev_operator = rev_operator
        fwd_in = fwd_operator.input_params
        fwd_out = fwd_operator.output_params
        rev_in = rev_operator.input_params
        rev_out = rev_operator.output_params
        # Symbols that need to be checkpointed
        self.needs_ckp = set(fwd_in).intersection(fwd_out)
        # Symbols that need to be passed to the forward operator as an argument
        self.needs_arg = set(fwd_in).union(fwd_out)
        # Symbols that need to be passed to the reverse operator as an argument
        self.adj_needs_arg = set(rev_in).union(rev_out)

    def apply(self, **kwargs):
        """Executes the forward and reverse computations. All arguments that are
        required by either the forward or reverse operator must be passed."""

        # workspace, containing symbols that the forward operator works with.
        working_data = {}
        # workspace for adjoint operator.
        adj_working_data = {}
        # references to the symbols that need to be checkpointed.
        checkpoint_data = {}
        for arg in self.needs_arg:
            working_data[arg] = kwargs[arg]
        for arg in self.adj_needs_arg:
            adj_working_data[arg] = kwargs[arg]
        for arg in self.needs_ckp:
            checkpoint_data[arg] = kwargs[arg]
        self.storage.register_symbols(checkpoint_data)

        while(True):
            # ask Revolve what to do next.
            action = self.ckp.revolve()
            if(action == cr.Action.advance):
                # advance forward computation by nSteps
                nSteps = self.ckp.capo-self.ckp.oldcapo
                self.fwd_operator.apply(nSteps, **working_data)
            elif(action == cr.Action.takeshot):
                # take a snapshot: copy from workspace into storage
                self.storage.store(self.ckp.check)
            elif(action == cr.Action.restore):
                # restore a snapshot: copy from storage into workspace
                self.storage.load(self.ckp.check)
            elif(action == cr.Action.firstrun):
                self.fwd_operator.apply(1, **working_data)
                # after the last forward iteration, the primal results in
                # workspace are copied into final_data, before working array is
                # reset to a previous state to restart computations.
                self.storage.turn()
                self.rev_operator.apply(1, **adj_working_data)
            elif(action == cr.Action.youturn):
                # advance adjoint computation by a single step
                self.rev_operator.apply(1, **adj_working_data)
            elif(action == cr.Action.terminate):
                # finalise: copy primal results from final_data back into
                # workspace, which additionally alsready contains final adjoint
                # results. Return.
                self.storage.finalise()
                break
