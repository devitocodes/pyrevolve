"""
This module includes classes and function definitions
provided as part of the H-Revolve python implementation.
The original implementation was developed by authors
Julien Herrmann and Guillaume Aupy and is orignally
distributed under GNU GPL v.3 license terms.
The original H-Revolve source code can be found in the
following Gitlab repository:

Original H-Revolve source-code:
https://gitlab.inria.fr/adjoint-computation/H-Revolve/tree/master

The H-Revolve library is described in detail in the
paper "H-Revolve: A Framework for Adjoint Computation on
Synchronous Hierarchical Platforms" by Herrmann and Pallez [1].

Some minor modifications where made to adapt this libray for the
PyRevolve API.

Authors: Julien Herrmann, Guillaume Aupy

Refs:
[1] Herrmann, Pallez, "H-Revolve: A Framework for
    Adjoint Computation on Synchronous Hierarchical
    Platforms", ACM Transactions on Mathematical
    Software  46(2), 2020.
"""

official_names = {
    "Forward": "F",
    "Forwards": "F",
    "Backward": "B",
    "Checkpoint": "C",
    "Read_disk": "RD",
    "Write_disk": "WD",
    "Read_memory": "RM",
    "Write_memory": "WM",
    "Discard_disk": "DD",
    "Discard_memory": "DM",
    "Read": "R",
    "Write": "W",
    "Discard": "D",
    "Forward_branch": "F",
    "Forwards_branch": "F",
    "Backward_branch": "B",
    "Turn": "T",
    "Discard_branch": "DB",
    "Checkpoint_branch": "C",
}


def argmin(l):
    # Return the last argmin (1-based)
    index = 0
    m = l[0]
    for i in range(len(l)):
        if l[i] <= m:
            index = i
            m = l[i]
    return 1 + index


def get_hopt_table(lmax, architecture, uf=1, ub=1):
    """Compute the HOpt table for architecture and l=0...lmax
    This computation uses a dynamic program
    uf: Cost of the forward steps (default: 1)
    ub: Cost of the backward steps (default: 1)
    """
    K = architecture.nblevels
    cvect = architecture.sizes
    wvect = architecture.wd
    rvect = architecture.rd
    assert len(wvect) == K and len(rvect) == K and len(cvect) == K
    opt = [[[float("inf")] * (cvect[i] + 1) for _ in range(lmax + 1)] for i in range(K)]
    optp = [
        [[float("inf")] * (cvect[i] + 1) for _ in range(lmax + 1)] for i in range(K)
    ]
    # Initialize borders of the table
    for k in range(K):
        mmax = cvect[k]
        for m in range(mmax + 1):
            opt[k][0][m] = ub
            optp[k][0][m] = ub
        for m in range(mmax + 1):
            if (m == 0) and (k == 0):
                continue
            optp[k][1][m] = uf + 2 * ub + rvect[0]
            opt[k][1][m] = wvect[0] + optp[k][1][m]
    # Fill K = 0
    mmax = cvect[0]
    for l in range(2, lmax + 1):
        optp[0][l][1] = (l + 1) * ub + l * (l + 1) / 2 * uf + l * rvect[0]
        opt[0][l][1] = wvect[0] + optp[0][l][1]
    for m in range(2, mmax + 1):
        for l in range(2, lmax + 1):
            optp[0][l][m] = min(
                [
                    j * uf + opt[0][l - j][m - 1] + rvect[0] + optp[0][j - 1][m]
                    for j in range(1, l)
                ]
                + [optp[0][l][1]]
            )
            opt[0][l][m] = wvect[0] + optp[0][l][m]
    # Fill K > 0
    for k in range(1, K):
        mmax = cvect[k]
        for l in range(2, lmax + 1):
            opt[k][l][0] = opt[k - 1][l][cvect[k - 1]]
        for m in range(1, mmax + 1):
            for l in range(1, lmax + 1):
                optp[k][l][m] = min(
                    [opt[k - 1][l][cvect[k - 1]]]
                    + [
                        j * uf + opt[k][l - j][m - 1] + rvect[k] + optp[k][j - 1][m]
                        for j in range(1, l)
                    ]
                )
                opt[k][l][m] = min(
                    opt[k - 1][l][cvect[k - 1]], wvect[k] + optp[k][l][m]
                )

    return (optp, opt)


class Function:
    def __init__(self, name, l, index):
        self.name = name
        self.l = l
        self.index = index

    def __repr__(self):
        if self.name == "HRevolve" or self.name == "HRevolve_aux":
            return (
                self.name
                + "_"
                + str(self.index[0])
                + "("
                + str(self.l)
                + ", "
                + str(self.index[1])
                + ")"
            )
        else:
            return self.name + "(" + str(self.l) + ", " + str(self.index) + ")"


class Sequence:
    def __init__(self, function, architecture, uf=1, ub=1, up=1):
        self.sequence = []  # List of Operation and Sequence
        self.function = function  # Description the function (name and parameters)
        self.makespan = 0  # Makespan to be updated
        self.architecture = architecture
        self.nblevels = architecture.nblevels
        self.uf = uf  # forward cost (default=1)
        self.ub = ub  # backward cost (default=1)
        self.up = up  # turn cost (default=1)
        if self.function.name == "HRevolve" or self.function.name == "HRevolve_aux":
            self.storage = [
                [] for _ in range(self.nblevels)
            ]  # List of list of checkpoints in hierarchical storage
        else:
            self.memory = []  # List of memory checkpoints
            self.disk = []  # List of disk checkpoints
        self.type = "Function"

    @property
    def size(self):
        return len(self.sequence)

    def concat_sequence(self, concat=0):
        l = []
        for x in self.sequence:
            if x.__class__.__name__ == "Operation":
                l.append(x)
            elif x.__class__.__name__ == "Sequence":
                if concat == 0:
                    l += x.concat_sequence(concat=concat)
                elif concat == 1:
                    if x.function.name == "Revolve":
                        l.append(x.function)
                    else:
                        l += x.concat_sequence(concat=concat)
                elif concat == 2:
                    if x.function.name == "Revolve" or x.function.name == "1D-Revolve":
                        l.append(x.function)
                    else:
                        l += x.concat_sequence(concat=concat)
                else:
                    raise ValueError("Unknown concat value: " + str(concat))
            else:
                raise ValueError("Unknown class name: " + x.__class__.__name__)
        return l

    def concat_sequence_hierarchic(self, concat=0):
        l = []
        for x in self.sequence:
            if x.__class__.__name__ == "Operation":
                l.append(x)
            elif x.__class__.__name__ == "Sequence":
                if concat == 0:
                    l += x.concat_sequence_hierarchic(concat=concat)
                elif (
                    x.function.name == "HRevolve" and x.function.index[0] <= concat - 1
                ):
                    l.append(x.function)
                else:
                    l += x.concat_sequence_hierarchic(concat=concat)
            else:
                raise ValueError("Unknown class name: " + x.__class__.__name__)
        return l

    def insert(self, operation):
        self.sequence.append(operation)
        self.makespan += operation.cost(self.architecture, self.uf, self.ub, self.up)
        if operation.type == "Write_memory":
            self.memory.append(operation.index)
        if operation.type == "Write_disk":
            self.disk.append(operation.index)
        if operation.type == "Checkpoint":
            self.memory.append(operation.index)
        if operation.type == "Write":
            self.storage[operation.index[0]].append(operation.index[1])
        if operation.type == "Checkpoint_branch":
            self.memory.append((operation.index[0], operation.index[1]))

    def remove(self, operation_index):
        self.makespan -= self.sequence[operation_index].cost(
            self.architecture, self.uf, self.ub, self.up
        )
        if self.sequence[operation_index].type == "Write_memory":
            self.memory.remove(self.sequence[operation_index].index)
        if self.sequence[operation_index].type == "Write_disk":
            self.disk.remove(self.sequence[operation_index].index)
        if self.sequence[operation_index].type == "Write":
            self.storage[self.sequence[operation_index].index[0]].remove(
                self.sequence[operation_index].index[1]
            )
        if self.sequence[operation_index].type == "Checkpoint":
            self.memory.remove(self.sequence[operation_index].index)
        del self.sequence[operation_index]

    def insert_sequence(self, sequence):
        self.sequence.append(sequence)
        self.makespan += sequence.makespan
        if self.function.name == "HRevolve" or self.function.name == "HRevolve_aux":
            for i in range(len(self.storage)):
                self.storage[i] += sequence.storage[i]
        else:
            self.memory += sequence.memory
            self.disk += sequence.disk

    def shift(self, size, branch=-1):
        for x in self.sequence:
            x.shift(size, branch=branch)
        if self.function.name == "HRevolve" or self.function.name == "HRevolve_aux":
            for i in range(len(self.storage)):
                self.storage[i] = [x + size for x in self.storage[i]]
        else:
            self.memory = [
                x + size
                if type(x) is int
                else (x[0], x[1] + size)
                if x[0] == branch
                else (x[0], x[1])
                for x in self.memory
            ]
            self.disk = [
                x + size
                if type(x) is int
                else (x[0], x[1] + size)
                if x[0] == branch
                else (x[0], x[1])
                for x in self.disk
            ]
        return self

    def remove_useless_wm(self, K=-1):
        if len(self.sequence) > 0:
            if (
                self.sequence[0].type == "Write_memory"
                or self.sequence[0].type == "Checkpoint"
            ):
                self.remove(0)
                return self
        if len(self.sequence) > 0:
            if self.sequence[0].type == "Write":
                if self.sequence[0].index[0] == K:
                    self.remove(0)
                    return self
        return self

    def remove_last_discard(self):
        if self.sequence[-1].type == "Function":
            self.sequence[-1].remove_last_discard()
        if self.sequence[-1].type in [
            "Discard_memory",
            "Discard_disk",
            "Discard",
            "Discard_branch",
        ]:
            self.remove(-1)

    def opcount(self, sz=0):
        """Returns the total number of operations
        for this sequence
        """
        for seq in self.sequence:
            if seq.type == "Function":
                sz = seq.opcount(sz)
            else:
                sz += 1
        return sz

    def get_flat_op_list(self, i=0):
        op_list = []
        for seq in self.sequence:
            if seq.type == "Function":
                op_list += seq.get_flat_op_list(i+1)
            else:
                op_list.append(seq)
        return op_list

    def first_operation(self):
        if self.sequence[0].type == "Function":
            return self.sequence[0].first_operation()
        else:
            return self.sequence[0]

    def next_operation(self, i):
        if self.sequence[i + 1].type == "Function":
            return self.sequence[i + 1].first_operation()
        else:
            return self.sequence[i + 1]

    def convert_old_to_branch(self, index):
        for (i, x) in enumerate(self.memory):
            if type(x) is int:
                self.memory[i] = (index, x)
        to_remove = []
        for (i, op) in enumerate(self.sequence):
            if op.type == "Function":
                self.sequence[i] = self.sequence[i].convert_old_to_branch(index)
            elif op.type == "Forward":
                op.type = "Forward_branch"
                op.index = [index, op.index]
            elif op.type == "Forwards":
                op.type = "Forwards_branch"
                op.index = [index] + op.index
            elif op.type == "Backward":
                op.type = "Backward_branch"
                op.index = [index, op.index]
            elif op.type == "Read":
                if self.next_operation(i).type == "Backward":
                    to_remove.append(i)
                else:
                    op.type = "Checkpoint_branch"
                    op.index = [index, op.index]
            elif op.type == "Write":
                op.type = "Checkpoint_branch"
                op.index = [index, op.index]
            elif op.type == "Discard":
                to_remove.append(i)
            elif op.type == "Read_memory":
                if self.next_operation(i).type == "Backward":
                    to_remove.append(i)
                else:
                    op.type = "Checkpoint_branch"
                    op.index = [index, op.index]
            elif op.type == "Write_memory":
                op.type = "Checkpoint_branch"
                op.index = [index, op.index]
            elif op.type == "Discard_memory":
                to_remove.append(i)
            elif op.type in ["Read_disk", "Write_disk", "Discard_disk"]:
                ValueError(
                    "Cannot use convert_old_to_branch on sequences \
                        from two-memory architecture"
                )
            else:
                ValueError("Unknown data type %s in convert_old_to_branch" % op.type)
        for (i, index) in enumerate(to_remove):
            self.remove(index - i)
        return self

    def convert_new_to_branch(self, index):
        for (i, x) in enumerate(self.memory):
            if type(x) is int:
                self.memory[i] = (index, x)
        to_remove = []
        for (i, op) in enumerate(self.sequence):
            if op.type == "Function":
                self.sequence[i] = self.sequence[i].convert_new_to_branch(index)
            elif op.type == "Forward":
                op.type = "Forward_branch"
                op.index = [index, op.index]
            elif op.type == "Forwards":
                op.type = "Forwards_branch"
                op.index = [index] + op.index
            elif op.type == "Backward":
                op.type = "Backward_branch"
                op.index = [index, op.index]
            elif op.type == "Checkpoint":
                op.type = "Checkpoint_branch"
                op.index = [index, op.index]
            elif op.type in [
                "Forward_branch",
                "Forwards_branch",
                "Turn",
                "Discard_branch",
                "Checkpoint",
                "Backward_branch",
            ]:
                continue
            elif op.type in ["Read_disk", "Write_disk", "Discard_disk"]:
                ValueError(
                    "Cannot use convert_new_to_branch on sequences \
                        from two-memory architecture"
                )
            else:
                ValueError("Unknown data type %s in convert_new_to_branch" % op.type)
        for (i, index) in enumerate(to_remove):
            self.remove(index - i)
        return self


class Operation:
    def __init__(self, operation_type, operation_index):
        if operation_type not in official_names:
            raise ValueError("Unreconized operation name: " + operation_type)
        self.type = operation_type
        self.index = operation_index
        if self.type == "Forwards" and self.index[0] == self.index[1]:
            self.type = "Forward"
            self.index = self.index[0]
        if self.type == "Forwards_branch" and self.index[1] == self.index[2]:
            self.type = "Forward_branch"
            self.index = [self.index[0], self.index[1]]

    def __repr__(self):
        if self.index is None:
            return official_names[self.type]
        if type(self.index) is int:
            return official_names[self.type] + "_" + str(self.index)
        elif type(self.index) is list:
            if self.type == "Forwards":
                return (
                    official_names[self.type]
                    + "_"
                    + str(self.index[0])
                    + "->"
                    + str(self.index[1])
                )
            elif self.type == "Forwards_branch":
                return (
                    official_names[self.type]
                    + "^"
                    + str(self.index[0])
                    + "_"
                    + str(self.index[1])
                    + "->"
                    + str(self.index[2])
                )
            else:
                return (
                    official_names[self.type]
                    + "^"
                    + str(self.index[0])
                    + "_"
                    + str(self.index[1])
                )

    def cost(self, architecture, uf=1, ub=1, up=1):
        if self.type == "Forward":
            return uf
        if self.type == "Forwards":
            return (self.index[1] - self.index[0] + 1) * uf
        if self.type == "Backward":
            return ub
        if self.type == "Checkpoint":
            return 0
        if self.type == "Read_disk":
            return architecture.rd
        if self.type == "Write_disk":
            return architecture.wd
        if self.type == "Read_memory":
            return 0
        if self.type == "Write_memory":
            return 0
        if self.type == "Discard_disk":
            return 0
        if self.type == "Discard_memory":
            return 0
        if self.type == "Read":
            return architecture.rd[self.index[0]]
        if self.type == "Write":
            return architecture.wd[self.index[0]]
        if self.type == "Discard":
            return 0
        if self.type == "Forward_branch":
            return uf
        if self.type == "Forwards_branch":
            return (self.index[2] - self.index[1] + 1) * uf
        if self.type == "Backward_branch":
            return ub
        if self.type == "Turn":
            return up
        if self.type == "Discard_branch":
            return 0
        if self.type == "Checkpoint_branch":
            return 0
        raise ValueError("Unknown cost for operation type " + self.type)

    def shift(self, size, branch=-1):
        if type(self.index) is int:
            self.index += size
        elif type(self.index) is list:
            if self.type == "Forwards":
                self.index[0] += size
                self.index[1] += size
            elif self.type == "Forwards_multi":
                self.index[1] += size
                self.index[2] += size
            elif self.type in [
                "Forward_branch",
                "Forwards_branch",
                "Discard_branch",
                "Checkpoint_branch",
                "Backward_branch",
            ]:
                if self.index[0] == branch:
                    for i in range(1, len(self.index)):
                        self.index[i] += size
            else:
                self.index[1] += size
