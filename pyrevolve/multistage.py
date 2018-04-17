import numpy as np
from threading import Lock, Thread
from queue import Queue

class AtomicCounter:
    # From https://gist.github.com/benhoyt/8c8a8d62debe8e5aa5340373f9c509c7
    """An atomic, thread-safe incrementing counter.
    >>> counter = AtomicCounter()
    >>> counter.increment()
    1
    >>> counter.increment(4)
    5
    >>> counter = AtomicCounter(42.5)
    >>> counter.value
    42.5
    >>> counter.increment(0.5)
    43.0
    >>> counter = AtomicCounter()
    >>> def incrementor():
    ...     for i in range(100000):
    ...         counter.increment()
    >>> threads = []
    >>> for i in range(4):
    ...     thread = threading.Thread(target=incrementor)
    ...     thread.start()
    ...     threads.append(thread)
    >>> for thread in threads:
    ...     thread.join()
    >>> counter.value
    400000
    """
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
        return self.value

    def decrement(self, num=1):
        self.increment(-num)

class ReadLock(object):
    def __init__(self):
        self.counter = AtomicCounter()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.release()

    def free(self):
        return self.counter.value == 0

    def acquire(self):
        return self.counter.increment()

    def release(self):
        assert(not self.free())
        return self.counter.decrement()

class MemorySlot(object):
    def __init__(self, slot, memory):
        self.slot = slot
        self.memory = memory
        self.write_lock = Lock()
        self.read_lock = ReadLock()
        self.meta = None

    @property
    def data(self):
        return self.memory[self.slot]

    def free(self):
        return (not self.write_lock.locked()) and self.read_lock.free()

    def __str__(self):
        return str(self.meta)
        
    
class MemoryManager(object):
    
    def __init__(self, size_ckp, dtype, ncp):
        self.ncp = ncp
        self._storage = np.zeros((ncp, size_ckp), order='C', dtype=dtype)
        self._slots = [MemorySlot(i, self) for i in range(ncp)]
        
    def get_free(self):
        for slot in self._slots:
            if slot.free():
                return slot
        return None

    def __getitem__(self, key):
        return self._storage[key, :]

    def meta(self, key):
        for slot in self._slots:
            if slot.meta==key:
                return slot
        return None
            
class DiskWriter(object):
    def __init__(self, queue):
        self.queue = queue
        self.t = None
        self.wait = True

    def stop(self):
        self.wait = False

    def start(self):
        self.t = Thread(target=self._execute, args=())
        self.t.start()

    def _execute(self):
        print("Starting writer thread")
        while self.wait or not self.queue.empty():
            mem = self.queue.get() # this blocks on an empty queue
            n_ts = mem.meta
            print("About to write timestep %d" % n_ts)
            with open("%d.npy" % n_ts, "wb") as f:
                np.save(f, mem.data)
            print("Written")
            mem.read_lock.release()
        print("Ending writer thread")

class DummyContext(object):
    def __init__(self, data):
        self.data = data
        self.read_lock = Lock()

class Checkpointer(object):
    def __init__(self, checkpoint, fwd_op, rev_op, ncp, nt, interval=1):
        self.checkpoint = checkpoint
        size_ckp = checkpoint.size
        dtype = checkpoint.dtype
        self.fwd_op = fwd_op
        self.rev_op = rev_op
        self.ncp = ncp
        self.nt = nt
        self.interval = interval
        self.memory = MemoryManager(size_ckp, dtype, ncp)
        self.write_queue = Queue()
        self.disk_writer = DiskWriter(self.write_queue)

    def apply_forward(self, init_buff):
        ob = self.memory.get_free()
        ib = DummyContext(init_buff)
        i = 0
        try:
            self.disk_writer.start()
            while i < (self.nt - 2 * self.interval):
                with ib.read_lock:
                    with ob.write_lock:
                        print("cp", id(ob.data))
                        self.fwd_op.apply(ib.data, ob.data, t_start=i, t_end=i+self.interval)
                        print(ob.data)
                        ob.meta = i + self.interval
                        ob.read_lock.acquire()
                        self.write_queue.put(ob)
                i+=self.interval
                ib = ob
                ob = self.memory.get_free()
        finally:
            self.disk_writer.stop()
        while i < self.nt:
            with ib.read_lock:
                with ob.write_lock:
                    self.fwd_op.apply(ib.data, ob.data, t_start=i, t_end=i+1)
                    ob.meta = i + 1
                    ob.read_lock.acquire()
            i += 1
            ib = ob
            ob = self.memory.get_free()

        print("Forward done")
        print([str(x) for x in self.memory._slots])

    def apply_reverse(self, init_buff):
        ob = self.memory.get_free()
        ib = DummyContext(init_buff)
        for i in range(self.nt, self.nt - 2 * self.interval, -1):
            with ib.read_lock:
                with ob.write_lock:
                    self.rev_op.apply(ib.data, ob.data, self.memory.meta(i).data, t_start=i, t_end=i-1)
        

    
