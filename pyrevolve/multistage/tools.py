from threading import Lock

class DummyContext(object):
    def __init__(self, data):
        self.data = data
        self.read_lock = Lock()



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
        if self.free():
            raise RuntimeError("The lock is already free")
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

    def reset(self):
        try:
            self.read_lock.release()
            self.write_lock.release()
        except RuntimeError:
            pass
