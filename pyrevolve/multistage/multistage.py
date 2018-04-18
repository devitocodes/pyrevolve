import numpy as np
from threading import Thread
from queue import Queue
from tools import MemorySlot, DummyContext


class MemoryManager(object):    
    def __init__(self, ncp, shape, dtype):
        self.ncp = ncp
        self.shape = shape
        self._storage = np.zeros(tuple([ncp]+list(shape)), order='C', dtype=dtype)
        self._slots = [MemorySlot(i, self) for i in range(ncp)]
        
    def get_free(self):
        for i, slot in enumerate(self._slots):
            if slot.free():
                print("MemoryManager: Returning slot %d" % i)
                return slot
        return None

    def __getitem__(self, key):
        return self._storage[key, :]

    def meta(self, key):
        for slot in self._slots:
            if slot.meta==key:
                return slot
        return None


class DiskThread(object):
    def __init__(self):
        self.wait = True
        self.t = None
        
    def stop(self):
        self.wait = False


class DiskWriter(DiskThread):
    def __init__(self, queue):
        super(DiskWriter, self).__init__()
        self.queue = queue
        self.written = []

    def start(self):
        self.t = Thread(target=self._execute, args=())
        self.t.daemon = True
        self.t.start()

    def _execute(self):
        while self.wait or not self.queue.empty():
            mem = self.queue.get() # this blocks on an empty queue
            n_ts = mem.meta
            self.written.append(n_ts)
            with open("%d.npy" % n_ts, "wb") as f:
                np.save(f, mem.data)
            mem.read_lock.release()


class DiskReader(DiskThread):
    def read_one(self, data):
        pass
    def read_continous(self, idxs, memory):
        self.wait = True
        self.c_t = Thread(target=self._execute, args=(idxs, memory))
        self.c_t.daemon = True
        self.c_t.start()
        
    def _execute(self, idxs, memory):
        while self.wait:
            slot = memory.get_free()
            if slot:
                c_idx = idxs.pop()
                print("Reading file %d.npy" % c_idx)
                with open("%d.npy" % c_idx, "rb") as f:
                    np.load(f, slot, memmap=None)
                    slot.read_lock.acquire()
            else:
                print("Sleeping for 0.1 seconds")
                sleep(0.1)
            
class Checkpointer(object):
    def __init__(self, fwd_op, rev_op, ncp, nt, shape, dtype, interval=1):
        self.fwd_op = fwd_op
        self.rev_op = rev_op
        self.ncp = ncp
        self.nt = nt
        self.interval = interval
        self.memory = MemoryManager(ncp, shape, dtype)
        self.write_queue = Queue()
        self.disk_writer = DiskWriter(self.write_queue)
        self.disk_reader = DiskReader()

    def apply_forward(self, init_buff):
        ob = self.memory.get_free()
        ib = DummyContext(init_buff)
        i = 0
        try:
            self.disk_writer.start()
            while i < (self.nt - 2 * self.interval):
                with ib.read_lock:
                    with ob.write_lock:
                        self.fwd_op.apply(ib.data, ob.data, t_start=i, t_end=i+self.interval)
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

    def apply_reverse(self, init_buff):
        ob = self.memory.get_free()
        ib = DummyContext(init_buff)
        self.disk_reader.read_continous(self.disk_writer.written, self.memory)
        for i in range(self.nt, self.nt - 2 * self.interval, -1):
            with ib.read_lock:
                with ob.write_lock:
                    fwd = self.memory.meta(i)
                    self.rev_op.apply(ib.data, ob.data,fwd.data, t_start=i, t_end=i-1)
                    fwd.read_lock.release() # Locked when loaded
            ib = ob
            ob = self.memory.get_free()
                    
        

    
