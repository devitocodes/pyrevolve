import numpy as np
from time import sleep
from threading import Thread
from queue import Queue
from .tools import MemorySlot, DummyContext


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
                slot.meta = None
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
    def read_one(self, idx, slot):
        t = Thread(target=self.read_step, args=(idx, slot))
        t.daemon = True
        t.start()

    def read_continous(self, idxs, memory):
        self.wait = True
        self.c_t = Thread(target=self._execute, args=(idxs, memory))
        self.c_t.daemon = True
        self.c_t.start()

    def read_step(self, idx, slot):
        print("Reading file %d.npy into slot %d" % (idx, slot.slot))
        with open("%d.npy" % idx, "rb") as f:
            with slot.write_lock:
                slot.data[:] = np.load(f, mmap_mode=None)
        
    def _execute(self, idxs, memory):
        while self.wait and len(idxs):
            slot = memory.get_free()
            if slot:
                slot.read_lock.acquire()
                c_idx = idxs.pop()
                self.read_step(c_idx, slot)
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
        ib.meta = 0
        ib.read_lock.acquire()
        self.write_queue.put(ib)
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
        with ib.read_lock:
            with ob.write_lock:
                self.fwd_op.apply(ib.data, ob.data, t_start=i, t_end=i+self.interval)
                ob.meta = i + self.interval
                ob.read_lock.acquire()
                i += self.interval
                ib = ob
                ob = self.memory.get_free()
        while i < self.nt:
            with ib.read_lock:
                with ob.write_lock:
                    self.fwd_op.apply(ib.data, ob.data, t_start=i, t_end=i+1)
                    ob.meta = i + 1
                    ob.read_lock.acquire()
            i += 1
            ib = ob
            ob = self.memory.get_free()
        print("Forward complete")

    def prefetch(self):
        if not len(self.disk_writer.written):
            return (None, None)# Prefetched all
        idx = self.disk_writer.written.pop()
        slot = self.memory.get_free()
        slot.read_lock.acquire()
        slot.meta = idx
        self.disk_reader.read_one(idx, slot)
        return idx, slot

    def apply_reverse(self, init_buff):
        ob = self.memory.get_free()
        assert(ob is not None)
        ob.read_lock.acquire()
        ib = DummyContext(init_buff)
        ib.read_lock.acquire()
        next_idx, next_slot = self.prefetch()
        for i in range(self.nt, self.nt - (self.nt % self.interval) - 1, -1):
            ib.read_lock.release()
            with ib.read_lock:
                with ob.write_lock:
                    ob.read_lock.release()
                    fwd = self.memory.meta(i)
                    print(i, ib, ob, fwd)
                    assert(fwd is not None)
                    self.rev_op.apply(ib.data, ob.data,fwd.data, t_start=i, t_end=i-1)
                    ob.meta = -i
                    fwd.read_lock.release() # Locked when loaded
            ib = ob
            ib.read_lock.acquire()
            ob = self.memory.get_free()
            assert(ob is not None)
            ob.read_lock.acquire()
        i -= 1

        while i>0:
            
            curr_idx, curr_slot = next_idx, next_slot
            next_idx, next_slot = self.prefetch()
            # Acquire and release the lock to ensure the prefetching is complete
            curr_slot.write_lock.acquire() # Blocks if the prefetcher is still writing
            curr_slot.write_lock.release()

            # Recomputation
            fib = curr_slot
            fob = self.memory.get_free()
            for fi in range(curr_idx, i, 1):
                with fib.read_lock:
                    with fob.write_lock:
                        self.fwd_op.apply(fib.data, fob.data, t_start=fi, t_end=fi+1)
                        fob.meta = fi + 1
                        fob.read_lock.acquire()
                fib = fob
                fob = self.memory.get_free()
    
            while i >= max(curr_idx, 1):
                with ib.read_lock:
                    with ob.write_lock:
                        ob.read_lock.release()
                        fwd = self.memory.meta(i)
                        print(i, ib, ob, fwd)
                        assert(fwd is not None)
                        self.rev_op.apply(ib.data, ob.data, fwd.data, t_start=i, t_end=i-1)
                        fwd.read_lock.release()
                        ob.meta = -i
                ib = ob
                ob = self.memory.get_free()
                assert(ob is not None)
                ob.read_lock.acquire()
                i -= 1
                    
        

    
