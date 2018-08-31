import numpy as np
from time import sleep
from threading import Thread, Event
from queue import Queue
from .tools import MemorySlot, DummyContext
import sys
from .debugger import trace_start, trace_stop


class MemoryManager(object):    
    def __init__(self, ncp, shape, dtype):
        self.ncp = ncp
        self.shape = shape
        self._storage = np.zeros(tuple([ncp]+list(shape)), order='C', dtype=dtype)
        self._slots = [MemorySlot(i, self) for i in range(ncp)]
        
    def get_free(self, block=True):
        while True:
            for i, slot in enumerate(self._slots):
                if slot.free():
                    slot.meta = None
                    return slot
            if not block:
                break
            else:
                print("Waiting for slot to free up")
                sleep(0.1)
        return None

    def __getitem__(self, key):
        return self._storage[key, :]

    def meta(self, key):
        for slot in self._slots:
            if slot.meta==key:
                return slot
        return None

    def reset(self):
        for s in self._slots:
            s.reset()


class DiskThread(object):
    def __init__(self, prefix):
        self.stop_requested = Event()
        self.t = None
        if len(prefix) > 0:
            prefix=prefix+"/"
        self.prefix = prefix
        
    def done(self):
        self.stop_requested.set()
        self.t.join()
        trace_stop()


class DiskWriter(DiskThread):
    def __init__(self, queue, prefix, write_file=True):
        super(DiskWriter, self).__init__(prefix)
        self.queue = queue
        self.written = []
        self.writing = False
        self.write_files = write_file

    def start(self):
        self.wait = True
        self.t = Thread(target=self._execute, args=())
        self.t.daemon = True
        self.t.start()

    def _execute(self):
        self.writing = True
        while not self.stop_requested.isSet() or not self.queue.empty():
            try:
                mem = self.queue.get(True, timeout=1) # this waits on empty queue upto 1 sec
            except Exception as e:
                print(e, file=sys.stderr)
                break
            n_ts = mem.meta if hasattr(mem, 'meta') else -1
            self.written.append(n_ts)
            lock_value = mem.read_lock.counter.value if hasattr(mem, 'slot') else -1
            if self.write_files:
                with open("%s%d.npy" % (self.prefix, n_ts), "wb") as f:
                    np.save(f, mem.data)
            slot = -1
            if hasattr(mem, "slot"):
                slot = mem.slot
            mem.read_lock.release()
        self.writing = False
        print("Writer thread exiting", file=sys.stderr)


class DiskReader(DiskThread):
    def read_one(self, idx, slot):
        t = Thread(target=self.read_step, args=(idx, slot))
        t.daemon = True
        t.start()

    def read_continous(self, idxs, memory):
        self.stop_requested.clear()
        self.c_t = Thread(target=self._execute, args=(idxs, memory))
        self.c_t.daemon = True
        self.c_t.start()

    def read_step(self, idx, slot):
        with open("%s%d.npy" % (self.prefix, idx), "rb") as f:
            with slot.write_lock:
                slot.data[:] = np.load(f, mmap_mode=None)
        
    def _execute(self, idxs, memory):
        while not self.stop_requested.isSet() and len(idxs):
            slot = memory.get_free()
            if slot:
                slot.read_lock.acquire()
                c_idx = idxs.pop()
                self.read_step(c_idx, slot)
            else:
                sleep(0.1)
            
class Checkpointer(object):
    def __init__(self, fwd_op, rev_op, ncp, nt, fwcp, revcp, interval=1, nrevcp=3, file_prefix="", write_files=True):
        self.fwd_op = fwd_op
        self.rev_op = rev_op
        self.ncp = ncp
        self.nt = nt
        self.interval = interval
        self.memory = MemoryManager(ncp, (fwcp.size, ), fwcp.dtype)
        self.r_memory = MemoryManager(nrevcp, (revcp.size, ), revcp.dtype)
        self.write_queue = Queue()
        self.disk_writer = DiskWriter(self.write_queue, prefix=file_prefix, write_file=write_files)
        self.disk_reader = DiskReader(prefix=file_prefix)

    def apply_forward(self, init_buff):
        trace_start("trace.html",interval=5,auto=True) 
        self.memory.reset()
        ob = self.memory.get_free(block=True)
        ib = DummyContext(init_buff)
        ib.meta = 0
        ib.read_lock.acquire()
        self.write_queue.put(ib)
        i = 0
        try:
            self.disk_writer.start()
            
            while i < (self.nt - self.nt%self.interval):# - 2 * self.interval):
                with ib.read_lock:
                    with ob.write_lock:
                        self.fwd_op.apply(ib.data, ob.data, t_start=i, t_end=i+self.interval)
                        ob.meta = i + self.interval
                        l = ob.read_lock.acquire()
                        self.write_queue.put(ob)
                i+=self.interval
                ib = ob
                ob = self.memory.get_free(block=True)
        finally:
            print("Compute exiting", file=sys.stderr)
            self.disk_writer.done()
        """
        with ib.read_lock:
            with ob.write_lock:
                self.fwd_op.apply(ib.data, ob.data, t_start=i, t_end=i+self.interval)
                ob.meta = i + self.interval
                ob.read_lock.acquire()
                i += self.interval
                ib = ob
                ob = self.memory.get_free(block=True)
        
        while i < self.nt:
            with ib.read_lock:
                with ob.write_lock:
                    self.fwd_op.apply(ib.data, ob.data, t_start=i, t_end=i+1)
                    ob.meta = i + 1
                    ob.read_lock.acquire()
            i += 1
            ib = ob
            ob = self.memory.get_free(block=True)
        
        """

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
        self.r_memory.reset()
        return_value = None
        ob = self.r_memory.get_free(block=True)
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
                    assert(fwd is not None)
                    return_value = self.rev_op.apply(ib.data, ob.data,fwd.data, t_start=i, t_end=i-1)
                    ob.meta = -i
                    fwd.read_lock.release() # Locked when loaded
            ib = ob
            ib.read_lock.acquire()
            ob = self.r_memory.get_free(block=True)
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
                        assert(fwd is not None)
                        return_value = self.rev_op.apply(ib.data, ob.data, fwd.data, t_start=i, t_end=i-1)
                        fwd.read_lock.release()
                        ob.meta = -i
                ib = ob
                ob = self.r_memory.get_free()
                assert(ob is not None)
                ob.read_lock.acquire()
                i -= 1
        self.r_memory.reset()
        return return_value
                    
        

    
