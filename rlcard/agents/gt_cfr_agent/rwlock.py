# ========================================================================= #
#                                                                           #
# Read-write lock that works ACROSS processes.                              #
#                                                                           #
# Guards the shared-memory weight block of the CFVN: trainer processes      #
# take the write lock to publish new weights, solver processes take the     #
# read lock to copy them in before a solve.                                 #
#                                                                           #
# The reader/writer counters live in shared memory (mp.Value), guarded by   #
# the condition's own lock, so every process sees the same state. The       #
# previous version (adapted from the O'Reilly Python Cookbook) kept them    #
# as plain Python attributes, which each process copied at spawn, so it     #
# gave no exclusion between processes at all.                               #
#                                                                           #
# Pass one ReadWriteLock to worker processes through Process(args=...),     #
# the way multiprocessing synchronisation objects are meant to travel.      #
#                                                                           #
# ========================================================================= #

import multiprocessing as mp


class ReadWriteLock:
  """ Many simultaneous readers, or one writer. Writers take priority:
  a waiting writer blocks new readers, so a stream of readers cannot
  starve the trainer. """

  def __init__(self):
    self._cond = mp.Condition(mp.Lock())
    self._readers = mp.Value('i', 0, lock=False)   # guarded by _cond
    self._writer = mp.Value('i', 0, lock=False)    # 1 while a writer holds or waits for the lock

  def acquire_read(self):
    """ Acquire a read lock. Blocks while a writer holds or waits for the lock. """
    with self._cond:
      while self._writer.value:
        self._cond.wait()
      self._readers.value += 1

  def release_read(self):
    """ Release a read lock. """
    with self._cond:
      self._readers.value -= 1
      if self._readers.value == 0:
        self._cond.notify_all()

  def acquire_write(self):
    """ Acquire the write lock. Blocks until no other writer holds it and
    every in-flight reader has released. """
    with self._cond:
      while self._writer.value:
        self._cond.wait()
      self._writer.value = 1
      while self._readers.value:
        self._cond.wait()

  def release_write(self):
    """ Release the write lock. """
    with self._cond:
      self._writer.value = 0
      self._cond.notify_all()

#----------------------------------------------------------------------------------------------------------

class ReadRWLock:
  # Context Manager class for ReadWriteLock
  def __init__(self, rwLock):
    self.rwLock = rwLock

  def __enter__(self):
    self.rwLock.acquire_read()
    return self         # Not mandatory, but returning to be safe

  def __exit__(self, exc_type, exc_value, traceback):
    self.rwLock.release_read()
    return False        # Raise the exception, if exited due to an exception

#----------------------------------------------------------------------------------------------------------

class WriteRWLock:
  # Context Manager class for ReadWriteLock
  def __init__(self, rwLock):
    self.rwLock = rwLock

  def __enter__(self):
    self.rwLock.acquire_write()
    return self         # Not mandatory, but returning to be safe

  def __exit__(self, exc_type, exc_value, traceback):
    self.rwLock.release_write()
    return False        # Raise the exception, if exited due to an exception

#----------------------------------------------------------------------------------------------------------
