# ========================================================================= #
#                                                                           #
# This file defines the replay buffer.                                      #
#                                                                           #
# The replay buffer is a FIFO buffer that allows for randomly sampling      #
# entries in the buffer.                                                    #
#                                                                           #
# The buffer has a fixed size, after which elements are evicted on a        #
# FIFO basis.                                                               #
#                                                                           #
# Elements are sampled with a preference towards elements that have         #
# not been sampled many times in the past, and evicted once they have       #
# been sampled a fixed number of times (the paper's "max grad updates       #
# per example").                                                            #
#                                                                           #
# ========================================================================= #

# External imports
from collections import deque
import numpy as np

#
# Store elements and allow elements to be randomly sampled.
#
# Evict elements after they've been sampled a fixed number of times.
#
class ReplayBuffer:
    #
    # Two parallel deques, oldest item first. Both have the same maxlen,
    # so at capacity they drop their oldest entry together and stay aligned.
    #
    def __init__(self, max_size: int =int(1e6), evict_after_n_samples: int =10):
        #
        # FIFO buffer of items
        #
        self.buffer = deque(maxlen=max_size)
        #
        # Number of times each item has been sampled
        #
        self.sample_counts = deque(maxlen=max_size)
        #
        # Evict an element after being sampled a fixed number of times
        #
        self.max_samples = evict_after_n_samples

    #
    # Add an element
    #
    # If the buffer is full, then the oldest element is evicted.
    #
    # Note: allow 'item' to be any data type
    #
    def put(self, item):
        self.buffer.append(item)
        self.sample_counts.append(0)

    #
    # Remove and return the oldest item
    #
    def get(self):
        if len(self.buffer) == 0:
            raise IndexError("Buffer is empty")
        self.sample_counts.popleft()
        return self.buffer.popleft()

    #
    # Sample a fixed number of distinct elements from the buffer.
    #
    # The probability of sampling an item is inversely proportional to
    # 1 + the number of times the item has been sampled in the past,
    # so fresh targets are preferred.
    #
    # One weighted draw over the whole buffer: O(N) per batch, a few
    # milliseconds at a million items.
    #
    def sample(self, batch_size: int =1):
        n = len(self.buffer)
        if n == 0:
            raise IndexError("Buffer is empty")
        if n < batch_size:
            raise IndexError(
                f"Insufficient buffer size ({n}) for desired batch size ({batch_size})"
            )
        #
        # Weighted draw without replacement
        #
        counts = np.fromiter(self.sample_counts, dtype=np.float64, count=n)
        weights = 1.0 / (counts + 1.0)
        idxs = np.random.choice(n, size=batch_size, replace=False, p=weights / weights.sum())
        samples = [self.buffer[i] for i in idxs]
        #
        # Record the use, then evict items that have been used enough times.
        # Delete from the highest index down so earlier indices stay valid.
        #
        for i in idxs:
            self.sample_counts[i] += 1
        for i in sorted(idxs, reverse=True):
            if self.sample_counts[i] >= self.max_samples:
                del self.buffer[i]
                del self.sample_counts[i]
        return samples

    #
    # Return the buffer size
    #
    def size(self):
        return len(self.buffer)

    #
    # Return if the buffer is empty
    #
    def is_empty(self):
        return len(self.buffer) == 0
