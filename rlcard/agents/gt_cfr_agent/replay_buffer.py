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
# not been sampled many times in the past.                                  #
#                                                                           #
# ========================================================================= # 

# External imports
from bisect import bisect_left
from collections import deque
import numpy as np
import random

#
# Store elements and allow elements to be randomly sampled.
#
# Evict elements after they've been sampled a fixed number of times.
#
class ReplayBuffer:
    #
    # Initialize the buffer as queue with a fixed size.
    #
    # The lock controls access in and out of the buffer.
    #
    def __init__(self, max_size: int =int(1e6), evict_after_n_samples: int =10):
        #
        # FIFO buffer
        #
        self.buffer = deque(maxlen=max_size)
        #
        # This is a parallel buffer to the FIFO buffer that tracks
        # the number of times an item in the buffer has been sampled.
        #
        # self.sample_counts[idx]
        #   = num. times self.buffer[idx] item has been sampled
        #
        self.sample_counts = deque(maxlen=max_size)
        #
        # Evict an element after being sampled a fixed number of times
        #
        self.max_samples = evict_after_n_samples
        #
        # Cumulative weights for fast sampling
        #
        self.cum_weights = []
    
    #
    # Add an element
    #
    # If the buffer is full, then evict the oldest element
    #
    # Note: allow 'item' to be any data type
    #
    def put(self, item):
        #
        # Prevent race conditions
        #
        # Add the item to the buffer
        #
        self.buffer.append(item)
        #
        # The new item starts with a sample count of zero.
        #
        self.sample_counts.append(0)
        #
        # Append cumulative weights for binary search sampling
        #
        if self.cum_weights:
            #
            # New weight = 1 / (count + 1)
            #
            self.cum_weights.append(self.cum_weights[-1] + 1)
        else:
            #
            # Else, start the cumulative weights list with 1
            #
            self.cum_weights.append(1)
    
    #
    # Remove and return the oldest item
    #
    def get(self):
        #
        # Handle empty buffer case
        #
        if len(self.buffer) == 0:
            raise IndexError("Buffer is empty")
        #
        # Remove the item, its counts, and its cumulative weights
        #
        self.sample_counts.popleft()
        self.cum_weights.pop(0)
        return self.buffer.popleft()
    
    #
    # Update the cumulative weights list when an element is sampled.
    #
    # Complexity: O(log N)
    #
    # Note: it is assumed the caller is holding the lock
    #
    def _update_weights(self, idx):
        #
        # Get the weight amount that needs to be added to the item's weight
        #
        # new weight = 1 / (counts + 1)
        #
        # old weights = 1 / counts
        #
        # Note: diff < 0
        #
        diff = (1 / (self.sample_counts[idx] + 1)) - (1 / self.sample_counts[idx])
        #
        # Shift the cumulative weights after the index down by the change
        # in weights
        #
        for i in range(idx, len(self.cum_weights)):
            self.cum_weights[i] += diff
    
    #
    # Evict an item after it has been sampled too many times
    #
    # Note: it is assumed that the caller is holding the lock
    #
    def _evict(self, idx):
        #
        # Delete the item from memory
        #
        del self.buffer[idx]
        del self.sample_counts[idx]
        del self.cum_weights[idx]
        #
        # Recalculate cumulative weights after idx
        #
        # This means subtracting the weight of the removed item
        # from all cum_weights after the item.
        #
        for i in range(idx, len(self.cum_weights)):
            self.cum_weights[i] -= 1 / (self.sample_counts[i] + 1)

    #
    # Sample a fixed number of elements in the buffer.
    #
    # The probability of sampling an item is inversely proportional
    # to the number of times the item has been sampled in the past.
    #
    # Use binary seach sampling, O(batch_size * log N)
    # because we know this buffer will hold a million items.
    #
    # Standard sampling using np.random.choice is O(N) which
    # is too slow for a million items.
    #
    def sample(self, batch_size: int =1):
        #
        # Handle empty buffer case
        #
        if len(self.buffer) == 0:
            raise IndexError("Buffer is empty")
        #
        # Generate batch indices using binary search
        #
        samples = []
        #
        # Track sampled indices to prevent duplicate items being sampled
        #
        sampled_idxs = set()
        #
        # While we still need samples...
        #
        while len(samples) < batch_size:
            #
            # Generate a random number for weighted sampling
            # in the range [0, sum(weights)]
            #
            rand_val = random.uniform(0, self.cum_weights[-1])
            #
            # Get the idx corresponding to the random value
            #
            # Run binary search to find the idx in the list
            # such that, 
            # 
            # cum_weights[idx] <= rand_val < cum_weights[idx + 1]
            #
            # O(log N)
            #
            idx = bisect_left(self.cum_weights, rand_val)
            #
            # If the index is unique...
            #
            if idx not in sampled_idxs:
                #
                # Add the corresponding item to the samples list
                #
                samples.append(self.buffer[idx])
                #
                # Remember that we have sampled this item already
                #
                sampled_idxs.add(idx)
                # 
                # Update weights
                #
                self.sample_counts[idx] += 1
                #
                # If this item has been sampled too many times,
                # then evict it.
                #
                if self.sample_counts[idx] >= self.max_samples:
                    self._evict(idx)
                #
                # Else, update the sample weights for this element.
                #
                else:
                    self._update_weights(idx)
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