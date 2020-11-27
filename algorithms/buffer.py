import numpy as np
from collections import deque
import itertools
import random
import math

class ReplayBuffer(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, o, o_v, a, r, s_, o_, o_v_):
        transition = np.hstack((s, o, o_v, a, [r], s_, o_, o_v_))
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


class ReservoirBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store_transition(self, state, action):
        state = state[np.newaxis, :]
        action = action[np.newaxis, :]
        self.buffer.append((state, action))

    def sample(self, batch_size):
        """
        Efficient Reservoir Sampling
        Args:
            batch_size: n_batch

        Returns:
            state([batch_size, state_dim]), action([batch_size, action_dim])
        """
        n = len(self.buffer)
        reservoir = list(itertools.islice(self.buffer, 0, batch_size))
        threshold = batch_size * 4
        idx = batch_size
        while (idx < n and idx <= threshold):
            m = random.randint(0, idx)
            if m < batch_size:
                reservoir[m] = self.buffer[idx]
            idx += 1

        while (idx < n):
            p = float(batch_size) / idx
            u = random.random()
            g = math.floor(math.log(u) / math.log(1 - p))
            idx = idx + g
            if idx < n:
                k = random.randint(0, batch_size - 1)
                reservoir[k] = self.buffer[idx]
            idx += 1
        state, action = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), np.concatenate(action)

if __name__ == '__main__':
    buffer = ReservoirBuffer(10)
    state = np.ones(3)
    action = np.zeros(2)
    for i in range(50):
        buffer.store_transition(state, action)
    samples = buffer.sample(10)
    print(samples)
    print(samples[0].shape, samples[1].shape)