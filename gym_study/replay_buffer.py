import numpy as np


class ReplayBuffer(object):
    def __init__(self, max_size: int, max_length: int = 5000):
        self.max_size = max_size
        self.max_length = max_length
        
        self.buffer = np.zeros((self.max_length, self.max_size))
        self.cursor = 0
        self.buffer_size = 0

    def append(self, transition: np.array):
        self.buffer[self.cursor, :] = transition
        self.cursor += 1
        self.cursor = self.cursor % self.max_length
        self.buffer_size += 1
        self.buffer_size = min(self.buffer_size, self.max_length)

    def seed(self, seed: int = None):
        np.random.seed(seed)

    def sample(self, batch_size: int = 32) -> np.array:
        idx = np.random.choice(range(self.buffer_size), size=batch_size)
        return self.buffer[idx, :]
