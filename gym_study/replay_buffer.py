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



class SumTree(object):
    def __init__(self,  max_size: int, capacity: int = 4096):
        self.max_size = max_size
        self.capacity = capacity
        self.buffer = np.zeros((capacity, max_size))
        self.cursor = 0 # buffer cursor

        self.p_tree = np.zeros(capacity * 2 - 1)
    
    def append(self, p: float, data: np.array):
        # upward update p_delta
        cur_idx = self.capacity - 1 + self.cursor
        delta = p - self.p_tree[cur_idx]

        self.p_tree[cur_idx] = p
        while cur_idx != 0:
            parent_idx = (cur_idx + 1) // 2 - 1
            self.p_tree[parent_idx] += delta
            cur_idx = parent_idx

        # update buffer
        self.buffer[self.cursor, :] = data
        self.cursor += 1
        self.cursor %= self.capacity

    def sample(self, v):
        cur_idx = 0
        while True:
            if cur_idx >= self.capacity:
                # find leaf node
                break
            else:
                # downward search
                l_idx = cur_idx * 2 + 1 # (cur_idx + 1) * 2 - 1
                r_idx = l_idx + 1
                if self.p_tree[l_idx] > v:
                    cur_idx = l_idx
                else:
                    v -= self.p_tree[l_idx]
                    cur_idx = r_idx

        return self.p_tree[cur_idx], self.buffer[cur_idx - (self.capacity - 1)]

    
    def sum_p(self):
        return self.p_tree[0]
