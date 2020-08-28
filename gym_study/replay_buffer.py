import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size: int, capacity: int = 5000):
        self.max_size = max_size
        self.max_length = capacity
        
        self.buffer = np.zeros((self.max_length, self.max_size))
        self.cursor = 0
        self.buffer_size = 0

    def append(self, data: np.array):
        self.buffer[self.cursor, :] = data
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

        self.pri_tree = np.zeros(capacity * 2 - 1)
    
    def append(self, pri: float, data: np.array):
        # upward update p_delta
        cur_idx = self.capacity - 1 + self.cursor
        delta = pri - self.pri_tree[cur_idx]

        self.pri_tree[cur_idx] = pri
        while cur_idx != 0:
            parent_idx = (cur_idx + 1) // 2 - 1
            self.pri_tree[parent_idx] += delta
            cur_idx = parent_idx

        # update buffer
        self.buffer[self.cursor, :] = data
        self.cursor += 1
        self.cursor %= self.capacity

    def sample(self, val: float) -> (float, np.array):
        cur_idx = 0
        while True:
            if cur_idx >= self.capacity - 1:
                # find leaf node
                break
            else:
                # downward search
                l_idx = cur_idx * 2 + 1 # (cur_idx + 1) * 2 - 1
                r_idx = l_idx + 1
                if self.pri_tree[l_idx] > val:
                    cur_idx = l_idx
                else:
                    val -= self.pri_tree[l_idx]
                    cur_idx = r_idx

        return self.pri_tree[cur_idx], self.buffer[cur_idx - (self.capacity - 1)]

    def sum_pri(self) -> float:
        return self.pri_tree[0]


class PrioritizedReplayBuffer(object):
    def __init__(self, max_size: int, capacity: int = 5000, alpha: float = 0.6, beta: float = 0.4):
        self.max_size = max_size
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = SumTree(max_size, capacity)

    def append(self, p: float, data: np.array):
        self.buffer.append(p, data)

    def seed(self, seed: int = None):
        np.random.seed(seed)

    def sample(self, batch_size: int = 32) -> np.array:
        sum_pri = self.buffer.sum_pri()
        batch_range = sum_pri / batch_size
        batch_pri = np.zeros(batch_size)
        batch_sample = np.zeros((batch_size, self.max_size))

        for i in range(batch_size):
            low = max(0, batch_range * i)
            high = min(batch_range * (i + 1), sum_pri)
            pri = np.random.uniform(low, high)
            batch_pri[i], batch_sample[i, :] = self.buffer.sample(pri)

        return batch_pri, batch_sample
