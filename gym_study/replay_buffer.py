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

    def update(self, idx: int, pri: float):
        # idx range: [0, capacity * 2 - 1)
        delta = pri - self.pri_tree[idx]

        self.pri_tree[idx] = pri
        while idx != 0:
            parent_idx = (idx + 1) // 2 - 1
            self.pri_tree[parent_idx] += delta
            idx = parent_idx
    
    def append(self, pri: float, data: np.array):
        # upward update p_delta
        idx = self.capacity - 1 + self.cursor
        self.update(idx, pri)

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

        return cur_idx, self.pri_tree[cur_idx], self.buffer[cur_idx - (self.capacity - 1)]

    def sum_pri(self) -> float:
        return self.pri_tree[0]
    
    def max_pri(self) -> float:
        return np.max(self.get_all_pris())
    
    def get_all_pris(self) -> np.array:
        return self.pri_tree[-self.capacity:]


class PrioritizedReplayBuffer(object):
    def __init__(self, max_size: int, capacity: int = 5000, alpha: float = 0, beta: float = 1):
        self.max_size = max_size
        self.capacity = capacity
        self.buffer = SumTree(max_size, capacity)
        self.pri_max_limit = 1.0
        self.epsilon = 1e-4
        self.alpha = alpha
        self.beta = beta

    def append(self, data: np.array):
        pri = np.max(self.buffer.max_pri())
        if pri == 0:
            pri = self.pri_max_limit
        self.buffer.append(pri, data)

    def seed(self, seed: int = None):
        np.random.seed(seed)

    def sample(self, batch_size: int = 32) -> (np.array, np.array, np.array):
        sum_pri = self.buffer.sum_pri()
        nb_size = self.buffer.capacity
        batch_range = sum_pri / batch_size
        batch_idx = np.zeros(batch_size)
        batch_data = np.zeros((batch_size, self.max_size))
        
        # compute all pris with alpha exponent
        pris_with_alpha = _compute_pris_with_alpha(self.buffer.get_all_pris(), self.alpha)

        # compute all importance sampling weight
        importance_samplings = _compute_importance_samplings()
        
        # batch sampling, get idx and data
        for i in range(batch_size):
            low = max(0, batch_range * i)
            high = min(batch_range * (i + 1), sum_pri)
            pri = np.random.uniform(low, high)
            batch_idx[i], _, batch_data[i, :] = self.buffer.sample(pri)
            batch_idx[i] -= (self.capacity - 1) # [0, capacity)

        return batch_idx, importance_samplings[batch_idx], batch_data
    
    def _compute_pris_with_alpha(self, pris: np.array, alpha: float) -> np.array:
        return np.power(pris, alpha) / np.power(pris, alpha).sum()
    
    def _compute_importance_samplings(self, pris: np.array, n: int, beta: float) -> np.array:
        is_vals = np.power(n * pris, -beta)
        max_val = is_vals.max()
        return is_vals / max_val

    def update(self, idx: int, pri: float):
        # simple epsilon
        if pri == 0:
            pri += self.epsilon
        
        # TODO Peusdo: pri_i = 1/rank(i)

        # idx range [0, self.capacity)
        self.buffer.update(self.capacity - 1 + idx, pri)
    
    def batch_update(self, idxs: np.array, pris: np.array):
        for idx in idxs:
            self.update(idx, pris[idx])
