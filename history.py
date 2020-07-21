import numpy as np


class History(object):
    def __init__(self):
        self.m = dict()

    def store(self, prop, episode, val):
        key = str(prop)
        if key not in self.m:
            self.m[key] = np.array(val.reshape(1, -1))
        else:
            self.m[key] = np.insert(self.m.get(key), -1, values=val, axis=0)

    def get(self, prop):
        key = str(prop)
        return self.m.get(key)

    def clear(self):
        self.m.clear()



if __name__ == "__main__":
    h = History()
    h.store('test', 1, np.random.standard_normal(5))
    h.store('test', 2, np.random.standard_normal(5))
    h.store('test', 3, np.random.standard_normal(5))
    print(h.get('test'))
    h.clear()
    print(h.get('test'))
    