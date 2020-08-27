import pickle
from queue import Queue


class HistoryManager(object):
    def __init__(self, filename: str, nb_batch_save_interval: int = 500):
        self.filename = filename
        self.nb_batch_save_interval = nb_batch_save_interval
        self.q = Queue()

    def save(self, history, mode: str = 'ab'):
        self.q.put(history)
        self._batch_save(mode)

    def dump(self):
        self._batch_save('ab', force=True)
    
    def _batch_save(self, mode: str, force: bool = False):
        if force or self.q.qsize() >= self.nb_batch_save_interval:
            with open(self.filename, 'ab') as f:
                while not self.q.empty():
                    history = self.q.get_nowait()
                    pickle.dump(history, f)
    
    def _reset(self):
        del self.q
        self.q = Queue()

    def load(self, mode: str = 'rb'):
        self._reset()
        with open(self.filename, 'rb') as f:
            while True:
                try:
                    history = pickle.load(f)
                    self.q.put(history)
                except EOFError:
                    break

    def histories(self):
        return self.q
