import pickle


class HistoryManager(object):
    def __init__(self, filename: str, nb_batch_save_interval: int = 500):
        self.filename = filename
        self.nb_batch_save_interval = nb_batch_save_interval
        self.history_list = []

    def save(self, history, mode: str = 'ab'):
        self.history_list.append(history)
        self._batch_save(mode)

    def dump(self):
        self._batch_save('ab', True)
    
    def _batch_save(self, mode: str, force: bool = False):
        if force or len(self.history_list) >= self.nb_batch_save_interval:
            with open(self.filename, 'ab') as f:
                while len(self.history_list) > 0:
                    history = self.history_list.pop()
                    pickle.dump(history, f)
    
    def _reset(self):
        del self.history_list
        self.history_list = []

    def load(self, mode: str = 'rb'):
        self._reset()
        with open(self.filename, 'rb') as f:
            while True:
                try:
                    history = pickle.load(f)
                    self.history_list.append(history)
                except EOFError:
                    break

    def histories(self):
        return self.history_list
