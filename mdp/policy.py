import numpy as np


class AbstractPolicy(object):
    def __init__(self, *args, **kwargs):
        self.name = 'AbstractPolicy'
    
    def choose(self, *args, **kwargs):
        """
        π(a|s)
        """
        raise NotImplementedError()


class RandomPolicy(AbstractPolicy):
    def __init__(self, *args, **kwargs):
        super(RandomPolicy, self).__init__(*args, **kwargs)
        self.name = 'RandomPolicy'
    
    def choose(self, actions_probs, *args, **kwargs):
        actions = actions_probs['a'].unique()
        return np.random.choice(actions)


# class EpsilonGreedyPolicy(AbstractPolicy):
#     def __init__(self, epsilon=0.1, *args, **kwargs):
#         super(EpsilonGreedyPolicy, self).__init__(*args, **kwargs)
#         self.epsilon = epsilon
#         self.name = 'EpsilonGreedyPolicy(e=%.4f)' % self.epsilon
    
#     def choose(self, val_actions, ntimes_actions, t, *args, **kwargs):
#         if np.random.rand() >= self.epsilon:
#             actual = np.argmax(val_actions)
#         else:
#             actual = np.random.choice(len(val_actions))
        
#         return actual, val_actions
