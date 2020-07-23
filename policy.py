import numpy as np


class AbstractPolicy(object):
    def __init__(self, *args, **kwargs):
        self.name = 'AbstractPolicy'
    
    def choose(self, actions_prob):
        """
        π(a|s)
        """
        raise NotImplementedError()


class RandomPolicy(AbstractPolicy):
    def __init__(self, *args, **kwargs):
        super(RandomPolicy, self).__init__(*args, **kwargs)
        self.name = 'RandomPolicy'
    
    def choose(self, actions_prob):
        n_actions = len(actions_prob)
        action_idx = np.random.choice(n_actions)
        return action_idx, actions_prob[action_idx]


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, epsilon=0.1, *args, **kwargs):
        super(EpsilonGreedyPolicy, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.name = 'EpsilonGreedyPolicy(e=%.4f)' % self.epsilon
    
    def choose(self, actions_prob):
        if np.random.rand() >= self.epsilon:
            max_action_prob = np.max(actions_prob)
            action_idx_list = np.where(actions_prob==max_action_prob)[0]
            action_idx = np.random.choice(action_idx_list)
        else:
            n_actions = len(actions_prob)
            action_idx = np.random.choice(n_actions)
        
        return action_idx, actions_prob[action_idx]


class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, *args, **kwargs):
        super(GreedyPolicy, self).__init__(epsilon=0.0, *args, **kwargs)
        self.name = 'GreedyPolicy'
