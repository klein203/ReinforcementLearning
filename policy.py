import numpy as np


class Policy(object):
    def __init__(self, *args, **kwargs):
        self.name = 'AbstractPolicy'
    
    def choose(self, actions_prob):
        """
        π(a|s), matrix (n_actions, 1)
        """
        raise NotImplementedError()


class RandomPolicy(Policy):
    def choose(self, actions_prob):
        n_actions = len(actions_prob)
        action_idx = np.random.choice(n_actions)
        return action_idx, actions_prob[action_idx]


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon=0.1, *args, **kwargs):
        super(EpsilonGreedyPolicy, self).__init__(*args, **kwargs)
        self.name = 'EpsilonGreedyPolicy'
        self.epsilon = epsilon
    
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
