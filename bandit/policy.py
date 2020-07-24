import numpy as np


class AbstractPolicy(object):
    def __init__(self, *args, **kwargs):
        self.name = 'AbstractPolicy'
    
    def choose(self, q_actions, n_actions, t, *args, **kwargs):
        """
        π(a|s)
        """
        raise NotImplementedError()


class RandomPolicy(AbstractPolicy):
    def __init__(self, *args, **kwargs):
        super(RandomPolicy, self).__init__(*args, **kwargs)
        self.name = 'RandomPolicy'
    
    def choose(self, q_actions, n_actions, t, *args, **kwargs):
        action_idx = np.random.choice(len(q_actions))
        return action_idx, q_actions[action_idx]


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, epsilon=0.1, *args, **kwargs):
        super(EpsilonGreedyPolicy, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.name = 'EpsilonGreedyPolicy(e=%.4f)' % self.epsilon
    
    def choose(self, q_actions, n_actions, t, *args, **kwargs):
        if np.random.rand() >= self.epsilon:
            max_q_actions = np.max(q_actions)
            action_idx_list = np.where(q_actions==max_q_actions)[0]
            action_idx = np.random.choice(action_idx_list)
        else:
            action_idx = np.random.choice(len(q_actions))
        
        return action_idx, q_actions[action_idx]


class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, *args, **kwargs):
        super(GreedyPolicy, self).__init__(epsilon=0.0, *args, **kwargs)
        self.name = 'GreedyPolicy'


class UpperConfidenceBoundPolicy(AbstractPolicy):
    def __init__(self, c=2, *args, **kwargs):
        super(UpperConfidenceBoundPolicy, self).__init__(*args, **kwargs)
        self.name = 'UpperConfidenceBoundPolicy'
        self.c = c

    def choose(self, q_actions, n_actions, t, *args, **kwargs):
        # choose action randomly when N(a) == 0
        action_idx_list = np.where(n_actions==0)[0]
        if len(action_idx_list) > 0:
            action_idx = np.random.choice(action_idx_list)
        else:
            # choose argmax
            action_idx = np.argmax(q_actions + self.c * np.sqrt(np.log(t) / n_actions))
        
        return action_idx, q_actions[action_idx]
