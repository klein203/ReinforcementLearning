import numpy as np


class AbstractPolicy(object):
    def __init__(self, *args, **kwargs):
        self.name = 'AbstractPolicy'
    
    def choose(self, val_actions, ntimes_actions, t, *args, **kwargs):
        """
        π(a|s)
        """
        raise NotImplementedError()


class RandomPolicy(AbstractPolicy):
    def __init__(self, *args, **kwargs):
        super(RandomPolicy, self).__init__(*args, **kwargs)
        self.name = 'RandomPolicy'
    
    def choose(self, val_actions, ntimes_actions, t, *args, **kwargs):
        actual = np.random.choice(len(val_actions))
        return actual, val_actions


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, epsilon=0.1, *args, **kwargs):
        super(EpsilonGreedyPolicy, self).__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.name = 'EpsilonGreedyPolicy(e=%.4f)' % self.epsilon
    
    def choose(self, val_actions, ntimes_actions, t, *args, **kwargs):
        if np.random.rand() >= self.epsilon:
            actual = np.argmax(val_actions)
        else:
            actual = np.random.choice(len(val_actions))
        
        return actual, val_actions


class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, *args, **kwargs):
        super(GreedyPolicy, self).__init__(epsilon=0.0, *args, **kwargs)
        self.name = 'GreedyPolicy'


class UpperConfidenceBoundPolicy(AbstractPolicy):
    def __init__(self, c=2, *args, **kwargs):
        super(UpperConfidenceBoundPolicy, self).__init__(*args, **kwargs)
        self.c = c
        self.name = 'UpperConfidenceBoundPolicy(c=%d)' % self.c

    def choose(self, val_actions, ntimes_actions, t, *args, **kwargs):
        # choose action randomly when N(a) == 0
        action_idx_list = np.where(ntimes_actions==0)[0]
        if len(action_idx_list) > 0:
            actual = np.random.choice(action_idx_list)
        else:
            # choose argmax(Q_t(a) + c * sqrt(ln(t)/N_t(a)))
            actual = np.argmax(val_actions + self.c * np.sqrt(np.log(t) / ntimes_actions))
        
        return actual, val_actions


class SoftmaxPolicy(AbstractPolicy):
    def __init__(self, *args, **kwargs):
        super(SoftmaxPolicy, self).__init__(*args, **kwargs)
        self.name = 'SoftmaxPolicy'
    
    def choose(self, val_actions, ntimes_actions, *args, **kwargs):
        probabilities = np.exp(val_actions) / np.sum(np.exp(val_actions))
        actual = np.argmax(probabilities)
        return actual, probabilities
