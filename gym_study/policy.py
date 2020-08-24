import numpy as np


class AbstractPolicy(object):
    def __init__(self, nb_actions: int, *args, **kwargs):
        self.name = 'AbstractPolicy'
        self.nb_actions = nb_actions
    
    def spec(self) -> dict:
        return {
            'name': self.name,
            'nb_actions': self.nb_actions,
            'desc': 'abstract policy',
        }
    
    def choose(self, *args, **kwargs) -> int:
        """
        π(a|s)
        """
        raise NotImplementedError()


class RandomPolicy(AbstractPolicy):
    def __init__(self, nb_action: int, *args, **kwargs):
        super(RandomPolicy, self).__init__(nb_action, *args, **kwargs)
        self.name = 'RandomPolicy'

    def spec(self) -> dict:
        return {
            'name': self.name,
            'nb_actions': self.nb_actions,
            'desc': 'random policy for action choosing with equal probability',
        }
        
    def choose(self, *args, **kwargs) -> int:
        action = np.random.choice(self.nb_actions)
        return action


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, nb_action: int, epsilon: float = 0.1, *args, **kwargs):
        super(EpsilonGreedyPolicy, self).__init__(nb_action, *args, **kwargs)
        self.name = 'EpsilonGreedyPolicy'
        self.epsilon = epsilon

    def spec(self) -> dict:
        return {
            'name': self.name,
            'nb_actions': self.nb_actions,
            'episilon': self.epsilon,
            'desc': 'choose argmax(action_vals) while random value larger than episilon. \
                otherwise, choose random action.',
        }
    
    def choose(self, action_vals: np.array, *args, **kwargs) -> int:
        if np.random.rand() >= self.epsilon:
            action = np.argmax(action_vals)
        else:
            action = np.random.choice(self.nb_actions)
        return action


class GreedyPolicy(AbstractPolicy):
    def __init__(self, nb_action: int, *args, **kwargs):
        super(GreedyPolicy, self).__init__(nb_action, *args, **kwargs)
        self.name = 'GreedyPolicy'
    
    def spec(self) -> dict:
        return {
            'name': self.name,
            'nb_actions': self.nb_actions,
            'desc': 'always choose argmax(action_vals). same as EpsilonGreedyPolicy(epsilon=0.0)',
        }
        
    def choose(self, action_vals: np.array, *args, **kwargs) -> int:
        action = np.argmax(action_vals)
        return action


class SoftmaxPolicy(AbstractPolicy):
    def __init__(self, nb_actions: int, *args, **kwargs):
        super(SoftmaxPolicy, self).__init__(nb_actions, *args, **kwargs)
        self.name = 'SoftmaxPolicy'

    def spec(self) -> dict:
        return {
            'name': self.name,
            'nb_actions': self.nb_actions,
            'desc': 'choose actions randomly under softmax(action_vals) probabilities',
        }
    
    def choose(self, action_vals: np.array, *args, **kwargs):
        probs = np.exp(action_vals) / np.sum(np.exp(action_vals))
        action = np.random.choice(range(self.nb_actions), p=probs)
        return action
