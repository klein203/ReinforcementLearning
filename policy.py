import numpy as np


class Policy(object):
    def __init__(self, *args, **kwargs):
        self.name = 'AbstractPolicy'
    
    def choose(self, state, actions_prob):
        """
        π(a|s), matrix (n_actions, 1)
        """
        raise NotImplementedError()


class RandomPolicy(Policy):
    def choose(self, state, actions_prob):
        n_actions = len(actions_prob)
        action_idx = np.random.choice(n_actions)
        return action_idx, actions_prob[action_idx]


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon=0.9, *args, **kwargs):
        super(EpsilonGreedyPolicy, self).__init__(*args, **kwargs)
        self.name = 'EpsilonGreedyPolicy'
        self.epsilon = epsilon
    
    def choose(self, state, actions_prob):
        if np.random.random() <= self.epsilon:
            max_action_prob = np.max(actions_prob)
            action_idx_list = np.where(actions_prob==max_action_prob)[0]
            action_idx = np.random.choice(action_idx_list)
        else:
            n_actions = len(actions_prob)
            action_idx = np.random.choice(n_actions)
        
        return action_idx, actions_prob[action_idx]


class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, *args, **kwargs):
        super(GreedyPolicy, self).__init__(epsilon=1.0, *args, **kwargs)
        self.name = 'GreedyPolicy'


# if __name__ == "__main__":
#     policy = EpsilonGreedyPolicy()

#     # actions_prob = np.random.random(4)
#     actions_prob = np.array(
#         [0.2, 0.35, 0.1, 0.35]
#     )

#     for i in range(100):
#         action, probs = policy.choose(None, actions_prob)
#         print(action, probs)
