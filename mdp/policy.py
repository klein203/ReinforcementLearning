import numpy as np
import pandas as pd
import itertools as iter


class AbstractPolicy(object):
    def __init__(self, *args, **kwargs):
        self.name = 'AbstractPolicy'

    def choose(self, *args, **kwargs):
        """
        π(a|s)
        s:object -> a:object
        """
        raise NotImplementedError()


class RandomPolicy(AbstractPolicy):
    def __init__(self, *args, **kwargs):
        super(RandomPolicy, self).__init__(*args, **kwargs)
        self.name = 'RandomPolicy'
    
    def choose(self, actions_space):
        # p = 1/len(self.actions_space)
        return np.random.choice(actions_space)


class EpsilonGreedyPolicy(AbstractPolicy):
    def __init__(self, states_space, actions_space, epsilon=0.1, *args, **kwargs):
        super(EpsilonGreedyPolicy, self).__init__(states_space, actions_space, *args, **kwargs)
        self.epsilon = epsilon
        self.name = 'EpsilonGreedyPolicy(e=%.4f)' % self.epsilon

    def choose(self, actions_q):
        if np.random.rand() >= self.epsilon:
            actual = np.argmax(val_actions)
        else:
            actual = np.random.choice(len(val_actions))
        
        return actual, val_actions
    
    # def _check_available(self, s):
    #     df = self.q_df
    #     filter_df = df[(df['s'==s])]
    #     if filter_df.empty:
    #         q_default = 0
    #         self.q_df = df.append(
    #             pd.DataFrame(data=iter.product(s, self.actions_space, [q_default]), columns=['s', 'a', 'q']).astype({'s': object, 'a': object, 'q': float})
    #         )
        