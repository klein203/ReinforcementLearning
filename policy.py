import random


class Policy(object):
    def __init__(self, *args, **kwargs):
        self.name = 'AbstractPolicy'
    
    def choose(self, state, actions_prob):
        """
        π(a|s), { a: p(a|s) }
        """
        raise NotImplementedError()


class RandomPolicy(Policy):
    def choose(self, state, actions_prob):
        action_list = list(actions_prob.keys())
        action_choose = random.choice(action_list)
        return action_choose, actions_prob.get(action_choose)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon=0.9, *args, **kwargs):
        super(EpsilonGreedyPolicy, self).__init__(*args, **kwargs)
        self.name = 'EpsilonGreedyPolicy'
        self.epsilon = epsilon
    
    def choose(self, state, actions_prob):
        action_list = []
        if random.random() <= self.epsilon:
            max_action_prob = max(actions_prob.values())
            action_list = [key for key, val in actions_prob.items() if val == max_action_prob]
        else:
            action_list = list(actions_prob.keys())
        
        action_choose = random.choice(action_list)
        return action_choose, actions_prob.get(action_choose)


class GreedyPolicy(EpsilonGreedyPolicy):
    def __init__(self, *args, **kwargs):
        super(GreedyPolicy, self).__init__(epsilon=1.0, *args, **kwargs)
        self.name = 'GreedyPolicy'


if __name__ == "__main__":
    policy = EpsilonGreedyPolicy()
    actions_space = ['a1', 'a2', 'a3', 'a4']
    # actions = { action: .25 for action in actions_space }
    actions = { 
        'a1': .25,
        'a2': .3,
        'a3': .3,
        'a4': .15,
    }
    for i in range(100):
        action, probs = policy.choose(None, actions)
        print(action, probs)
