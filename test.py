import numpy as np
import pandas as pd
import itertools as iter
# import seaborn as sns
# import matplotlib.pyplot as plt
from mdp.env import MarkovEnv


alpha = 0.8
beta = 0.7
states_space = ['high', 'low']
actions_space = ['search', 'wait', 'recharge']

r_research = 1.5
r_wait = 1
rewards_space = [-3, 0, r_wait, r_research]

p_matrix = np.zeros((len(states_space), len(actions_space), len(states_space), len(rewards_space)))
env = MarkovEnv(states_space, actions_space, rewards_space, p_matrix)
env.set_prob('high', 'search', 'high', r_research, alpha)
env.set_prob('high', 'search', 'low', r_research, 1-alpha)
env.set_prob('low', 'search', 'high', -3, 1-beta)
env.set_prob('low', 'search', 'low', r_research, beta)
env.set_prob('high', 'wait', 'high', r_wait, 1)
env.set_prob('low', 'wait', 'low', r_wait, 1)
env.set_prob('low', 'recharge', 'high', 0, 1)

s = 'high'
for i in range(10):
    a = np.random.choice(env.get_actions(s))
    s_ = env.get_s_(s, a)
    print("%d: %s, %s -> %s" % (i, s, a, s_))
    s = s_
