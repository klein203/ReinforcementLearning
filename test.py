import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mdp.env import MarkovDecisionProcess as MDP
from mdp.policy import RandomPolicy


alpha = 0.8
beta = 0.7
p_data = np.array([
    [0, 0, 0, alpha],
    [0, 0, 1, 1-alpha],
    [1, 0, 0, 1-beta],
    [1, 0, 1, beta],
    [0, 1, 0, 1],
    # [0, 1, 1, 0],
    # [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 2, 0, 1],
    # [1, 2, 1, 0],
])

r_research = 1.5
r_wait = 1
r_data = np.array([
    [0, 0, 0, r_research],
    [0, 0, 1, r_research],
    [1, 0, 0, -3],
    [1, 0, 1, r_research],
    [0, 1, 0, r_wait],
    # [0, 1, 1, -10],
    # [1, 1, 0, -10],
    [1, 1, 1, r_wait],
    [1, 2, 0, 0],
    # [1, 2, 1, -10],
])

states_space = ['high', 'low']
actions_space = ['search', 'wait', 'recharge']

mdp = MDP(states_space, actions_space, p_data, r_data, discount_factor=0.9, init_state=0, terminal_states=[1])

s = 1
actions_probs = mdp.get_actions_probs(s)
print(actions_probs)

policy = RandomPolicy()
a = policy.choose(actions_probs)
print(a)

s_ = mdp.get_next_state(s, a)
print(s_)
