import numpy as np
import pandas as pd
import itertools as iter
import mdp
from mdp.env import MarkovEnv
from mdp.agent import PolicyIteration


cfg = mdp.gridworld_mdp_config
# cfg = mdp.student_mdp_config
env = MarkovEnv(cfg.get('states_space'),
    cfg.get('actions_space'),
    cfg.get('rewards_space'),
    cfg.get('p_matrix'))

# Policy Iteration
agent = PolicyIteration(env, gamma=0.9)
agent.policy_iter()


# cfg = mdp.autocleaner_mdp_config
# env = MarkovEnv(cfg.get('states_space'),
#     cfg.get('actions_space'),
#     cfg.get('rewards_space'),
#     cfg.get('p_matrix'))


