import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd
import itertools as iter
import mdp
from mdp.env import MarkovEnv, Maze2DEnv
from mdp.agent import PolicyIteration, InteractiveAgent, ValueIteration


# import gym

# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()



cfg = mdp.student_mdp_config
env = MarkovEnv(cfg.get('states_space'), cfg.get('actions_space'), cfg.get('rewards_space'), cfg.get('p_matrix'))

policy_matrix = np.array([
    [.0, .2, .7, .1],
    [.1, .1, .1, .7],
    [.5, .5, .0, .0],
    [1., .0, .0, .0],
    [.0, .0, .0, .0],
])

def p(env, s, a):
    return policy_matrix[env.s(s), env.a(a)]

agent = ValueIteration(env)

v = np.zeros((env.n_states))
for s in env.states_space:
    for a in env.get_actions(s):
        v[env.s(s)] += agent._compute_v(s, a) * p(env, s, a)

print(v)
