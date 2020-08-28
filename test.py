import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

import time
import numpy as np
import pandas as pd
import itertools as iter
from matplotlib import pyplot as plt
# import seaborn as sns
import pickle


# import gym
# from gym_study.utils import HistoryManager

# data = {
#     'epochs': [],
#     'loss': [],
#     'accuracy': [],
# }

# with open('histories.1598408218.369302.pkl', mode='rb') as f:
#     i_epoch = 0
#     while True:
#         try:
#             history = pickle.load(f)
#             data['epochs'].append(i_epoch)
#             data['loss'].extend(history.get('loss'))
#             data['accuracy'].extend(history.get('accuracy'))
#             i_epoch += 1
#         except EOFError:
#             break

# df = pd.DataFrame(data)

# sns.scatterplot(x='epochs', y='loss', data=df)
# plt.show()


# env = gym.make('CartPole-v0')
# agent = DQNAgent()
# for i_episode in range(1):
#     obs = env.reset()
#     for t in range(100):
#         # env.render()
#         a = env.action_space.sample()

#         obs_, r, done, info = env.step(a)
        
#         obs = obs_
#         print(type(obs))
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()

from gym_study.replay_buffer import PrioritizedReplayBuffer

p = np.abs(np.random.standard_normal((12)))
data = np.random.randint(0, 40, (12, 5))

prb = PrioritizedReplayBuffer(5, 8)
for i in range(6):
    prb.append(p[i], data[i, :])
    print(i)
    print(prb.buffer.pri_tree[:1])
    print(prb.buffer.pri_tree[1:3])
    print(prb.buffer.pri_tree[3:7])
    print(prb.buffer.pri_tree[7:])

pri, sample = prb.sample(4)
print(pri)
print(sample)

for i in range(6, 12):
    prb.append(p[i], data[i, :])
    print(i)
    print(prb.buffer.pri_tree[:1])
    print(prb.buffer.pri_tree[1:3])
    print(prb.buffer.pri_tree[3:7])
    print(prb.buffer.pri_tree[7:])

pri, sample = prb.sample(4)
print(pri)
print(sample)