import logging
import numpy as np
from env.multi_armed_bandit import MultiArmedBanditEnv
from policy import EpsilonGreedyPolicy
from history import History
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BanditAlgorithm(object):
    def __init__(self, env):
        self.env = env
        self.k = env.get_k()
        self.q = np.zeros(self.k)
        self.n_times = np.zeros(self.k)
        self.history = History()
    
    def reset(self):
        self.q = np.zeros(self.k)
        self.n_times = np.zeros(self.k)

    def _store_params(self, episode, prop, val):
        self.history.store(episode, prop, val)
    
    def _update(self, action_idx, reward):
        self.n_times[action_idx] += 1
        self.q[action_idx] = self.q[action_idx] + 1 / self.n_times[action_idx] * (reward - self.q[action_idx])

    def run(self, policy, i_episode, n_steps=1000):
        for i_step in range(1, n_steps+1):
            action_idx, p = policy.choose(None, self.q)
            reward = self.env.get_reward(action_idx)

            self._update(action_idx, reward)
            # self._store_params(i_episode, 'step_seq', action_idx)
    
    def run_episode(self, policy, n_episodes=2000):
        for i_episode in range(1, n_episodes+1):
            self.reset()
            self.run(policy, i_episode)

            # final q values after an episode
            self._store_params(i_episode, 'q_per_action', self.q)
            
    def report(self):
        q_per_action = self.history.get('q_per_action')
        logging.info('Q Mean Value')
        logging.info(np.mean(q_per_action, axis=0))
        logging.info('Q Var Value')
        logging.info(np.var(q_per_action, axis=0))

    def plot_params(self):
        q_per_action = self.history.get('q_per_action')
        df = pd.DataFrame(q_per_action, columns=self.env.get_actions_space())

        plt.figure(figsize=(10, 10))
        ax = sns.swarmplot(data=df)
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 10 Armed
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(1, n_arm + 1)]

    bandit = BanditAlgorithm(MultiArmedBanditEnv(actions_space))

    e_greedy_policy = EpsilonGreedyPolicy()
    bandit.run_episode(e_greedy_policy, n_episodes=2000)
    
    bandit.report()
    # bandit.plot_params()
