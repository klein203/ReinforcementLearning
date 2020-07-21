import logging
import numpy as np
from env.multi_armed_bandit import MultiArmedBanditEnv
from policy import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy
from history import History
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BanditAlgorithm(object):
    def __init__(self, env, *args, **kwargs):
        self.env = env
        self.k = env.get_k()
        self.q = np.zeros(self.k)
        self.n_times = np.zeros(self.k)
        self.history = History()
    
    def reset(self):
        self.q = np.zeros(self.k)
        self.n_times = np.zeros(self.k)

    def compute_g(self):
        return np.sum(np.dot(self.q, self.n_times))

    def store_params(self, prop, episode, val):
        self.history.store(prop, episode, val)
    
    def _update(self, action_idx, action_idx_reward):
        raise NotImplementedError()

    def run(self, policy, i_episode, n_steps):
        for _ in range(1, n_steps+1):
            action_idx, _ = policy.choose(self.q)
            action_idx_reward = self.env.get_reward(action_idx)

            self._update(action_idx, action_idx_reward)
            # self.store_params('step_seq', i_episode, action_idx)
    
    def run_episodes(self, policy, n_episodes=2000, n_steps=1000):
        self.history.clear()
        for i_episode in range(1, n_episodes+1):
            self.reset()
            self.run(policy, i_episode, n_steps)

            # final q values after an episode
            self.store_params('q(a)', i_episode, self.q)
            self.store_params('n(a)', i_episode, self.n_times)
            self.store_params('G', i_episode, self.compute_g())
            
    def report(self):
        q_per_action = self.history.get('q(a)')
        logging.info('Q(a) Mean Value')
        logging.info(np.mean(q_per_action, axis=0))
        logging.info('Q(a) Var Value')
        logging.info(np.var(q_per_action, axis=0))

        n_per_action = self.history.get('n(a)')
        logging.info('N(a) Mean Value')
        logging.info(np.mean(n_per_action, axis=0))

        g = self.history.get('G')
        logging.info('Final G Mean Value: %f' % np.mean(g, axis=0))

    def plot_params(self):
        q_per_action = self.history.get('q(a)')
        df = pd.DataFrame(q_per_action, columns=self.env.get_actions_space())

        plt.figure(figsize=(10, 10))
        _ = sns.swarmplot(data=df)
        plt.show()


class ActionValueMethods(BanditAlgorithm):
    def __init__(self, env, *args, **kwargs):
        super(ActionValueMethods, self).__init__(env, *args, **kwargs)
        self.sum_q = np.zeros(self.k)
    
    def reset(self):
        super(ActionValueMethods, self).reset()
        self.sum_q = np.zeros(self.k)

    def _update(self, action_idx, action_idx_reward):
        self.sum_q[action_idx] += action_idx_reward
        self.n_times[action_idx] += 1
        self.q[action_idx] = self.sum_q[action_idx] / self.n_times[action_idx]


class IncrementalImpl(BanditAlgorithm):    
    def _update(self, action_idx, reward):
        self.n_times[action_idx] = self.n_times[action_idx] + 1
        self.q[action_idx] = self.q[action_idx] + 1 / self.n_times[action_idx] * (reward - self.q[action_idx])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 10 Armed
    n_arm = 10
    actions_space = ['Arm_%d' % i for i in range(1, n_arm + 1)]
    env = MultiArmedBanditEnv(actions_space)
    logging.info('------------------------------------------------------------')
    logging.info('MultiArmedBanditEnv_Action_Target')
    env.info()


    agent = ActionValueMethods(env)

    logging.info('------------------------------------------------------------')
    logging.info('ActionValueMethods_RandomPolicy_EP100_ST1000')
    agent.run_episodes(RandomPolicy(), n_episodes=100, n_steps=1000)
    agent.report()

    logging.info('------------------------------------------------------------')
    logging.info('ActionValueMethods_GreedyPolicy_EP100_ST1000')
    agent.run_episodes(GreedyPolicy(), n_episodes=100, n_steps=1000)
    agent.report()

    logging.info('------------------------------------------------------------')
    logging.info('ActionValueMethods_EpsilonGreedyPolicy_0.01_EP100_ST1000')
    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.01), n_episodes=100, n_steps=1000)
    agent.report()

    logging.info('------------------------------------------------------------')
    logging.info('ActionValueMethods_EpsilonGreedyPolicy_0.1_EP100_ST1000')
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes=100, n_steps=1000)
    agent.report()

    logging.info('------------------------------------------------------------')
    logging.info('ActionValueMethods_EpsilonGreedyPolicy_0.5_EP100_ST1000')
    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.5), n_episodes=100, n_steps=1000)
    agent.report()


    agent = IncrementalImpl(env)

    logging.info('------------------------------------------------------------')
    logging.info('IncrementalImplementation_RandomPolicy_EP100_ST1000')
    agent.run_episodes(RandomPolicy(), n_episodes=100, n_steps=1000)
    agent.report()

    logging.info('------------------------------------------------------------')
    logging.info('IncrementalImplementation_GreedyPolicy_EP100_ST1000')
    agent.run_episodes(GreedyPolicy(), n_episodes=100, n_steps=1000)
    agent.report()

    logging.info('------------------------------------------------------------')
    logging.info('IncrementalImplementation_EpsilonGreedyPolicy_0.01_EP100_ST1000')
    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.01), n_episodes=100, n_steps=1000)
    agent.report()

    logging.info('------------------------------------------------------------')
    logging.info('IncrementalImplementation_EpsilonGreedyPolicy_0.1_EP100_ST1000')
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes=100, n_steps=1000)
    agent.report()

    logging.info('------------------------------------------------------------')
    logging.info('IncrementalImplementation_EpsilonGreedyPolicy_0.5_EP100_ST1000')
    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.5), n_episodes=100, n_steps=1000)
    agent.report()
    
    # agent.plot_params()
