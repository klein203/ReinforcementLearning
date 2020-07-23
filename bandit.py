import logging
import numpy as np
from env.multi_armed_bandit import MultiArmedBanditEnv, NonstationaryMultiArmedBanditEnv
from policy import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy
from history import History
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class BanditAlgorithm(object):
    def __init__(self, env, *args, **kwargs):
        self.name = 'BanditAlgorithm Abstract'
        self.env = env
        self.k = env.get_k()
        self.q = np.zeros(self.k)
        self.n_times = np.zeros(self.k)
        self.history = History()
    
    def reset(self):
        self.q = np.zeros(self.k)
        self.n_times = np.zeros(self.k)

    def store_params(self, prop, episode, val):
        self.history.store(prop, episode, val)
    
    def _update(self, action_idx, action_idx_reward):
        raise NotImplementedError()

    def _run(self, policy, i_episode):
        q_step = np.zeros(self.n_steps)

        for i_step in range(self.n_steps):
            action_idx, _ = policy.choose(self.q)
            action_idx_reward = self.env.get_reward(action_idx)

            self._update(action_idx, action_idx_reward)
            
            # history gathering
            q_step[i_step] = action_idx_reward
        
        # Q_step: n_episode x n_step
        self.store_params('Q_steps', i_episode, q_step)
        # N_arm: n_episode x n_arms
        self.store_params('N_actions', i_episode, self.n_times)
    
    def run_episodes(self, policy, n_episodes=2000, n_steps=1000):
        self.history.clear()
        self.policy = policy
        self.n_episodes = n_episodes
        self.n_steps = n_steps

        for i_episode in range(n_episodes):
            self.reset()
            self._run(policy, i_episode)
            
    def report(self):
        logging.info('------------------------------------------------------------')
        logging.info("%s_%s_EP%d_ST%d" % (self.name, self.policy.name, self.n_episodes, self.n_steps))
        logging.info('------------------------------------------------------------')

        q_steps = self.history.get('Q_steps')
        q_steps_mean = np.mean(q_steps, axis=0)
        logging.debug('Q Mean Value: %s' % q_steps_mean)
        logging.info('Final Q Mean Value: %.4f' % q_steps_mean[-1])

        n_actions = self.history.get('N_actions')
        n_actions_mean = np.mean(n_actions, axis=0)
        logging.info('N Mean Value: %s' % n_actions_mean)


class ActionValueMethods(BanditAlgorithm):
    def __init__(self, env, *args, **kwargs):
        super(ActionValueMethods, self).__init__(env, *args, **kwargs)
        self.name = 'ActionValueMethods'
        self.sum_q = np.zeros(self.k)
    
    def reset(self):
        super(ActionValueMethods, self).reset()
        self.sum_q = np.zeros(self.k)

    def _update(self, action_idx, action_idx_reward):
        self.sum_q[action_idx] += action_idx_reward
        self.n_times[action_idx] += 1
        self.q[action_idx] = self.sum_q[action_idx] / self.n_times[action_idx]


class IncrementalImpl(BanditAlgorithm):
    def __init__(self, env, *args, **kwargs):
        super(IncrementalImpl, self).__init__(env, *args, **kwargs)
        self.name = 'IncrementalImplementation'
    
    def _update(self, action_idx, action_idx_reward):
        self.n_times[action_idx] = self.n_times[action_idx] + 1
        self.q[action_idx] = self.q[action_idx] + 1 / self.n_times[action_idx] * (action_idx_reward - self.q[action_idx])


class ExponentialRecencyWeightedAverage(BanditAlgorithm):
    def __init__(self, env, step_size, *args, **kwargs):
        super(ExponentialRecencyWeightedAverage, self).__init__(env, *args, **kwargs)
        self.name = 'ExponentialRecencyWeightedAverage'
        # alpha
        self.step_size = step_size

    def _update(self, action_idx, action_idx_reward):
        # exponential recency-weighted average: the bigger the i is, the bigger the weight of R_i is
        self.n_times[action_idx] = self.n_times[action_idx] + 1
        self.q[action_idx] = self.q[action_idx] + self.step_size * (action_idx_reward - self.q[action_idx])


def fig_1(n_episodes=100, n_steps=1000):
    """
    ref figure on page 29
    """
    # plots init and config
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1st diagram format
    ax1.set_title('Reward Distribution of Arms', fontsize=10)
    ax1.set_xlabel('Arms', fontsize=8)
    ax1.set_ylabel('R', fontsize=8)
    ax1.tick_params(labelsize=6)
    ax1.grid(True, linestyle=':')
    
    # 2nd diagram format
    ax2.set_title('N Distribution of Arms on e-greedy(0.1) Policy', fontsize=10)
    ax2.set_xlabel('Arms', fontsize=8)
    ax2.set_ylabel('N(a)', fontsize=8)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, axis='y', linestyle=':')

    # 3rd diagram format
    ax3.set_title('Q Trends', fontsize=10)
    ax3.set_xlabel('Steps', fontsize=8)
    ax3.set_ylabel('Q(a)', fontsize=8)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, axis='y', linestyle=':')
    ax3.legend(loc='lower right', fontsize=7)


    # 10-armed bandit enviroment
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(n_arm)]
    env = MultiArmedBanditEnv(actions_space)
    env.info()

    # plot reward target distribution of given environment
    df = env.sampling()
    sns.swarmplot(data=df, size=1, ax=ax1)

    # evaluation using incremental implementation algorithm with various policies
    agent = IncrementalImpl(env)

    # params storing
    q_steps_list = []

    # random
    agent.run_episodes(RandomPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='random'))

    # greedy, e=0.0
    agent.run_episodes(GreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='greedy'))

    # e-greedy, e=0.01
    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.01), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='e=0.01'))

    # e-greedy, e=0.5
    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.5), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='e=0.5'))

    # e-greedy, e=0.1
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='e=0.1'))
    
    # plot Q trends over steps with various policies
    sns.lineplot(data=q_steps_list, size=0.5, ax=ax3)

    # plot N distribution using e-greedy(0.1) policy
    # by using e-greedy policy, action reward with higher target mean value in the first place 
    # has much higher possiblity to be choosed
    n_actions_mean = np.mean(agent.history.get('N_actions'), axis=0)
    df = pd.DataFrame(data=n_actions_mean.reshape(1, -1), columns=env.actions_space)
    sns.barplot(data=df, ax=ax2)

    # plot show
    plt.tight_layout()
    plt.show()

def fig_2(n_episodes=100, n_steps=10000):
    """
    ref practice 2.5 on page 33
    """
    # plots init and config
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # 1st diagram format
    ax1.set_title('Reward Distribution of Arms', fontsize=10)
    ax1.set_xlabel('Arms', fontsize=8)
    ax1.set_ylabel('R', fontsize=8)
    ax1.tick_params(labelsize=6)
    ax1.grid(True, linestyle=':')
    
    # 2nd diagram format
    ax2.set_title('N Dist, exp recency-weighted avg + e-greedy(0.1)', fontsize=10)
    ax2.set_xlabel('Arms', fontsize=8)
    ax2.set_ylabel('N(a)', fontsize=8)
    ax2.tick_params(labelsize=6)
    ax2.grid(True, axis='y', linestyle=':')

    # 3rd diagram format
    ax3.set_title('Q Trends', fontsize=10)
    ax3.set_xlabel('Steps', fontsize=8)
    ax3.set_ylabel('Q(a)', fontsize=8)
    ax3.tick_params(labelsize=6)
    ax3.grid(True, axis='y', linestyle=':')
    ax3.legend(loc='lower right', fontsize=7)

    # 10-armed bandit enviroment
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(n_arm)]
    env = NonstationaryMultiArmedBanditEnv(actions_space)
    env.info()

    # plot reward target distribution of given environment
    df = env.sampling()
    sns.swarmplot(data=df, size=1, ax=ax1)

    # params storing
    q_steps_list = []

    # incremental implementation algorithm with e-greedy(0.1) policy
    agent = IncrementalImpl(env)
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='incremental, e-greedy(e=0.1)'))

    # exponential recency weighted average with e-greedy(0.1) policy
    agent = ExponentialRecencyWeightedAverage(env, step_size=0.1)
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='step move(a=0.1), e-greedy(e=0.1)'))

    # plot Q trends over steps with various policies
    sns.lineplot(data=q_steps_list, size=0.5, ax=ax3)

    # plot N distribution using exponential recency weighted average with e-greedy(0.1) policy
    n_actions_mean = np.mean(agent.history.get('N_actions'), axis=0)
    df = pd.DataFrame(data=n_actions_mean.reshape(1, -1), columns=env.actions_space)
    sns.barplot(data=df, ax=ax2)

    # plot show
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    np.random.seed(seed=7)

    # fig_1()
    fig_2()
