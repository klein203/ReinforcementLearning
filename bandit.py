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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    np.random.seed(seed=7)

    # plot config
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 10 armed bandit
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(n_arm)]
    env = MultiArmedBanditEnv(actions_space)
    env.info()

    reward_target_action = []
    reward_target_mean = env.get_reward_target_mean()
    reward_target_var = env.get_reward_target_var()
    sample_size = 100

    for item in zip(actions_space, reward_target_mean, reward_target_var):
        series = pd.Series(np.random.normal(loc=item[1], scale=item[2], size=sample_size), name=item[0])
        reward_target_action.append(series)

    sns.swarmplot(data=reward_target_action, size=1, ax=axes[0])

    # agent_clazzes = [ActionValueMethods, IncrementalImpl]
    # policy_clazzes = [RandomPolicy, GreedyPolicy, EpsilonGreedyPolicy]

    agent = IncrementalImpl(env)
    n_episodes=2000
    n_steps=1000


    q_steps_policy = []

    agent.run_episodes(RandomPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_policy.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='random'))


    agent.run_episodes(GreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_policy.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='greedy'))


    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.01), n_episodes, n_steps)
    agent.report()
    q_steps_policy.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='e=0.01'))


    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.5), n_episodes, n_steps)
    agent.report()
    q_steps_policy.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='e=0.5'))


    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_policy.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='e=0.1'))
    
    sns.lineplot(data=q_steps_policy, size=0.5, ax=axes[1])

    n_actions_mean = np.mean(agent.history.get('N_actions'), axis=0)
    df = pd.DataFrame(data=n_actions_mean.reshape(1, -1), columns=env.actions_space)
    sns.barplot(data=df, ax=axes[2])


    # 1st diagram format
    axes[0].set_title('Reward Distribution of Arms', fontsize=10)
    axes[0].set_xlabel('Arms', fontsize=8)
    axes[0].set_ylabel('Q*(a)', fontsize=8)
    axes[0].tick_params(labelsize=6)
    axes[0].grid(True, linestyle=':')
    
    # 2nd diagram format
    axes[1].set_title('Q Trends', fontsize=10)
    axes[1].set_xlabel('Steps', fontsize=8)
    axes[1].set_ylabel('Q(a)', fontsize=8)
    axes[1].tick_params(labelsize=6)
    axes[1].grid(True, axis='y', linestyle=':')
    axes[1].legend(loc='lower right', fontsize=8)

    # 3rd diagram format
    axes[2].set_title('N Distribution of Arms on e-greedy(0.1) Policy', fontsize=10)
    axes[2].set_xlabel('Arms', fontsize=8)
    axes[2].set_ylabel('N(a)', fontsize=8)
    axes[2].tick_params(labelsize=6)
    axes[2].grid(True, axis='y', linestyle=':')

    plt.tight_layout()
    plt.show()
