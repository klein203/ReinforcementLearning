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

    def run(self, policy, i_episode, n_steps):
        sum_q = np.zeros(n_steps)

        for i_step in range(n_steps):
            action_idx, _ = policy.choose(self.q)
            action_idx_reward = self.env.get_reward(action_idx)

            self._update(action_idx, action_idx_reward)

            sum_q[i_step] = np.sum(self.q)
        
        # final q values after an episode
        self.store_params('q(a)', i_episode, self.q)
        self.store_params('n(a)', i_episode, self.n_times)
        self.store_params('G(step)', i_episode, sum_q)
    
    def run_episodes(self, policy, n_episodes=2000, n_steps=1000):
        self.history.clear()
        for i_episode in range(n_episodes):
            self.reset()
            self.run(policy, i_episode, n_steps)
            
    def report(self):
        q_per_action = self.history.get('q(a)')
        logging.info('Q(a) Mean Value')
        logging.info(np.mean(q_per_action, axis=0))
        logging.info('Q(a) Var Value')
        logging.info(np.var(q_per_action, axis=0))

        n_per_action = self.history.get('n(a)')
        logging.info('N(a) Mean Value')
        logging.info(np.mean(n_per_action, axis=0))

        g_per_step = self.history.get('G(step)')
        logging.debug('G(step) Mean Value')
        logging.debug(np.mean(g_per_step, axis=0))
        logging.info('Final G Mean Value: %f' % np.mean(g_per_step, axis=0)[-1])


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

    # 10 Armed
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(n_arm)]
    env = MultiArmedBanditEnv(actions_space)
    logging.info('------------------------------------------------------------')
    logging.info('MultiArmedBanditEnv_Action_Target')
    logging.info('------------------------------------------------------------')
    env.info()

    # agent_clazzes = [ActionValueMethods, IncrementalImpl]
    # policy_clazzes = [RandomPolicy, GreedyPolicy, EpsilonGreedyPolicy]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    agent = ActionValueMethods(env)
    n_episodes=100
    n_steps=1000

    g_per_policy = []

    policy = RandomPolicy()
    logging.info('------------------------------------------------------------')
    logging.info("%s_%s_EP%d_ST%d" % (agent.name, policy.name, n_episodes, n_steps))
    logging.info('------------------------------------------------------------')
    agent.run_episodes(policy, n_episodes, n_steps)
    agent.report()

    q_df = pd.DataFrame(agent.history.get('q(a)'), columns=env.get_actions_space())
    sns.swarmplot(data=q_df, size=1, ax=axes[0])

    g_per_policy.append(pd.Series(np.mean(agent.history.get('G(step)'), axis=0), name='random'))


    policy = GreedyPolicy()
    logging.info('------------------------------------------------------------')
    logging.info("%s_%s_EP%d_ST%d" % (agent.name, policy.name, n_episodes, n_steps))
    logging.info('------------------------------------------------------------')
    agent.run_episodes(policy, n_episodes, n_steps)
    agent.report()

    g_per_policy.append(pd.Series(np.mean(agent.history.get('G(step)'), axis=0), name='greedy'))


    # policy = EpsilonGreedyPolicy(epsilon=0.01)
    # logging.info('------------------------------------------------------------')
    # logging.info("%s_%s_EP%d_ST%d" % (agent.name, policy.name, n_episodes, n_steps))
    # logging.info('------------------------------------------------------------')
    # agent.run_episodes(policy, n_episodes, n_steps)
    # agent.report()

    # g_per_policy.append(pd.Series(np.mean(agent.history.get('G(step)'), axis=0), name='e-greedy %.2f' % policy.epsilon))


    policy = EpsilonGreedyPolicy()
    logging.info('------------------------------------------------------------')
    logging.info("%s_%s_EP%d_ST%d" % (agent.name, policy.name, n_episodes, n_steps))
    logging.info('------------------------------------------------------------')
    agent.run_episodes(policy, n_episodes, n_steps)
    agent.report()

    g_per_policy.append(pd.Series(np.mean(agent.history.get('G(step)'), axis=0), name='e-greedy %.2f' % policy.epsilon))


    # policy = EpsilonGreedyPolicy(epsilon=0.5)
    # logging.info('------------------------------------------------------------')
    # logging.info("%s_%s_EP%d_ST%d" % (agent.name, policy.name, n_episodes, n_steps))
    # logging.info('------------------------------------------------------------')
    # agent.run_episodes(policy, n_episodes, n_steps)
    # agent.report()

    # g_per_policy.append(pd.Series(np.mean(agent.history.get('G(step)'), axis=0), name='e-greedy %.2f' % policy.epsilon))
    
    sns.lineplot(data=g_per_policy, size=0.5, ax=axes[1])


    # 1st diagram format
    axes[0].set_title('Q*(a) Distribution of Arms', fontsize=10)
    axes[0].set_xlabel('Actions', fontsize=8)
    axes[0].set_ylabel('Q*(a)', fontsize=8)
    axes[0].tick_params(labelsize=6)
    axes[0].grid(True, linestyle=':')
    
    # 2nd diagram format
    axes[1].set_title('G(a) Trends', fontsize=10)
    axes[1].set_xlabel('Steps', fontsize=8)
    axes[1].set_ylabel('G(a)', fontsize=8)
    axes[1].tick_params(labelsize=6)
    axes[1].grid(True, axis='y', linestyle=':')
    axes[1].legend(loc='center right', fontsize=8)

    plt.tight_layout()
    plt.show()
