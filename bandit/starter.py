import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bandit.env import MultiArmedBanditEnv, NonstationaryMultiArmedBanditEnv
from bandit.policy import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy
from bandit.alg import IncrementalValueUpdAlg, ExpRecencyWeightedAvgAlg


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

    # evaluation using incremental value update algorithm with various policies
    agent = IncrementalValueUpdAlg(env)

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

    # 10-armed non-stationary bandit enviroment
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(n_arm)]
    env = NonstationaryMultiArmedBanditEnv(actions_space)
    env.info()

    # plot reward target distribution of given environment
    df = env.sampling()
    sns.swarmplot(data=df, size=1, ax=ax1)

    # params storing
    q_steps_list = []

    # incremental implementation Q value update algorithm with e-greedy(0.1) policy
    agent = IncrementalValueUpdAlg(env)
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='incremental, e-greedy(e=0.1)'))

    # exponential recency weighted average algorithm with e-greedy(0.1) policy
    agent = ExpRecencyWeightedAvgAlg(env, step_size=0.1)
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q_steps'), axis=0), name='step move(a=0.1), e-greedy(e=0.1)'))

    # plot Q trends over steps with various policies
    sns.lineplot(data=q_steps_list, size=0.5, ax=ax3)

    # plot N distribution using exponential recency weighted average algorithm with e-greedy(0.1) policy
    n_actions_mean = np.mean(agent.history.get('N_actions'), axis=0)
    df = pd.DataFrame(data=n_actions_mean.reshape(1, -1), columns=env.actions_space)
    sns.barplot(data=df, ax=ax2)

    # plot show
    plt.tight_layout()
    plt.show()
