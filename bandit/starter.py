import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bandit.env import MultiArmedBanditEnv, NonstationaryMultiArmedBanditEnv
from bandit.policy import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy, UpperConfidenceBoundPolicy, SoftmaxPolicy
from bandit.alg import IncrementalValueUpdAlg, ExpRecencyWeightedAvgAlg, BetaMoveStepAlg, GradientBanditAlg


def ch2_3(n_episodes=100, n_steps=1000):
    """
    chapter 2.3 (page 29)
    1) Reward distribution of arms
    2) N distribution of arms using incremental value update algorithm on e-greedy(0.1)
    3) Q trends over steps with various policies (random, e-greedy(0.5; 0.1; 0.01), greedy(0))
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
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='random'))

    # greedy, e=0.0
    agent.run_episodes(GreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='greedy'))

    # e-greedy, e=0.01
    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.01), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='e=0.01'))

    # e-greedy, e=0.5
    agent.run_episodes(EpsilonGreedyPolicy(epsilon=0.5), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='e=0.5'))

    # e-greedy, e=0.1
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='e=0.1'))
    
    # plot Q trends over steps with various policies
    sns.lineplot(data=q_steps_list, size=0.5, ax=ax3)

    # plot N distribution using e-greedy(0.1) policy
    # by using e-greedy policy, action reward with higher target mean value in the first place 
    # has much higher possiblity to be choosed
    n_actions_mean = np.mean(agent.history.get('NTimes_actions'), axis=0)
    df = pd.DataFrame(data=n_actions_mean.reshape(1, -1), columns=env.actions_space)
    sns.barplot(data=df, ax=ax2)

    # plot show
    plt.tight_layout()
    plt.show()

def ch2_5(n_episodes=100, n_steps=10000):
    """
    chapter 2.5 (page 33)
    1) Reward distribution of arms in non-stationary environment
    2) N distribution of arms using exponential recency-weighted average algorithm on e-greedy(0.1)
    3) Q trends over steps with various algorithms (incremental val upd + e-greedy(0.1); exp recency-weighted avg + e-greedy(0.1))
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
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='incremental, e-greedy(e=0.1)'))

    # exponential recency weighted average algorithm with e-greedy(0.1) policy
    agent = ExpRecencyWeightedAvgAlg(env, step_size=0.1)
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='step move(a=0.1), e-greedy(e=0.1)'))

    # beta move step algorithm with e-greedy(0.1) policy
    # agent = BetaMoveStepAlg(env, step_size=0.1)
    # agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    # agent.report()
    # q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='beta step move(a=0.1), e-greedy(e=0.1)'))

    # plot Q trends over steps with various policies
    sns.lineplot(data=q_steps_list, size=0.5, ax=ax3)

    # plot N distribution using exponential recency weighted average algorithm with e-greedy(0.1) policy
    n_actions_mean = np.mean(agent.history.get('NTimes_actions'), axis=0)
    df = pd.DataFrame(data=n_actions_mean.reshape(1, -1), columns=env.actions_space)
    sns.barplot(data=df, ax=ax2)

    # plot show
    plt.tight_layout()
    plt.show()

def move_step_value_trends():
    """
    move step value trends diagram
    1) move step trends over times with various algorithms (incremental val upd; exp recency-weighted avg; beta)
    """
    # plots init and config
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 5))
    
    # 1st diagram format
    ax1.set_title('Move Step Value Trends', fontsize=10)
    ax1.set_xlabel('Time', fontsize=8)
    ax1.set_ylabel('Move step value', fontsize=8)
    ax1.tick_params(labelsize=6)
    ax1.grid(True, axis='y', linestyle=':')
    ax1.legend(loc='lower right', fontsize=7)

    env = MultiArmedBanditEnv(['Dummy'])
    sample_size = 50
    sample_list = []

    algo = IncrementalValueUpdAlg(env)
    sample_list.append(algo.sampling_alpha(sample_size))

    algo = ExpRecencyWeightedAvgAlg(env, step_size=0.1)
    sample_list.append(algo.sampling_alpha(sample_size))

    algo = BetaMoveStepAlg(env, step_size=0.1)
    sample_list.append(algo.sampling_alpha(sample_size))

    sns.lineplot(data=sample_list, size=0.5, ax=ax1)

    # plot show
    plt.tight_layout()
    plt.show()

def ch2_7(n_episodes=1000, n_steps=1000):
    """
    chapter 2.7 (page 36)
    1) Q trends over steps with various policies (ubc(c=2); e-greedy(0.1))
    """
    # plots init and config
    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))

    # 10-armed bandit enviroment
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(n_arm)]
    env = MultiArmedBanditEnv(actions_space)
    env.info()

    # params storing
    q_steps_list = []

    agent = IncrementalValueUpdAlg(env)
    agent.run_episodes(EpsilonGreedyPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='e-greedy(0.1)'))

    agent.run_episodes(UpperConfidenceBoundPolicy(), n_episodes, n_steps)
    agent.report()
    q_steps_list.append(pd.Series(np.mean(agent.history.get('Q(a)_steps'), axis=0), name='ubc(2)'))

    # plot trends over steps with various policies
    sns.lineplot(data=q_steps_list, size=0.5, ax=ax1)

    # plot show
    ax1.set_title('Q Trends', fontsize=10)
    ax1.set_xlabel('Steps', fontsize=8)
    ax1.set_ylabel('Q(a)', fontsize=8)
    ax1.tick_params(labelsize=6)
    ax1.grid(True, axis='y', linestyle=':')
    ax1.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.show()

def ch2_8(n_episodes=1000, n_steps=1000):
    """
    chapter 2.8 (page 38)
    1) % optimal action over steps (gradient algo: a=0.1/0.4 with/without baseline)
    """
    # plots init and config
    fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))

    # 10-armed bandit enviroment
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(n_arm)]
    env = MultiArmedBanditEnv(actions_space)
    env.set_reward_target_dist_mean(4.0, 1.0)
    env.info()
    optimal_action = env.get_optimal_target_action()
    logging.info('Optimal Action Index = %d' % optimal_action)

    # params storing
    optimal_actions_percentage = []

    agent = GradientBanditAlg(env, step_size=0.1, r_baseline=0)
    agent.run_episodes(SoftmaxPolicy(), n_episodes, n_steps)
    agent.report()
    # optimal_action = np.argmax(np.mean(agent.history.get('NTimes_actions'), axis=0))
    actions_steps = agent.history.get('actions_steps')
    optimal_actions_percentage.append(pd.Series(np.mean((actions_steps==optimal_action), axis=0), name='a=0.1 without baseline'))

    agent = GradientBanditAlg(env, step_size=0.1, r_baseline=4)
    agent.run_episodes(SoftmaxPolicy(), n_episodes, n_steps)
    agent.report()
    # optimal_action = np.argmax(np.mean(agent.history.get('NTimes_actions'), axis=0))
    actions_steps = agent.history.get('actions_steps')
    optimal_actions_percentage.append(pd.Series(np.mean((actions_steps==optimal_action), axis=0), name='a=0.1 with baseline'))

    agent = GradientBanditAlg(env, step_size=0.4, r_baseline=0)
    agent.run_episodes(SoftmaxPolicy(), n_episodes, n_steps)
    agent.report()
    # optimal_action = np.argmax(np.mean(agent.history.get('NTimes_actions'), axis=0))
    actions_steps = agent.history.get('actions_steps')
    optimal_actions_percentage.append(pd.Series(np.mean((actions_steps==optimal_action), axis=0), name='a=0.4 without baseline'))

    agent = GradientBanditAlg(env, step_size=0.4, r_baseline=4)
    agent.run_episodes(SoftmaxPolicy(), n_episodes, n_steps)
    agent.report()
    # optimal_action = np.argmax(np.mean(agent.history.get('NTimes_actions'), axis=0))
    actions_steps = agent.history.get('actions_steps')
    optimal_actions_percentage.append(pd.Series(np.mean((actions_steps==optimal_action), axis=0), name='a=0.4 with baseline'))

    # plot trends
    sns.lineplot(data=optimal_actions_percentage, size=0.5, ax=ax1)

    # plot show
    ax1.set_title('Optimal Actions Percentage', fontsize=10)
    ax1.set_xlabel('Steps', fontsize=8)
    ax1.set_ylabel('% Optimal Actions', fontsize=8)
    ax1.tick_params(labelsize=6)
    ax1.grid(True, axis='y', linestyle=':')
    ax1.legend(loc='lower right', fontsize=7)

    plt.tight_layout()
    plt.show()

def test(n_episodes=100, n_steps=1000):
    """
    for test
    """
    n_arm = 10
    actions_space = ['Arm%d' % i for i in range(n_arm)]
    env = MultiArmedBanditEnv(actions_space)
    env.set_reward_target_dist_mean(0.0, 1.0)
    env.info()
    optimal_action = env.get_optimal_target_action()
    logging.info('Optimal Action Index = %d' % optimal_action)

    agent = GradientBanditAlg(env, step_size=0.1, r_baseline=0)
    agent.run_episodes(SoftmaxPolicy(), n_episodes, n_steps)
    agent.report()
    # actions_steps = agent.history.get('actions_steps')
    # logging.info(np.mean((actions_steps==optimal_action), axis=0))
    
    # hs = agent.history.get('H(a)_steps')
    # logging.info(hs)
    