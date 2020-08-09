# import logging
import os
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import mdp
from mdp.env import Maze2DEnv, MarkovEnv
from mdp.agent import InteractiveAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent


def interactive_agent_run():
    """
    Maze2D manual play mode, interact with human
    """
    env = Maze2DEnv(config=mdp.default_maze_config)

    player = InteractiveAgent(env)
    player.play()

def ch3_6(n_step=20):
    """
    chapter 3.6 (page xxx)
    a typical MDP process demo
    """
    alpha = 0.8
    beta = 0.7
    r_research = 1.5
    r_wait = 1.0

    states_space = ['high', 'low']
    actions_space = ['search', 'wait', 'recharge']
    rewards_space = [-3, 0, r_wait, r_research]

    p_matrix = np.zeros((len(states_space), len(actions_space), len(states_space), len(rewards_space)))
    env = MarkovEnv(states_space, actions_space, rewards_space, p_matrix)
    env.set_prob('high', 'search', 'high', r_research, alpha)
    env.set_prob('high', 'search', 'low', r_research, 1-alpha)
    env.set_prob('low', 'search', 'high', -3, 1-beta)
    env.set_prob('low', 'search', 'low', r_research, beta)
    env.set_prob('high', 'wait', 'high', r_wait, 1)
    env.set_prob('low', 'wait', 'low', r_wait, 1)
    env.set_prob('low', 'recharge', 'high', 0, 1)

    s = 'high'
    for i_step in range(n_step):
        actions = env.get_actions(s)
        a = np.random.choice(actions)
        s_ = env.get_s_(s, a)
        print('#%d: Battery[%s], Action[%s] -> Battery[%s]' % (i_step+1, s, a, s_))
        s = s_

def ch3_qlearning(mode='train', n_episode=200, filename='q_learning', load_file=False, silent_mode=False):
    env = Maze2DEnv(config=mdp.default_maze_config)
    agent = QLearningAgent(env)
    agent.silent_mode = silent_mode
    if mode == 'train':
        if load_file:
            agent.train(n_episode, save_fname=filename, load_fname=filename)
        else:
            agent.train(n_episode, save_fname=filename)
    elif mode == 'play':
        agent.play(load_fname=filename)
    else:
        raise Exception('invalid mode')

def ch3_sarsa(mode='train', n_episode=200, filename='sarsa', load_file=False, silent_mode=False):
    env = Maze2DEnv(config=mdp.default_maze_config)
    agent = SarsaAgent(env)
    agent.silent_mode = silent_mode
    if mode == 'train':
        if load_file:
            agent.train(n_episode, save_fname=filename, load_fname=filename)
        else:
            agent.train(n_episode, save_fname=filename)
    elif mode == 'play':
        agent.play(load_fname=filename)
    else:
        raise Exception('invalid mode')

def ch3_sarsa_lambda(mode='train', n_episode=200, filename='sarsa_lambda', load_file=False, silent_mode=False):
    env = Maze2DEnv(config=mdp.default_maze_config)
    agent = SarsaLambdaAgent(env)
    agent.silent_mode = silent_mode
    if mode == 'train':
        if load_file:
            agent.train(n_episode, save_fname=filename, load_fname=filename)
        else:
            agent.train(n_episode, save_fname=filename)
    elif mode == 'play':
        agent.play(load_fname=filename)
    else:
        raise Exception('invalid mode')

