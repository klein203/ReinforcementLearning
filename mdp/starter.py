# import logging
import os
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import mdp
from mdp.env import Maze2DEnv, MarkovDecisionProcess as MDP
from mdp.agent import InteractiveAgent, QLearningAgent, SarsaAgent


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
    p_data = np.array([
        ['high', 'search', 'high', r_research, alpha],
        ['high', 'search', 'low', r_research, 1-alpha],
        ['low', 'search', 'high', -3, 1-beta],
        ['low', 'search', 'low', r_research, beta],
        ['high', 'wait', 'high', r_wait, 1],
        ['high', 'wait', 'low', r_wait, 0], # ignore
        ['low', 'wait', 'high', r_wait, 0], # ignore
        ['low', 'wait', 'low', r_wait, 1],
        ['low', 'recharge', 'high', 0, 1],
        ['low', 'recharge', 'low', 0, 0], # ignore
    ])
    states_space = ['high', 'low']
    actions_space = ['search', 'wait', 'recharge']

    mdp = MDP(states_space, actions_space, p_data, discount_factor=0.9)

    s = 'high'
    for i_step in range(n_step):
        actions = mdp.get_actions(s)
        a = np.random.choice(actions)
        s_ = mdp.get_next_state(s, a)
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
