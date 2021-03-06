import logging
import os
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import mdp
from mdp.env import Maze2DEnv, MarkovEnv
from mdp.agent import InteractiveAgent, QLearningAgent, SarsaAgent, SarsaLambdaAgent, \
    PolicyIteration, ValueIteration


def interactive_agent_run():
    """
    Maze2D manual play mode, interact with human
    """
    env = Maze2DEnv(config=mdp.default_maze_config)

    player = InteractiveAgent(env)
    player.play()

def ch3_1_autocleaner(n_step=20):
    """
    chapter 3.1 (page 50)
    a typical MDP process demo
    """
    cfg = mdp.autocleaner_mdp_config
    env = MarkovEnv(cfg.get('states_space'),
        cfg.get('actions_space'),
        cfg.get('rewards_space'),
        cfg.get('p_matrix'))

    s = 'high'
    logging.info('init s = %s' % str(s))
    for i_step in range(50):
        a = np.random.choice(env.get_actions(s))
        s_ = env.get_s_(s, a)

        logging.info('#%d: Battery[%s], Action[%s] -> Battery[%s]' % (i_step+1, s, a, s_))
        if env.is_terminal(s_):
            logging.info('end')
            break
        s = s_

def ch3_qlearning(mode='train', n_episode=200, path='.', filename='q_learning', load_file=False, silent_mode=False):
    env = Maze2DEnv(config=mdp.default_maze_config)
    agent = QLearningAgent(env)
    agent.silent_mode = silent_mode
    if mode == 'train':
        if load_file:
            agent.train(n_episode, path=path, save_fname=filename, load_fname=filename)
        else:
            agent.train(n_episode, path=path, save_fname=filename)
    elif mode == 'play':
        agent.play(path=path, load_fname=filename)
    else:
        raise Exception('invalid mode')

def ch3_sarsa(mode='train', n_episode=200, path='.', filename='sarsa', load_file=False, silent_mode=False):
    env = Maze2DEnv(config=mdp.default_maze_config)
    agent = SarsaAgent(env)
    agent.silent_mode = silent_mode
    if mode == 'train':
        if load_file:
            agent.train(n_episode, path=path, save_fname=filename, load_fname=filename)
        else:
            agent.train(n_episode, path=path, save_fname=filename)
    elif mode == 'play':
        agent.play(path=path, load_fname=filename)
    else:
        raise Exception('invalid mode')

def ch3_sarsa_lambda(mode='train', n_episode=200, path='.', filename='sarsa_lambda', load_file=False, silent_mode=False):
    env = Maze2DEnv(config=mdp.default_maze_config)
    agent = SarsaLambdaAgent(env)
    agent.silent_mode = silent_mode
    if mode == 'train':
        if load_file:
            agent.train(n_episode, path=path, save_fname=filename, load_fname=filename)
        else:
            agent.train(n_episode, path=path, save_fname=filename)
    elif mode == 'play':
        agent.play(path=path, load_fname=filename)
    else:
        raise Exception('invalid mode')

def ch4_3_gridworld_policy_iteration():
    env = Maze2DEnv(config=mdp.gridworld_config)
    agent = PolicyIteration(env)
    agent.policy_iter()

    # report
    logging.info('Policy Iteration - π*(v*)')
    for r in range(env.maze_nrows):
        line = ''
        for c in range(env.maze_ncols):
            if env.is_terminal((c, r)):
                line = '%s\t%s(%s)' % (line, '--', '-.--')
            else:
                line = '%s\t%s(%.2f)' % (line, env.actions_space[agent.action_policy[env.s((c, r))]], agent.get_v((c, r)))
        logging.info(line)

def ch4_4_gridworld_value_iteration():
    env = Maze2DEnv(config=mdp.gridworld_config)
    agent = ValueIteration(env)
    agent.value_iter()

    # report
    logging.info('Value Iteration - π*(v*)')
    for r in range(env.maze_nrows):
        line = ''
        for c in range(env.maze_ncols):
            if env.is_terminal((c, r)):
                line = '%s\t%s(%s)' % (line, '--', '-.--')
            else:
                line = '%s\t%s(%.2f)' % (line, env.actions_space[agent.action_policy[env.s((c, r))]], agent.get_v((c, r)))
        logging.info(line)

