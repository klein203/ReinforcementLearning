import logging
import gym
from gym_study.agent import DQNAgent
from gym_study.policy import EpsilonGreedyPolicy, GreedyPolicy


def dqn_agent_train():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    env = gym.make('CartPole-v0')
    nb_features = env.observation_space.shape[0]
    nb_actions = env.action_space.n

    train_policy = EpsilonGreedyPolicy(nb_actions)    # epsilon = 0.1
    eval_policy = GreedyPolicy(nb_actions)

    agent = DQNAgent(nb_features, nb_actions, train_policy, eval_policy)

    weight_filename = 'dqn_weights.h5'
    agent.train(env, nb_episodes=1000, nb_warmup_steps=200, weight_filename=weight_filename, load_config=False)

    env.close()

def dqn_agent_play():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    env = gym.make('CartPole-v0')
    nb_features = env.observation_space.shape[0]
    nb_actions = env.action_space.n

    train_policy = EpsilonGreedyPolicy(nb_actions)    # epsilon = 0.1
    eval_policy = GreedyPolicy(nb_actions)

    agent = DQNAgent(nb_features, nb_actions, train_policy, eval_policy)

    weight_filename = 'dqn_weights.h5'
    agent.play(env, nb_episodes=10, weight_filename=weight_filename, render_mode=True)

    env.close()