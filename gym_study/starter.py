import logging
import gym
from gym_study.agent import DQNAgent
from gym_study.policy import EpsilonGreedyPolicy, GreedyPolicy


def dqn_agent_train(weight_filename: str, nb_episodes: int = 500, load_config: bool = False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    env = gym.make('CartPole-v1')
    nb_features = env.observation_space.shape[0]
    nb_actions = env.action_space.n

    train_policy = EpsilonGreedyPolicy(nb_actions)    # epsilon = 0.1
    eval_policy = GreedyPolicy(nb_actions)

    agent = DQNAgent(nb_features, nb_actions, train_policy, eval_policy)

    agent.train(env, nb_episodes=nb_episodes, nb_warmup_steps=200, weight_filename=weight_filename, load_config=load_config)

    env.close()

def dqn_agent_play(weight_filename: str, nb_episodes: int = 10):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    env = gym.make('CartPole-v1')
    nb_features = env.observation_space.shape[0]
    nb_actions = env.action_space.n

    train_policy = EpsilonGreedyPolicy(nb_actions)    # epsilon = 0.1
    eval_policy = GreedyPolicy(nb_actions)

    agent = DQNAgent(nb_features, nb_actions, train_policy, eval_policy)

    agent.eval(env, nb_episodes=nb_episodes, weight_filename=weight_filename, render_mode=True)

    env.close()
