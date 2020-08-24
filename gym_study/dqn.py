import os
import logging
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras.losses import MSE
import gym
import time
from copy import deepcopy
from buffer.replay import ReplayBuffer
from network.DQN import DeepQNetwork
from policy.policy import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy


class DQNAgent(object):
    def __init__(self, nb_features, nb_actions, train_policy, eval_policy, \
        reward_decay=0.9, learning_rate=1e-3, weight_sync_interval=500, \
        memory_size=5000, batch_size=32, \
        nb_iterations=100):

        self.nb_features = nb_features
        self.nb_actions = nb_actions
        self.reward_decay = reward_decay  # decay rate
        self.learning_rate = learning_rate
        
        self.weight_sync_interval = weight_sync_interval
        self.batch_size = batch_size
        
        self.epsilon = epsilon

        self.learn_step_counter = 0

        self.optimizer = RMSprop(self.learning_rate)

        # initialize replay memory D to capacity N
        self.replay_buffer = ReplayBuffer(self.nb_features * 2 + 3, self.memory_size, self.batch_size)

        # initialize action-value function Q with random weights theta
        self.main_net = DeepQNetwork(self.nb_features, self.nb_actions)    # training, predict network
        
        # initialize target action-value function Q^ with random weights theta^ = theta
        self.target_net = self.main_net.clone_model() # same structure as main_net, target network
        
        self.nb_train = 0

   
    def choose_action(self, obs):
        obs = obs.reshape((1, -1))

        # With probability epsilon, select a random action a_t
        # otherwise select a_t = argmax_a Q(obs(s_t), a; theta)
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.nb_actions)
        else:
            a_probs = self.main_net.predict(obs)
            a = np.argmax(a_probs)
        return a

    def save_transition(self, obs, a, r, done, obs_):
        transition = np.hstack((obs, a, r, done, obs_))
        self.replay_buffer.append(transition)
    
    def train(self):
        if self.memory_counter < 200:
            return 
            
        # sample random minibatch of transitions (obs_j, a_j, r_j, obs_j+1) from D
        batch = self.replay_buffer.sample()

        # compute q_target
        # magic
        q_eval_tmp = self.main_net.predict(batch[:, :self.nb_features])
        q_target = deepcopy(q_eval_tmp)
        
        # batch index
        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        # only update parameters with obs_j, a_j
        a = batch[:, self.nb_features].astype(int)

        r = batch[:, self.nb_features+1]
        q_target_obs_ = self.target_net.predict(batch[:, -self.nb_features:])
        done = batch[:, self.nb_features+2].astype(int)

        q_target[batch_idx, a] = r + self.reward_decay * np.max(q_target_obs_, axis=1) * (1 - done)
        
        # perform a gradient descent step on (y_target - Q_eval(obs_j, a_j; theta))^2 
        # with respect to the network parameters theta
        self.main_net.fit(batch[:, :self.nb_features], q_target, epochs=self.nb_train+1, initial_epoch=self.nb_train, verbose=2)
        self.nb_train += 1

        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # every C steps, reset target_net = main_net, that is to sync theta_eval to theta_target
        self.learn_step_counter += 1
        if self.learn_step_counter % self.weight_sync_interval == 0:
            self.sync_weights()

    def sync_weights(self):
        self.target_net.set_weights(self.main_net.get_weights())
    
    def save_weights(self, filename):
        self.main_net.save_weights(filename)
    
    def load_weights(self, filename):
        self.main_net.load_weights(filename)
        self.sync_weights()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    env = gym.make('CartPole-v0')
    nb_features = env.observation_space.shape[0]
    nb_actions = env.action_space.n

    train_policy = EpsilonGreedyPolicy()    # epsilon = 0.1
    eval_policy = GreedyPolicy()

    agent = DQNAgent(nb_features, nb_actions, train_policy, eval_policy)

    weight_filename = 'dqn_weights.h5'
    if os.path.exists(weight_filename):
        agent.load_weights(weight_filename)

    # for episode = 1, M do
    n_episodes = 100
    for i_episode in range(1, n_episodes+1):
        # initialize sequence s_1 = {x_1} and preprocessed sequence obs_1 = obs(s_1)
        obs = env.reset()

        # for t = 1, T do
        for tick in range(500):
            # env.render()
            a = agent.choose_action(obs)

            # execute action a_t in emulator and observe reward r_t and image x_t+1
            # set s_t+1 = s_t, a_t, x_t+1 and preprocess obs_t+1 = obs(s_t+1)
            obs_, r, done, info = env.step(a)

            # store transition (obs_t, a_t, r_t, obs_t+1) in D
            agent.save_transition(obs, a, r, done, obs_)

            agent.learn()

            obs = deepcopy(obs_)

            if done:
                logging.info('episode: %d|%d, score: %d' % (i_episode, n_episodes, tick))
                break
    
        if i_episode % 50 == 0:
            agent.save_weights(weight_filename)
    
    env.close()
