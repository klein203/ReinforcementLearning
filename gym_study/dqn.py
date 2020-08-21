import os
import logging
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras.losses import MSE
import gym
from tensorflow.keras.callbacks import TensorBoard
import time


class DQNAgent(object):
    def __init__(self, n_features, n_actions, learning_rate=1e-4, epsilon=0.1, reward_decay=0.9,\
        e_greedy=0.9, weight_sync_interval=500, e_greedy_increment=None, batch_size=32, memory_size=500):
        self.n_features = n_features
        self.n_actions = n_actions
        self.gamma = 0.9  # decay rate
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay    # gamma
        
        self.weight_sync_interval = weight_sync_interval
        self.batch_size = batch_size
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = epsilon
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0

        self.optimizer = RMSprop(self.learning_rate)

        # initialize replay memory D to capacity N
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 3))
        self.memory_counter = 0

        # initialize action-value function Q with random weights theta
        self.eval_net = self._build_model(n_features, n_actions)    # eval, predict; new, training
        
        # initialize target action-value function Q^ with random weights theta^ = theta
        self.target_net = self._build_model(n_features, n_actions)  # target, fact
        
        self.tensorboard = TensorBoard(log_dir='logs/DQN')

        self.nb_train = 0

    def _build_model(self, input_dim, out_dim):
        model = Sequential()
        # fc: input_dim + 1 x 64
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        # fc: 64 + 1 x 64
        model.add(Dense(64, activation='relu'))
        # fc: 64 + 1 x out_dim
        model.add(Dense(out_dim))
        model.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        return model
    
    def choose_action(self, obs):
        obs = obs.reshape((1, -1))

        # With probability epsilon, select a random action a_t
        # otherwise select a_t = argmax_a Q(obs(s_t), a; theta)
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a_probs = self.eval_net.predict(obs)
            a = np.argmax(a_probs)
        return a

    def save_transition(self, obs, a, r, done, obs_):
        transition = np.hstack((obs, a, r, done, obs_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.memory_counter < 200:
            return 
            
        # sample random minibatch of transitions (obs_j, a_j, r_j, obs_j+1) from D
        sample_index = np.random.choice(min(self.memory_size, self.memory_counter), size=self.batch_size)
        batch = self.memory[sample_index, :]

        # compute q_target
        # magic
        q_eval_tmp = self.eval_net.predict(batch[:, :self.n_features])
        q_target = q_eval_tmp.copy()
        
        # batch index
        batch_idx = np.arange(self.batch_size, dtype=np.int32)
        # only update parameters with obs_j, a_j
        a = batch[:, self.n_features].astype(int)

        r = batch[:, self.n_features+1]
        q_target_obs_ = self.target_net.predict(batch[:, -self.n_features:])
        done = batch[:, self.n_features+2].astype(int)

        q_target[batch_idx, a] = r + self.gamma * np.max(q_target_obs_, axis=1) * (1 - done)
        
        # perform a gradient descent step on (y_target - Q_eval(obs_j, a_j; theta))^2 
        # with respect to the network parameters theta
        self.eval_net.fit(batch[:, :self.n_features], q_target, epochs=self.nb_train+1, initial_epoch=self.nb_train, verbose=2, callbacks=[self.tensorboard])
        self.nb_train += 1

        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max

        # every C steps, reset target_net = eval_net, that is to sync theta_eval to theta_target
        self.learn_step_counter += 1
        if self.learn_step_counter % self.weight_sync_interval == 0:
            self.sync_weights()

    def sync_weights(self):
        self.target_net.set_weights(self.eval_net.get_weights())
        # logging.info('sync weights from eval net paras to target net paras')
    
    def save_weights(self, filename):
        self.eval_net.save_weights(filename)
    
    def load_weights(self, filename):
        self.eval_net.load_weights(filename)
        self.sync_weights()


def train(env, agent):
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

            obs = obs_.copy()

            if done:
                logging.info('episode: %d|%d, score: %d' % (i_episode, n_episodes, tick))
                break
    
        if i_episode % 50 == 0:
            agent.save_weights(weight_filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    env = gym.make('CartPole-v0')
    agent = DQNAgent(4, 2, learning_rate=1e-2, reward_decay=0.9, e_greedy=0.9,\
        weight_sync_interval=10, memory_size=4000)
    train(env, agent)
    env.close()
