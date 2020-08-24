# -*- coding: utf-8 -*-
import gym
import numpy as np
import copy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop


class DQNAgent(object):
    def __init__(self, env):
        self.env = env
        self.memory = []
        self.gamma = 0.9  # decay rate
        self.epsilon = 1  # exploration
        self.epsilon_decay = .995
        self.epsilon_min = 0.1
        self.learning_rate = 1e-4
        self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=4, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate))
        self.model = model
    
    def remember(self, obs, a, r, obs_, done):
        self.memory.append((obs, a, r, obs_, done))
    
    def choose_action(self, obs):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        act_values = self.model.predict(obs)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        batches = min(batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), batches)
        for i in batches:
            obs, a, r, obs_, done = self.memory[i]
            target = r
            if not done:
                target = r + self.gamma * \
                       np.amax(self.model.predict(obs_)[0])
            target_f = self.model.predict(obs)
            target_f[0][a] = target
            self.model.fit(obs, target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    n_episodes = 5000

    env = gym.make('CartPole-v0')
    agent = DQNAgent(env)

    for i_episode in range(n_episodes):
        obs = env.reset()
        obs = obs.reshape((1, 4))
  
        for tick in range(500):
            # env.render()
  
            a = agent.choose_action(obs)
  
            obs_, r, done, _ = env.step(a)
            obs_ = np.reshape(obs_, (1, 4))
  
            # reward缺省为1
            # 在每一个agent完成了目标的帧agent都会得到回报
            # 并且如果失败得到-100
            r = -100 if done else r
  
            agent.remember(obs, a, r, obs_, done)
  
            obs = copy.deepcopy(obs_)
  
            if done:
                print("episode: {}/{}, score: {}".format(i_episode, n_episodes, tick))
                break
        
        agent.replay(32)
