import os
import time
import logging
import numpy as np
from keras.optimizers import RMSprop
from keras.models import clone_model
from gym import Env
from copy import deepcopy
from gym_study.replay_buffer import ReplayBuffer
from gym_study.network import DeepQNetwork
from gym_study.policy import AbstractPolicy, EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy
from gym_study.utils import HistoryManager


class DQNAgent(object):
    def __init__(self, nb_features: int, nb_actions: int, train_policy: AbstractPolicy, eval_policy: AbstractPolicy, \
        reward_decay: float = 0.9, learning_rate: float = 1e-3, memory_size: int = 5000):

        self.nb_features = nb_features
        self.nb_actions = nb_actions
        self.reward_decay = reward_decay  # decay rate
        self.learning_rate = learning_rate
        
        self.train_policy = train_policy
        self.eval_policy = eval_policy

        self.optimizer = RMSprop(self.learning_rate)

        # initialize replay memory D to capacity N
        self.replay_buffer = ReplayBuffer(self.nb_features * 2 + 3, memory_size)

        # initialize action-value function Q with random weights theta
        self.main_net = DeepQNetwork(nb_features, nb_actions)    # training, predict network
        self.main_net.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        
        # initialize target action-value function Q^ with random weights theta^ = theta
        self.target_net = clone_model(self.main_net) # same structure as main_net, target network
        self.target_net.compile(optimizer=self.optimizer, loss='mse', metrics=['accuracy'])
        
        # training global step counter
        self.nb_global_steps = 0

        self.nb_learn_counter = 0

        # history manager
        self.history_manager = HistoryManager('histories.%f.pkl' % time.time())

    def store_transition(self, obs, a, r, done, obs_):
        transition = np.hstack((obs.flatten(), a, r, done, obs_.flatten()))
        self.replay_buffer.append(transition)

    def train(self, env: Env, nb_episodes: int = 100, nb_warmup_steps: int = 200, nb_steps_per_episode: int = 500, \
        render_mode: bool = False, load_config: bool = False, weight_filename: str = None, \
        nb_weight_sync_interval: int = 500, nb_weight_save_interval: int = 1000, batch_size: int = 32, \
        nb_eval_episode_interval: int = 20):

        self.nb_global_steps = 0
        self.nb_learn_counter = 0

        if load_config and os.path.exists(weight_filename):
            self.load_weights(weight_filename)

        # for episode = 0, M-1 do
        for i_episode in range(nb_episodes):
            # initialize sequence s_1 = {x_1} and preprocessed sequence obs_1 = obs(s_1)
            obs = env.reset()

            # for t = 0, T-1 do
            for t in range(nb_steps_per_episode):
                if render_mode:
                    env.render()
                
                # With probability epsilon, select a random action a_t
                # otherwise select a_t = argmax_a Q(obs(s_t), a; theta)
                obs = obs.reshape((1, -1))
                action_probs = self.main_net.predict(obs)
                a = self.train_policy.choose(action_probs)

                # execute action a_t in emulator and observe reward r_t and image x_t+1
                # set s_t+1 = s_t, a_t, x_t+1 and preprocess obs_t+1 = obs(s_t+1)
                obs_, r, done, _ = env.step(a)

                # store transition (obs_t, a_t, r_t, obs_t+1) in D
                self.store_transition(obs, a, r, done, obs_)

                # skip warmup steps
                if self.nb_global_steps >= nb_warmup_steps:
                    self.learn(batch_size)
                
                # every C steps, reset target_net = main_net, that is to sync theta_eval to theta_target
                if self.nb_learn_counter > 0 and self.nb_learn_counter % nb_weight_sync_interval == 0:
                    self.sync_weights()
                
                # save weights periodically
                if self.nb_learn_counter > 0 and self.nb_learn_counter % nb_weight_save_interval == 0:
                    logging.info('saving weights @%d times' % self.nb_learn_counter)
                    self.save_weights(weight_filename)

                obs = deepcopy(obs_)

                self.nb_global_steps += 1

                if done:
                    logging.info('Training Episode: %d|%d, Times: %d' % (i_episode+1, nb_episodes, t+1))
                    break
            
            # eval every nb_eval_episode_interval
            if self.nb_global_steps >= nb_warmup_steps and i_episode % nb_eval_episode_interval == 0:
                score = self._eval(env, nb_episodes=5, render_mode=False)
        
        # wrapup
        logging.info('Fin.')
        logging.info('saving weights @%d times' % self.nb_learn_counter)
        self.save_weights(weight_filename)
        logging.info('dump remaining histories to file')
        self.history_manager.dump()

    def learn(self, batch_size: int = 32):
        # sample random minibatch of transitions (obs_j, a_j, r_j, obs_j+1) from D
        batch = self.replay_buffer.sample(batch_size)

        # compute q_target
        # magic
        q_eval_tmp = self.main_net.predict(batch[:, :self.nb_features])
        q_target = deepcopy(q_eval_tmp)
        
        # batch index
        batch_idx = np.arange(batch_size, dtype=np.int32)
        # only update parameters with obs_j, a_j
        a = batch[:, self.nb_features].astype(int)

        r = batch[:, self.nb_features+1]
        q_target_obs_ = self.target_net.predict(batch[:, -self.nb_features:])
        done = batch[:, self.nb_features+2].astype(int)

        q_target[batch_idx, a] = r + self.reward_decay * np.max(q_target_obs_, axis=1) * (1 - done)
        
        # perform a gradient descent step on (y_target - Q_eval(obs_j, a_j; theta))^2 
        # with respect to the network parameters theta
        history = self.main_net.fit(batch[:, :self.nb_features], q_target, epochs=self.nb_learn_counter+1, \
            initial_epoch=self.nb_learn_counter, verbose=0)

        self.history_manager.save(history.history)

        self.nb_learn_counter += 1

    def _eval(self, env: Env, nb_episodes: int, render_mode: bool):
        rewards = np.zeros(nb_episodes)      
        for i_episode in range(nb_episodes):
            reward = 0
            obs = env.reset()

            while True:
                if render_mode:
                    env.render()
                
                obs = obs.reshape((1, -1))
                action_probs = self.main_net.predict(obs)
                a = self.eval_policy.choose(action_probs)

                obs_, r, done, _ = env.step(a)
                reward += r

                obs = deepcopy(obs_)

                if done:
                    rewards[i_episode] = reward
                    logging.info('Eval Episode: %d|%d, Reward: %d' % (i_episode+1, nb_episodes, reward))
                    break
        score = np.mean(rewards)
        logging.info('Mean Score for %d Episodes: %.2f' % (nb_episodes, score))

        return score

    def eval(self, env: Env, nb_episodes: int = 10, render_mode: bool = True, weight_filename: str = None):
        if os.path.exists(weight_filename):
            self.load_weights(weight_filename)

        return self._eval(env, nb_episodes, render_mode)
        
    def sync_weights(self):
        self.target_net.set_weights(self.main_net.get_weights())
    
    def save_weights(self, filename):
        self.main_net.save_weights(filename)
    
    def load_weights(self, filename):
        self.main_net.load_weights(filename)
        self.sync_weights()
