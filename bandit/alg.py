import logging
import numpy as np
import pandas as pd
from utils.history import History


class BanditAlgorithm(object):
    def __init__(self, env, *args, **kwargs):
        self.name = 'BanditAlgorithm Abstract'
        self.env = env
        self.k = env.get_k()
        self.q = np.zeros(self.k)
        self.n_times = np.zeros(self.k)
        self.history = History()
    
    def reset(self):
        self.q = np.zeros(self.k)
        self.n_times = np.zeros(self.k)

    def store_params(self, prop, episode, val):
        self.history.store(prop, episode, val)
    
    def _update(self, action_idx, action_idx_reward):
        raise NotImplementedError()

    def _run(self, policy, i_episode):
        q_step = np.zeros(self.n_steps)

        for i_step in range(self.n_steps):
            action_idx, _ = policy.choose(self.q, self.n_times, i_step+1)
            action_idx_reward = self.env.get_reward(action_idx)

            self._update(action_idx, action_idx_reward)
            
            # history gathering
            q_step[i_step] = action_idx_reward
        
        # Q_step: n_episode x n_step
        self.store_params('Q_steps', i_episode, q_step)
        # N_arm: n_episode x n_arms
        self.store_params('N_actions', i_episode, self.n_times)
    
    def run_episodes(self, policy, n_episodes=2000, n_steps=1000):
        self.history.clear()
        self.policy = policy
        self.n_episodes = n_episodes
        self.n_steps = n_steps

        for i_episode in range(n_episodes):
            self.reset()
            self._run(policy, i_episode)
            
    def report(self):
        logging.info('------------------------------------------------------------')
        logging.info("%s_%s_EP%d_ST%d" % (self.name, self.policy.name, self.n_episodes, self.n_steps))
        logging.info('------------------------------------------------------------')

        q_steps = self.history.get('Q_steps')
        q_steps_mean = np.mean(q_steps, axis=0)
        logging.debug('Q Mean Value: %s' % q_steps_mean)
        logging.info('Final Q Mean Value: %.4f' % q_steps_mean[-1])

        n_actions = self.history.get('N_actions')
        n_actions_mean = np.mean(n_actions, axis=0)
        logging.info('N Mean Value: %s' % n_actions_mean)


class MeanValueUpdAlg(BanditAlgorithm):
    """
    Mean Q value update algorithm, basic one
    """
    def __init__(self, env, *args, **kwargs):
        super(MeanValueUpdAlg, self).__init__(env, *args, **kwargs)
        self.name = 'ActionValueMethods'
        self.sum_q = np.zeros(self.k)
    
    def reset(self):
        super(MeanValueUpdAlg, self).reset()
        self.sum_q = np.zeros(self.k)

    def _update(self, action_idx, action_idx_reward):
        self.sum_q[action_idx] += action_idx_reward
        self.n_times[action_idx] += 1
        self.q[action_idx] = self.sum_q[action_idx] / self.n_times[action_idx]


class IncrementalValueUpdAlg(BanditAlgorithm):
    """
    Incremental Q value update algorithm.
    """
    def __init__(self, env, *args, **kwargs):
        super(IncrementalValueUpdAlg, self).__init__(env, *args, **kwargs)
        self.name = 'IncrementalImplementation'
    
    def _update(self, action_idx, action_idx_reward):
        self.n_times[action_idx] = self.n_times[action_idx] + 1
        self.q[action_idx] = self.q[action_idx] + 1 / self.n_times[action_idx] * (action_idx_reward - self.q[action_idx])
    
    def sampling_alpha(self, size=1000):
        self.reset()
        return pd.Series([1/i for i in range(1, size+1)], name='incremental val upd')


class ExpRecencyWeightedAvgAlg(BanditAlgorithm):
    """
    Exponetial recency-weighted average algorithm
    """
    def __init__(self, env, step_size=0.1, *args, **kwargs):
        super(ExpRecencyWeightedAvgAlg, self).__init__(env, *args, **kwargs)
        self.name = 'ExponentialRecencyWeightedAverage'
        # alpha
        self.alpha = step_size

    def _update(self, action_idx, action_idx_reward):
        # exponential recency-weighted average: the bigger the i is, the bigger the weight of R_i is
        self.n_times[action_idx] = self.n_times[action_idx] + 1
        self.q[action_idx] = self.q[action_idx] + self.alpha * (action_idx_reward - self.q[action_idx])
    
    def sampling_alpha(self, size=1000):
        self.reset()
        return pd.Series([self.alpha] * size, name='exp recency-weighted avg')


class BetaMoveStepAlg(BanditAlgorithm):
    """
    BetaMoveStep algorithm?
    β = α / ο_n
    ο_n = ο_n-1 + α * (1 - ο_n-1),  ο_0 = 0
    """
    def __init__(self, env, step_size=0.1, *args, **kwargs):
        super(BetaMoveStepAlg, self).__init__(env, *args, **kwargs)
        self.name = 'BetaMoveStepAlg'
        # alpha
        self.alpha = step_size
        self.omicron = np.zeros(self.k)
        self.beta = np.zeros(self.k)

    def _update(self, action_idx, action_idx_reward):
        self.n_times[action_idx] = self.n_times[action_idx] + 1
        self.omicron[action_idx] = (1 - self.alpha) * self.omicron[action_idx] + self.alpha
        self.beta[action_idx] = self.alpha / self.omicron[action_idx]
        self.q[action_idx] = self.q[action_idx] + self.beta[action_idx] * (action_idx_reward - self.q[action_idx])
    
    def sampling_alpha(self, size=1000):
        self.reset()
        alpha = self.alpha
        omicron = 0
        beta = 0
        samples = []
        for i in range(size):
            omicron = (1 - alpha) * omicron + alpha
            beta = alpha / omicron
            samples.append(beta)

        return pd.Series(samples, name='beta step move')
