import logging
import numpy as np
import pandas as pd
from utils.history import History


class BanditAlgorithm(object):
    def __init__(self, env, *args, **kwargs):
        self.name = 'BanditAlgorithm Abstract'
        self.env = env
        self.k = env.get_k()
        self.vals = np.zeros(self.k)
        self.ntimes = np.zeros(self.k)
        self.history = History()
    
    def reset(self):
        self.vals = np.zeros(self.k)
        self.ntimes = np.zeros(self.k)

    def store_params(self, prop, episode, val):
        self.history.store(prop, episode, val)
    
    def _update(self, *args, **kwargs):
        raise NotImplementedError()

    def _run(self, policy, i_episode):
        raise NotImplementedError()
    
    def run_episodes(self, policy, n_episodes=2000, n_steps=1000):
        self.history.clear()
        self.policy = policy
        self.n_episodes = n_episodes
        self.n_steps = n_steps

        for i_episode in range(n_episodes):
            self.reset()
            self._run(policy, i_episode)
            
    def report(self):
        raise NotImplementedError()


class QValueBasedAlg(BanditAlgorithm):
    def __init__(self, env, *args, **kwargs):
        super(QValueBasedAlg, self).__init__(env, *args, **kwargs)
        self.name = 'Q Value Based Algorithm Abstract'
    
    def _run(self, policy, i_episode):
        vals_step = np.zeros(self.n_steps)
        actions_step = np.zeros(self.n_steps)

        for i_step in range(self.n_steps):
            # choose A_t and get R_t
            action, _ = policy.choose(self.vals, self.ntimes, i_step+1)
            action_reward = self.env.get_reward(action)
            
            # history gathering, store action_t and vals_t
            actions_step[i_step] = action
            vals_step[i_step] = action_reward

            # Vals_t+1 updating
            self._update(action, action_reward)
        
        # Q_step: n_episode x n_step
        self.store_params('Q(a)_steps', i_episode, vals_step)
        # N_arm: n_episode x n_arms
        self.store_params('NTimes_actions', i_episode, self.ntimes)
        # action choose per step: n_episode x n_steps
        self.store_params('actions_steps', i_episode, actions_step)

    def report(self):
        logging.info('------------------------------------------------------------')
        logging.info("%s_%s_EP%d_ST%d" % (self.name, self.policy.name, self.n_episodes, self.n_steps))
        logging.info('------------------------------------------------------------')

        q_steps = self.history.get('Q(a)_steps')
        q_steps_mean = np.mean(q_steps, axis=0)
        logging.debug('Q Mean Value: %s' % q_steps_mean)
        logging.info('Final Q Mean Value: %.4f' % q_steps_mean[-1])

        ntimes_actions = self.history.get('NTimes_actions')
        ntimes_actions_mean = np.mean(ntimes_actions, axis=0)
        logging.info('NTimes Mean Value: %s' % ntimes_actions_mean)


class MeanValueUpdAlg(QValueBasedAlg):
    """
    Mean Q value update algorithm, basic one
    """
    def __init__(self, env, *args, **kwargs):
        super(MeanValueUpdAlg, self).__init__(env, *args, **kwargs)
        self.name = 'Naive Mean Value Update Algorithm'
        self.vals_sum = np.zeros(self.k)
    
    def reset(self):
        super(MeanValueUpdAlg, self).reset()
        self.vals_sum = np.zeros(self.k)

    def _update(self, action, action_reward):
        self.vals_sum[action] += action_reward
        self.ntimes[action] += 1
        self.vals[action] = self.vals_sum[action] / self.ntimes[action]


class IncrementalValueUpdAlg(QValueBasedAlg):
    """
    Incremental Q value update algorithm.
    """
    def __init__(self, env, *args, **kwargs):
        super(IncrementalValueUpdAlg, self).__init__(env, *args, **kwargs)
        self.name = 'Incremental Value Update Algorithm'
    
    def _update(self, action, action_reward):
        self.ntimes[action] = self.ntimes[action] + 1
        self.vals[action] = self.vals[action] + 1 / self.ntimes[action] * (action_reward - self.vals[action])
    
    def sampling_alpha(self, size=1000):
        self.reset()
        return pd.Series([1/i for i in range(1, size+1)], name='incremental val upd')


class ExpRecencyWeightedAvgAlg(QValueBasedAlg):
    """
    Exponetial recency-weighted average algorithm
    """
    def __init__(self, env, step_size=0.1, *args, **kwargs):
        super(ExpRecencyWeightedAvgAlg, self).__init__(env, *args, **kwargs)
        # alpha, constant
        self.alpha_ori = step_size
        self.alpha = self.alpha_ori
        self.name = 'Exponential Recency Weighted Average Algorithm (a=%.2f)' % self.alpha

    def reset(self):
        super(ExpRecencyWeightedAvgAlg, self).reset()
        self.alpha = self.alpha_ori

    def _update(self, action, action_reward):
        # exponential recency-weighted average: the bigger the i is, the bigger the weight of R_i is
        self.ntimes[action] = self.ntimes[action] + 1
        self.vals[action] = self.vals[action] + self.alpha * (action_reward - self.vals[action])
    
    def sampling_alpha(self, size=1000):
        self.reset()
        return pd.Series([self.alpha] * size, name='exp recency-weighted avg')


class BetaMoveStepAlg(QValueBasedAlg):
    """
    BetaMoveStep algorithm?
    ο_t+1 = ο_t + α * (1 - ο_t) = (1 - α) * ο_t + α,  ο_0 = 0
    β_t+1 = α / ο_t+1
    """
    def __init__(self, env, step_size=0.1, *args, **kwargs):
        super(BetaMoveStepAlg, self).__init__(env, *args, **kwargs)
        # alpha, constant
        self.alpha_ori = step_size
        self.alpha = self.alpha_ori
        self.omicron = np.zeros(self.k)
        self.beta = np.zeros(self.k)        
        self.name = 'Beta Move Step Algorithm (a=%.2f)' % self.alpha

    def reset(self):
        super(BetaMoveStepAlg, self).reset()
        self.alpha = self.alpha_ori
        self.omicron = np.zeros(self.k)
        self.beta = np.zeros(self.k)

    def _update(self, action, action_reward):
        self.ntimes[action] = self.ntimes[action] + 1
        self.omicron[action] = (1 - self.alpha) * self.omicron[action] + self.alpha
        self.beta[action] = self.alpha / self.omicron[action]
        self.vals[action] = self.vals[action] + self.beta[action] * (action_reward - self.vals[action])
    
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


class PreferenceBasedAlg(BanditAlgorithm):
    def __init__(self, env, step_size=0.1, *args, **kwargs):
        super(PreferenceBasedAlg, self).__init__(env, *args, **kwargs)
        self.name = 'H-Preference Function Based Algorithm Abstract'
        # alpha, constant
        self.alpha_ori = step_size
        self.alpha = self.alpha_ori
        self.probabilities = np.zeros(self.k)

    def reset(self):
        super(PreferenceBasedAlg, self).reset()
        self.alpha = self.alpha_ori
        self.probabilities = np.zeros(self.k)

    def _update(self, action, action_reward, i_step):
        raise NotImplementedError()

    def _run(self, policy, i_episode):
        vals_step = np.zeros(self.n_steps)
        actions_step = np.zeros(self.n_steps)

        for i_step in range(self.n_steps):
            # choose A_t and get R_t
            action, self.probabilities = policy.choose(self.vals, self.ntimes, i_step+1)
            action_reward = self.env.get_reward(action)
            
            # history gathering, store action_t, val_t
            actions_step[i_step] = action
            vals_step[i_step] = action_reward

            # Vals_t+1 updating, (H(a)_t+1 in gradient alg)
            self._update(action, action_reward, i_step+1)
        
        # H(a)_step: n_episodes x n_steps
        self.store_params('H(a)_steps', i_episode, vals_step)
        # N_arm: n_episodes x n_arms
        self.store_params('NTimes_actions', i_episode, self.ntimes)
        # action choose per step: n_episode x n_steps
        self.store_params('actions_steps', i_episode, actions_step)
    
    def report(self):
        logging.info('------------------------------------------------------------')
        logging.info("%s_%s_EP%d_ST%d" % (self.name, self.policy.name, self.n_episodes, self.n_steps))
        logging.info('------------------------------------------------------------')

        vals_steps = self.history.get('H(a)_steps')
        vals_steps_mean = np.mean(vals_steps, axis=0)
        logging.debug('H Mean Value: %s' % vals_steps_mean)

        ntimes_actions = self.history.get('NTimes_actions')
        ntimes_actions_mean = np.mean(ntimes_actions, axis=0)
        logging.info('NTimes Mean Value: %s' % ntimes_actions_mean)


class GradientBanditAlg(PreferenceBasedAlg):
    def __init__(self, env, step_size=0.1, r_baseline=0.0, *args, **kwargs):
        super(GradientBanditAlg, self).__init__(env, step_size, *args, **kwargs)
        # R_t_bar over t time, baseline
        self.r_baseline_ori = r_baseline
        self.r_baseline = self.r_baseline_ori
        self.name = 'H-Preference Gradient Bandit Algorithm (a=%.2f, r=%.2f)' % (self.alpha, self.r_baseline)

    def reset(self):
        super(GradientBanditAlg, self).reset()
        self.r_baseline = self.r_baseline_ori

    def _update(self, action, action_reward, i_step):
        self.ntimes[action] = self.ntimes[action] + 1

        # update H_t+1
        self.vals[action] = self.vals[action] + self.alpha * (action_reward - self.r_baseline) * (1 - self.probabilities[action])
        self.vals[:action] = self.vals[:action] - self.alpha * (action_reward - self.r_baseline) * self.probabilities[:action]
        self.vals[action+1:] = self.vals[action+1:] - self.alpha * (action_reward - self.r_baseline) * self.probabilities[action+1:]
        
        # update R_t+1_bar
        self.r_baseline = self.r_baseline + 1 / (i_step+1) * (action_reward - self.r_baseline)
