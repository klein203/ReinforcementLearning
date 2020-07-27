import logging
import numpy as np
import pandas as pd


class AbstractMultiArmedBanditEnv(object):
    def __init__(self, actions_space):
        self.name = 'AbstractMultiArmedBanditEnv'
        self.actions_space = actions_space
        self.k = len(actions_space)

    def get_reward(self, action_idx):
        raise NotImplementedError()
    
    def get_k(self):
        return self.k
    
    def get_actions_space(self):
        return self.actions_space
    
    def sampling(self, size):
        raise NotImplementedError()
    
    def info(self):
        raise NotImplementedError()


class MultiArmedBanditEnv(AbstractMultiArmedBanditEnv):
    def __init__(self, actions_space):
        super(MultiArmedBanditEnv, self).__init__(actions_space)
        self.name = 'MultiArmedBanditEnv'

        # q*(a) ~ N(0, 1)
        self.reward_target_dist_mean = np.random.standard_normal(self.k)
        self.reward_target_dist_var = np.ones(self.k)

    def set_reward_target_dist_mean(self, mean=0.0, var=1.0):
        self.reward_target_dist_mean = np.random.normal(loc=mean, scale=var, size=self.k)
    
    def get_reward_target_expectation(self):
        return self.reward_target_dist_mean

    def get_reward(self, action_idx):
        # R(a) ~ N(q*(a), 1)
        loc = self.reward_target_dist_mean[action_idx]
        scale = self.reward_target_dist_var[action_idx]
        return np.random.normal(loc=loc, scale=scale)
    
    def sampling(self, size=100):
        data = np.zeros((size, self.k))
        for i in range(self.k):
            loc = self.reward_target_dist_mean[i]
            scale = self.reward_target_dist_var[i]
            data[:, i] = np.random.normal(loc=loc, scale=scale, size=size)
        
        return pd.DataFrame(data=data, columns=self.actions_space)
    
    def info(self):
        logging.info('------------------------------------------------------------')
        logging.info('%s Reward Target Distribution' % self.name)
        logging.info('------------------------------------------------------------')
        logging.info('R_seed(a) ~ N(0, 1)')
        logging.info('R(a) ~ N(R_seed(a), 1)')
        for target in zip(self.actions_space, self.reward_target_dist_mean, self.reward_target_dist_var):
            logging.info("%s R ~ N(%.4f, %.2f)" % (target[0], target[1], target[2]))


class NonstationaryMultiArmedBanditEnv(AbstractMultiArmedBanditEnv):
    def __init__(self, actions_space):
        super(NonstationaryMultiArmedBanditEnv, self).__init__(actions_space)
        self.name = 'NonstationaryMultiArmedBanditEnv'
        self.reward_target = np.zeros(self.k)

    @property
    def _random_delta(self):
        return np.random.normal(loc=0, scale=0.01, size=self.k)
    
    def get_reward(self, action_idx):
        self.reward_target += self._random_delta
        return self.reward_target[action_idx]

    def sampling(self, size=100):
        data = np.zeros((size, self.k))
        for i in range(1, size):
            data[i, :] = data[i-1, :] + self._random_delta
        
        return pd.DataFrame(data=data, columns=self.actions_space)
    
    def info(self):
        logging.info('------------------------------------------------------------')
        logging.info('%s Reward Target Distribution' % self.name)
        logging.info('------------------------------------------------------------')
        logging.info('R_delta ~ N(0, 0.01)')
        logging.info('R(A_t+1) <- R(A_t) + R_delta')
