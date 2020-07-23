import logging
import numpy as np


class MultiArmedBanditEnv(object):
    def __init__(self, actions_space):
        self.name = 'MultiArmedBanditEnv'
        self.actions_space = actions_space
        self.k = len(actions_space)
        
        # q*(a) ~ N(0, 1)
        self.reward_target_mean = np.random.standard_normal(self.k)
        self.reward_target_var = np.ones(self.k)

    def get_reward(self, action_idx, size=None):
        # R(a) ~ N(q*(a), 1)
        loc = self.reward_target_mean[action_idx]
        scale = self.reward_target_var[action_idx]
        return np.random.normal(loc=loc, scale=scale, size=size)
    
    def get_k(self):
        return self.k
    
    def get_actions_space(self):
        return self.actions_space

    def get_reward_target_mean(self):
        return self.reward_target_mean
        
    def get_reward_target_var(self):
        return self.reward_target_var
    
    def info(self):
        logging.info('------------------------------------------------------------')
        logging.info('%s Reward Target Distribution' % self.name)
        logging.info('------------------------------------------------------------')
        for target in zip(self.actions_space, self.reward_target_mean, self.reward_target_var):
            logging.info("%s R ~ N(%.4f, %.2f)" % (target[0], target[1], target[2]))
