import logging
import numpy as np


class MultiArmedBanditEnv(object):
    def __init__(self, actions_space):
        self.name = 'MultiArmedBanditEnv'
        self.actions_space = actions_space
        self.k = len(actions_space)
        
        # r_mean_target ~ N(0, 1)
        self.reward_target_mean = np.random.standard_normal(self.k)
        self.reward_target_var = np.ones(self.k)

    def get_reward(self, action_idx):
        # q_target ~ N(r_mean_target, 1.0)
        loc = self.reward_target_mean[action_idx]
        scale = self.reward_target_var[action_idx]
        return np.random.normal(loc=loc, scale=scale)
    
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
        logging.info('%s Action Target Distribution' % self.name)
        logging.info('------------------------------------------------------------')
        for target in zip(self.actions_space, self.reward_target_mean, self.reward_target_var):
            logging.info("%s R ~ N (%.4f, %.2f)" % (target[0], target[1], target[2]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    actions_space = ['Arm%d' % i for i in range(1, 6)]
    env = MultiArmedBanditEnv(actions_space)
    env.info()

    r = []
    for i in range(1000):
        r.append(env.get_reward(1))
    print(np.mean(r))
    print(np.var(r))
