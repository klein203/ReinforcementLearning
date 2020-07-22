import logging
import numpy as np


class MultiArmedBanditEnv(object):
    def __init__(self, actions_space):
        self.actions_space = actions_space
        self.k = len(actions_space)
        
        # r_mean_target ~ N(0, 1)
        self.actions_reward_mean_target = np.random.standard_normal(self.k)
        self.actions_reward_var_target = np.ones(self.k)

    def get_reward(self, action_idx):
        # q_target ~ N(r_mean_target, 1.0)
        loc = self.actions_reward_mean_target[action_idx]
        scale = self.actions_reward_var_target[action_idx]
        return np.random.normal(loc=loc, scale=scale)
    
    def get_k(self):
        return self.k
    
    def get_actions_space(self):
        return self.actions_space

    def get_actions_reward_mean_target(self):
        return self.actions_reward_mean_target
        
    def get_actions_reward_var_target(self):
        return self.actions_reward_var_target
    
    def info(self):
        logging.info('Q Mean Target Value')
        logging.info(self.get_actions_reward_mean_target())
        logging.info('Q Var Target Value')
        logging.info(self.get_actions_reward_var_target())


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
