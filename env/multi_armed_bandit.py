import numpy as np


class MultiArmedBanditEnv(object):
    def __init__(self, actions_space, size=2000, mu=0.0, sigma=1.0):
        self.actions_space = actions_space
        self.k = len(actions_space)
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.k_dist = np.random.normal(loc=self.mu, scale=self.sigma, size=(self.k, self.size))

    def reward(self, action):
        action_idx = self.actions_space.index(action)
        return np.random.choice(self.k_dist[action_idx], 1)[0]


if __name__ == "__main__":
    actions_space = ['arm1', 'arm2', 'arm3', 'arm4']
    env = MultiArmedBanditEnv(actions_space, size=100)
    
    print(env.k_dist)

    action = 'arm1'
    print(env.reward(action))
    print(env.reward(action) in env.k_dist[actions_space.index(action)])
    