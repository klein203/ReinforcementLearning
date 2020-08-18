import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from keras.losses import MSE


class NN(object):
    def __init__(self, input_dim):
        self._build_model(input_dim)

    def _build_model(self, input_dim):
        model = Sequential()
        # fc: input_dim + 1 x 128
        model.add(Dense(128, input_dim=input_dim, activation='tanh'))
        # fc: 128 + 1 x 128
        model.add(Dense(128, activation='tanh'))
        # fc: 128 + 1 x 128
        model.add(Dense(128, activation='tanh'))
        # fc: 128 + 1 x 2
        model.add(Dense(2, activation='linear'))
        self.model = model

class DQNAgent(object):
    def __init__(self):
        self.target_net = NN(4)
        self.eval_net = NN(4)
        
        self.gamma = 0.9  # decay rate

        self.learning_rate = 1e-4
        self.optimizers = RMSprop(self.learning_rate)
        self.loss_func = MSE

if __name__ == "__main__":
    agent = DQNAgent()
    x = np.random.standard_normal((1, 4))
    y = agent.eval_net.model.predict(x)
    print(y)
    # agent.eval_net.model.summary()
    print(agent.eval_net.model.layers[-1].output)
