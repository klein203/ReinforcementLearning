from keras.models import Sequential
from keras.layers import Dense


class DeepQNetwork(Sequential):
    def __init__(self, input_dim: int, output_dim: int):
        self._build_model(input_dim, output_dim)

    def _build_model(self, input_dim: int, output_dim: int):
        # fc: input_dim + 1 x 64
        self.add(Dense(64, input_dim=input_dim, activation='relu'))
        # fc: 64 + 1 x 64
        self.add(Dense(64, activation='relu'))
        # fc: 64 + 1 x out_dim
        self.add(Dense(output_dim))
