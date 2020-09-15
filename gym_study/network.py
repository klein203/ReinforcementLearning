from keras import models
from keras import layers
import tensorflow as tf
    

class DQNNetworkLayer(layers.Layer):
    def __init__(self, output_dim: int = 32, **kwargs):
        super(DQNNetworkLayer, self).__init__(**kwargs)
        self.inner_layer_1 = layers.Dense(64, activation="relu")
        self.inner_layer_2 = layers.Dense(64, activation="relu")
        self.inner_layer_3 = layers.Dense(output_dim)
    
    def call(self, inputs: layers.Input):
        x = self.inner_layer_1(inputs)
        x = self.inner_layer_2(x)
        return self.inner_layer_3(x)


class DQNLossLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(DQNLossLayer, self).__init__(**kwargs)
    
    def call(self, inputs: layers.Input, mask: layers.Input, importance_sampling: layers.Input, targets: layers.Input):
        err = (inputs - targets) * mask * importance_sampling
        self.add_loss(tf.reduce_mean(err))
        return err


class DeepQNetwork():
    def __init__(self, input_dim: int, output_dim: int):
        super(DeepQNetwork, self).__init__()
        self._build_model(input_dim, output_dim)

    def _build_model(self, input_dim: int, output_dim: int):
        inputs = layers.Input((input_dim,), name="Inputs")
        mask = layers.Input((output_dim,), name="Mask")
        # importance_sampling = layers.Input((1,), name="IS")
        targets = layers.Input((output_dim,), name="Targets")

        self.logits = DQNNetworkLayer(output_dim, name="DQN_NN")(inputs)
        # self.mask_logits
        # self.loss = DQNLossLayer(name="DQN_Loss")(self.logits, mask, importance_sampling, targets)
        self.loss = DQNLossLayer(name="DQN_Loss")(self.logits, mask, targets)

        self.model = models.Model(
            # inputs=[inputs, mask, importance_sampling, targets],
            inputs=[inputs, mask, targets],
            outputs=self.loss
        )
    
    def fit(self, x, y, batch_size=32, validation_split=0.2):
        return self.model.fit(x, y, batch_size=batch_size, validation_split=validation_split)
    
    def predict_prob(self, x):
        model = models.Model(
            inputs=self.model.inputs,
            outputs=self.logits
        )
        return model.predict(x)
    
    def predict(self, x):
        return self.model.predict(x)
